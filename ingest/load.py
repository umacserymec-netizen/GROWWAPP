"""Stage 2c: load the cleaned page artifacts into validated documents.

``ingest/parse.py`` writes one ``.md`` + ``.json`` per page and a
``parse_report.json``. This module is the single gate between "files on disk"
and "documents the RAG may use":

- only pages marked ``ok=true`` in ``parse_report.json`` are loaded;
- every document is re-checked against ``sources.csv`` and the fetch manifest
  (slug, URL, domain, page type, ``fetched_at``, raw ``sha256``);
- sections must be non-empty strings and the document must still clear
  ``corpus.min_clean_chars``;
- exact duplicate content is dropped (first source wins) so the same fact
  cannot be retrieved twice under two citations.

Artifacts
---------
``data/02_clean/documents.jsonl``  canonical ordered ``LoadedDoc`` rows
``data/02_clean/load_report.json`` per-document provenance, rejections, summary

Chunking (Stage 3) must read ``documents.jsonl`` and nothing else, so an
unvalidated page can never leak into the index.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from config.settings import ConfigError, Settings
from core.types import LoadedDoc, Section, SourceRow
from ingest.fetch import load_sources, slug_for

LOADER_VERSION = "load_v1"

REQUIRED_DOC_FIELDS = (
    "scheme",
    "scheme_slug",
    "source_title",
    "source_url",
    "page_type",
    "fetched_at",
    "sections",
)
LOADABLE_PAGE_TYPES = ("help", "scheme", "amc")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class DocumentError(ValueError):
    """A clean artifact failed validation; the message is the report reason."""


@dataclass(slots=True)
class RejectedDoc:
    source_id: str
    slug: str
    reason: str


@dataclass(slots=True)
class LoadResult:
    documents: list[LoadedDoc] = field(default_factory=list)
    rejected: list[RejectedDoc] = field(default_factory=list)
    duplicates: list[RejectedDoc] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def load_documents(settings: Settings, *, write: bool = True) -> LoadResult:
    """Load every validated clean document and (optionally) write artifacts."""
    clean_dir = settings.paths.data_dir / "02_clean"
    parse_report_path = clean_dir / "parse_report.json"
    manifest_path = settings.paths.data_dir / "01_raw" / "manifest.json"

    if not parse_report_path.exists():
        raise ConfigError(
            f"{parse_report_path} not found; run 'python -m ingest.run parse' first"
        )
    if not manifest_path.exists():
        raise ConfigError(
            f"{manifest_path} not found; run 'python -m ingest.run fetch' first"
        )

    parse_report = json.loads(parse_report_path.read_text(encoding="utf-8"))
    manifest = {
        str(entry["id"]): entry
        for entry in json.loads(manifest_path.read_text(encoding="utf-8"))
    }
    sources = {row.id: row for row in load_sources(settings.paths.sources_csv)}

    result = LoadResult()
    seen: dict[str, str] = {}

    for page in parse_report.get("pages", []):
        source_id = str(page.get("id") or "")
        slug = str(page.get("slug") or "")
        source = sources.get(source_id)

        if not _safe_slug(slug):
            result.rejected.append(RejectedDoc(source_id, slug, "unsafe_slug"))
            continue
        if source is None:
            result.rejected.append(RejectedDoc(source_id, slug, "source_missing"))
            continue
        if not page.get("ok", False):
            reason = page.get("reason") or "not_ok"
            result.rejected.append(
                RejectedDoc(source_id, slug, f"quarantined: {reason}")
            )
            continue

        json_path = clean_dir / f"{slug}.json"
        md_path = clean_dir / f"{slug}.md"
        if not json_path.exists():
            result.rejected.append(RejectedDoc(source_id, slug, "clean_json_missing"))
            continue
        if not md_path.exists():
            result.rejected.append(
                RejectedDoc(source_id, slug, "clean_markdown_missing")
            )
            continue

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            result.rejected.append(
                RejectedDoc(source_id, slug, f"invalid_json: {exc.msg}")
            )
            continue

        try:
            doc = document_from_json(
                data,
                source=source,
                expected_slug=slug,
                manifest_entry=manifest.get(source_id),
                min_clean_chars=settings.corpus.min_clean_chars,
                allowed_domains=settings.corpus.allowed_domains,
            )
        except DocumentError as exc:
            result.rejected.append(RejectedDoc(source_id, slug, str(exc)))
            continue

        if doc.content_sha256 in seen:
            result.duplicates.append(
                RejectedDoc(
                    source_id,
                    slug,
                    f"duplicate_of: {seen[doc.content_sha256]}",
                )
            )
            continue

        seen[doc.content_sha256] = doc.doc_id
        result.documents.append(doc)

    priority = {row.id: row.priority for row in sources.values()}
    result.documents.sort(key=lambda doc: (priority.get(doc.source_id, 1), _id_key(doc)))

    result.payload = _payload(settings, result, clean_dir)
    if write:
        _write_artifacts(clean_dir, result)
    return result


def document_from_json(
    data: Any,
    *,
    source: SourceRow,
    expected_slug: str,
    manifest_entry: dict[str, Any] | None,
    min_clean_chars: int,
    allowed_domains: list[str],
) -> LoadedDoc:
    """Validate one ``<slug>.json`` payload. Pure: no filesystem access."""
    if not isinstance(data, dict):
        raise DocumentError("not_an_object")

    missing = [name for name in REQUIRED_DOC_FIELDS if name not in data]
    if missing:
        raise DocumentError("missing_fields: " + ", ".join(missing))

    slug = _text(data["scheme_slug"])
    if not _safe_slug(slug):
        raise DocumentError("unsafe_slug")
    if slug != expected_slug or slug != slug_for(source.url):
        raise DocumentError("slug_mismatch")

    page_type = _text(data["page_type"])
    if page_type not in LOADABLE_PAGE_TYPES:
        raise DocumentError(f"page_type_not_loadable: {page_type}")
    if page_type != source.page_type:
        raise DocumentError("page_type_mismatch")

    source_url = _text(data["source_url"])
    if _canonical_url(source_url) != _canonical_url(source.url):
        raise DocumentError("source_url_mismatch")
    if not _host_allowed(urlsplit(source_url).netloc, allowed_domains):
        raise DocumentError("domain_not_allowed")

    fetched_at = _text(data["fetched_at"])
    if not fetched_at:
        raise DocumentError("missing_fetched_at")
    raw_sha256 = _text(data.get("raw_sha256") or "")
    if manifest_entry is not None:
        if fetched_at != _text(manifest_entry.get("fetched_at")):
            raise DocumentError("fetched_at_mismatch")
        if raw_sha256 and raw_sha256 != _text(manifest_entry.get("sha256")):
            raise DocumentError("raw_sha256_mismatch")

    sections = _sections(data["sections"])
    char_len = sum(len(section.title) + len(section.body) for section in sections)
    declared = data.get("char_len")
    if declared is not None and int(declared) != char_len:
        raise DocumentError("char_len_mismatch")
    if char_len < min_clean_chars:
        raise DocumentError(f"too_short: {char_len} < {min_clean_chars}")

    scheme = _text(data["scheme"])
    if not scheme:
        raise DocumentError("missing_scheme")

    return LoadedDoc(
        doc_id=slug,
        source_id=source.id,
        scheme=scheme,
        scheme_slug=slug,
        source_title=_text(data["source_title"]) or scheme,
        source_url=source_url,
        page_type=page_type,  # type: ignore[arg-type]
        fetched_at=fetched_at,
        raw_sha256=raw_sha256,
        content_sha256=_content_hash(scheme, sections),
        char_len=char_len,
        sections=sections,
    )


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------


def _payload(
    settings: Settings, result: LoadResult, clean_dir: Path
) -> dict[str, Any]:
    documents = [
        {
            "doc_id": doc.doc_id,
            "source_id": doc.source_id,
            "scheme": doc.scheme,
            "scheme_slug": doc.scheme_slug,
            "page_type": doc.page_type,
            "section_count": len(doc.sections),
            "char_len": doc.char_len,
            "fetched_at": doc.fetched_at,
            "raw_sha256": doc.raw_sha256,
            "content_sha256": doc.content_sha256,
        }
        for doc in result.documents
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "loader_version": LOADER_VERSION,
        "config_hash": settings.config_hash,
        "clean_dir": str(clean_dir),
        "documents": documents,
        "rejected": [asdict(item) for item in result.rejected],
        "duplicates": [asdict(item) for item in result.duplicates],
        "summary": {
            "loaded": len(result.documents),
            "rejected": len(result.rejected),
            "duplicates": len(result.duplicates),
            "sections": sum(len(doc.sections) for doc in result.documents),
            "total_chars": sum(doc.char_len for doc in result.documents),
        },
        "checks": [
            "parse_report_ok",
            "sources_csv_provenance",
            "manifest_fetched_at",
            "manifest_raw_sha256",
            "domain_allow_list",
            "section_schema",
            "min_clean_chars",
            "content_dedupe",
        ],
    }


def _write_artifacts(clean_dir: Path, result: LoadResult) -> None:
    clean_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(asdict(doc), ensure_ascii=False, sort_keys=True)
        for doc in result.documents
    ]
    body = "\n".join(lines)
    (clean_dir / "documents.jsonl").write_text(
        body + ("\n" if body else ""), encoding="utf-8"
    )
    (clean_dir / "load_report.json").write_text(
        json.dumps(result.payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------


def _sections(raw: Any) -> list[Section]:
    if not isinstance(raw, list) or not raw:
        raise DocumentError("no_sections")
    sections: list[Section] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise DocumentError(f"section_{index}_not_an_object")
        title = _text(item.get("title") or "")
        body = _text(item.get("body") or "")
        if not title or not body:
            raise DocumentError(f"section_{index}_empty")
        level = item.get("level", 2)
        if not isinstance(level, int) or not 1 <= level <= 6:
            raise DocumentError(f"section_{index}_bad_level")
        sections.append(Section(title=title, body=body, level=level))
    return sections


def _content_hash(scheme: str, sections: list[Section]) -> str:
    payload = {
        "scheme": scheme,
        "sections": [[section.title, section.body] for section in sections],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _canonical_url(url: str) -> tuple[str, str, str]:
    parsed = urlsplit(url.strip())
    path = parsed.path.rstrip("/") or "/"
    return (parsed.netloc.lower(), path, parsed.query)


def _host_allowed(host: str, domains: list[str]) -> bool:
    host = host.lower()
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def _safe_slug(slug: str) -> bool:
    return (
        bool(slug)
        and ".." not in slug
        and "/" not in slug
        and "\\" not in slug
        and SLUG_RE.fullmatch(slug) is not None
    )


def _id_key(doc: LoadedDoc) -> tuple[int, str]:
    try:
        return (int(doc.source_id), doc.doc_id)
    except ValueError:
        return (10_000, doc.doc_id)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


__all__ = [
    "LOADER_VERSION",
    "DocumentError",
    "LoadResult",
    "RejectedDoc",
    "document_from_json",
    "load_documents",
]
