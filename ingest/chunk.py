"""Stage 3: turn validated documents into fact atoms + section windows.

Input is ``data/02_clean/documents.jsonl`` only (the load gate's output). Two
chunk shapes, both carrying full provenance:

``fact_atom``
    One label/value fact per chunk (``Expense ratio: 0.74%``). These are the
    retrieval-first units because a citation must map 1:1 to a fact.

``section_window``
    A 700-char (config) window inside one section, prefix with the section
    title. Context for prose (objective, AMC background) and for disambiguating
    terse atoms. Overlap is measured in whole lines/sentences so a label is
    never separated from its value and a table row is never split.

Atoms are exempt from ``chunking.min_chunk_chars``: ``Exit load: Nil`` is short
but is exactly the answer a user needs. The floor applies to windows only.

Artifacts
---------
``data/03_chunks/chunks.jsonl``   deterministic ``Chunk`` rows
``data/03_chunks/chunk_stats.json`` counts, char spread, coverage matrix,
                                    duplicates, skipped sections, missing
                                    required fields, strategy rationale
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import ConfigError, Settings
from core.fields import FIELD_SPECS, FieldSpec, match_field, normalise_label, required_fields
from core.types import Chunk, ChunkType, LoadedDoc, Section

CHUNKER_VERSION = "chunk_v1"

_SKIP_SECTION_TITLES = (
    "disclaimer",
    "terms",
    "privacy",
    "related funds",
    "about groww",
    "returns",
    "holdings",
    "compare",
)
_REJECT_VALUES = {"", "na", "n/a", "-", "--", "—", "none", "null", "nan"}
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_DIGIT_RE = re.compile(r"\d")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_ORDERED_STEP_RE = re.compile(r"^\d+\.\s+")
_LABEL_INDEX = {
    normalise_label(alias)
    for spec in FIELD_SPECS.values()
    for alias in (spec.key, spec.label, *spec.aliases)
}


@dataclass(slots=True)
class ChunkResult:
    chunks: list[Chunk] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    documents: int = 0
    duplicate_count: int = 0
    skipped_sections: list[dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def chunk_all(settings: Settings, *, write: bool = True) -> ChunkResult:
    """Read validated documents from disk, chunk them, and persist artifacts."""
    documents_path = settings.paths.data_dir / "02_clean" / "documents.jsonl"
    if not documents_path.exists():
        raise ConfigError(
            f"{documents_path} not found; run 'python -m ingest.run load' first"
        )

    docs = _read_documents(documents_path)
    if not docs:
        raise ConfigError(
            f"{documents_path} has no documents; run 'python -m ingest.run load'"
        )

    result = ChunkResult(documents=len(docs))
    seen: dict[str, str] = {}
    collected: list[Chunk] = []

    for doc in docs:
        for chunk in _chunk_document(doc, settings, result.skipped_sections):
            if chunk.chunk_id in seen:
                result.duplicate_count += 1
                continue
            seen[chunk.chunk_id] = chunk.text
            collected.append(chunk)

    order = {doc.scheme_slug: index for index, doc in enumerate(docs)}
    collected.sort(
        key=lambda chunk: (
            order.get(chunk.scheme_slug, 10_000),
            0 if chunk.chunk_type == "fact_atom" else 1,
            chunk.field,
            chunk.text,
        )
    )
    result.chunks = collected
    result.stats = _build_stats(
        settings,
        docs,
        collected,
        duplicate_count=result.duplicate_count,
        skipped_sections=result.skipped_sections,
    )

    if write:
        _write_artifacts(settings.paths.data_dir / "03_chunks", result)
    return result


def chunk_document(
    doc: LoadedDoc, settings: Settings
) -> tuple[list[Chunk], list[dict[str, str]]]:
    """Convenience wrapper used by tests: atoms + windows for one document."""
    skipped: list[dict[str, str]] = []
    chunks = _chunk_document(doc, settings, skipped)
    return chunks, skipped


# ---------------------------------------------------------------------------
# document input
# ---------------------------------------------------------------------------


def _read_documents(path: Path) -> list[LoadedDoc]:
    docs: list[LoadedDoc] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc
        docs.append(_document_from_payload(payload, path, line_no))
    return docs


def _document_from_payload(payload: dict[str, Any], path: Path, line_no: int) -> LoadedDoc:
    required = (
        "doc_id",
        "source_id",
        "scheme",
        "scheme_slug",
        "source_title",
        "source_url",
        "page_type",
        "fetched_at",
        "sections",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ConfigError(
            f"{path}:{line_no}: missing fields: {', '.join(missing)}; "
            "re-run 'python -m ingest.run load'"
        )

    raw_sections = payload["sections"]
    if not isinstance(raw_sections, list) or not raw_sections:
        raise ConfigError(
            f"{path}:{line_no}: no sections; re-run 'python -m ingest.run load'"
        )
    sections: list[Section] = []
    for index, item in enumerate(raw_sections):
        if not isinstance(item, dict):
            raise ConfigError(
                f"{path}:{line_no}: section {index} is not an object; "
                "re-run 'python -m ingest.run load'"
            )
        try:
            sections.append(
                Section(
                    title=str(item["title"]).strip(),
                    body=str(item["body"]).strip(),
                    level=int(item.get("level", 2)),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(
                f"{path}:{line_no}: section {index} is malformed; "
                "re-run 'python -m ingest.run load'"
            ) from exc

    return LoadedDoc(
        doc_id=str(payload["doc_id"]),
        source_id=str(payload["source_id"]),
        scheme=str(payload["scheme"]),
        scheme_slug=str(payload["scheme_slug"]),
        source_title=str(payload["source_title"]),
        source_url=str(payload["source_url"]),
        page_type=str(payload["page_type"]),  # type: ignore[arg-type]
        fetched_at=str(payload["fetched_at"]),
        sections=sections,
        raw_sha256=str(payload.get("raw_sha256") or ""),
        content_sha256=str(payload.get("content_sha256") or ""),
        char_len=int(payload.get("char_len") or 0),
    )


# ---------------------------------------------------------------------------
# chunk construction
# ---------------------------------------------------------------------------


def _chunk_document(
    doc: LoadedDoc, settings: Settings, skipped: list[dict[str, str]]
) -> list[Chunk]:
    cfg = settings.chunking
    atoms = _atoms(doc, cfg.atom_max_chars)
    windows = _windows(doc, cfg, skipped)
    return atoms + windows


def _atoms(doc: LoadedDoc, atom_max_chars: int) -> list[Chunk]:
    atoms: list[Chunk] = []
    for section in doc.sections:
        if _skip_section(section.title):
            continue
        for spec, value in _section_pairs(section, doc.page_type):
            if not _plausible(spec, value, atom_max_chars):
                continue
            text = f"{spec.label}: {value}"
            if len(text) > atom_max_chars:
                continue
            atoms.append(
                _make_chunk(
                    doc,
                    chunk_type="fact_atom",
                    field=spec.key,
                    section_title=section.title,
                    text=text,
                )
            )
    return atoms


def _section_pairs(section: Section, page_type: str) -> list[tuple[FieldSpec, str]]:
    """Extract matched label/value pairs, falling back to a field-titled section.

    The fallback is what captures ``## Exit load`` whose body is just
    ``Exit load of 1% if redeemed within 1 year.`` (no colon). It only runs when
    the section has no labelled pairs, so prose paragraphs are not mined blindly.
    """
    pairs: list[tuple[FieldSpec, str]] = []
    unmatched = False

    for line in section.body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if len(cells) >= 2:
                spec = match_field(cells[0], page_type)
                if spec is not None:
                    pairs.append((spec, _clean_value(" | ".join(cells[1:]))))
                else:
                    unmatched = True
            continue
        if ":" in stripped:
            label, value = stripped.split(":", 1)
            if len(label) > 80:
                unmatched = True
                continue
            spec = match_field(label, page_type)
            if spec is not None:
                pairs.append((spec, _clean_value(value)))
            else:
                unmatched = True

    if pairs:
        return pairs
    if unmatched:
        return []

    spec = match_field(section.title, page_type)
    if spec is None:
        return []
    value = _strip_leading_label(section.body.strip(), section.title, spec)
    return [(spec, _clean_value(value))]


def _strip_leading_label(value: str, section_title: str, spec: FieldSpec) -> str:
    candidates = {section_title, spec.label, *spec.aliases}
    for candidate in sorted(candidates, key=len, reverse=True):
        pattern = rf"^{re.escape(candidate)}\s*(?:of|is|:|-|–|—)\s*"
        stripped = re.sub(pattern, "", value, count=1, flags=re.IGNORECASE)
        if stripped != value:
            return stripped.strip()
    return value


def _plausible(spec: FieldSpec, value: str, atom_max_chars: int) -> bool:
    if not value or len(value) > atom_max_chars:
        return False
    if normalise_label(value) in _REJECT_VALUES:
        return False
    if _URL_RE.search(value):
        return False
    # A value that is itself a label means the parser caught a header row.
    if normalise_label(value) in _LABEL_INDEX:
        return False

    low = value.lower()
    key = spec.key
    if key == "expense_ratio":
        return "%" in value and bool(_DIGIT_RE.search(value))
    if key == "exit_load":
        return bool(_DIGIT_RE.search(value)) or "nil" in low
    if key == "lock_in":
        return bool(_DIGIT_RE.search(value)) or "lock" in low
    if key == "riskometer":
        return "risk" in low or bool(re.search(r"[A-Za-z]", value))
    if key == "benchmark":
        return bool(re.search(r"[A-Za-z]", value)) and "benchmark" not in low
    if key == "category":
        return bool(re.search(r"[A-Za-z]", value)) and "category" not in low
    if key == "amc_name":
        return bool(re.search(r"[A-Za-z]", value))
    if key in {"incorporation_date", "launch_date"}:
        return bool(_YEAR_RE.search(value))
    if spec.numeric:
        return bool(_DIGIT_RE.search(value))
    return True


def _windows(
    doc: LoadedDoc, cfg: Any, skipped: list[dict[str, str]]
) -> list[Chunk]:
    windows: list[Chunk] = []
    for section in doc.sections:
        if _skip_section(section.title):
            skipped.append(
                {
                    "scheme_slug": doc.scheme_slug,
                    "section": section.title,
                    "reason": "noise_section",
                }
            )
            continue
        for text in _section_windows(section, cfg.window_chars, cfg.window_overlap):
            if len(text) < cfg.min_chunk_chars:
                skipped.append(
                    {
                        "scheme_slug": doc.scheme_slug,
                        "section": section.title,
                        "reason": f"window_too_short: {len(text)} < {cfg.min_chunk_chars}",
                    }
                )
                continue
            windows.append(
                _make_chunk(
                    doc,
                    chunk_type="section_window",
                    field="",
                    section_title=section.title,
                    text=text,
                )
            )
    return windows


def _section_windows(section: Section, window_chars: int, overlap: int) -> list[str]:
    prefix = f"{section.title} - "
    units = _units(section.body, max(window_chars - len(prefix) - 2, 40))
    if not units:
        return []

    windows: list[str] = []
    index = 0
    while index < len(units):
        head = _overlap_units(units, index, overlap) if index > 0 else []
        picked: list[str] = []
        cursor = index
        while cursor < len(units):
            candidate = _render(prefix, head + picked + [units[cursor]])
            if len(candidate) > window_chars:
                if head:
                    head.pop(0)  # shrink the overlap before dropping new content
                    continue
                if picked:
                    break
            picked.append(units[cursor])
            cursor += 1
        windows.append(_render(prefix, head + picked))
        index = cursor
    return windows


def _units(body: str, limit: int) -> list[str]:
    units: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("### "):
            line = line[4:].strip()
        if line.startswith("|") and line.endswith("|"):
            units.append(line)  # a table row is never split
            continue
        if _ORDERED_STEP_RE.match(line):
            # Keep a numbered step together: "1." must not become its own unit.
            units.extend(_hard_wrap(line, limit))
            continue
        if _looks_like_pair(line):
            units.extend(_hard_wrap(line, limit))  # keep label + value together
            continue
        for sentence in _SENTENCE_RE.split(line):
            sentence = sentence.strip()
            if sentence:
                units.extend(_hard_wrap(sentence, limit))
    return units


def _looks_like_pair(line: str) -> bool:
    if ":" not in line:
        return False
    label, value = line.split(":", 1)
    return len(label.strip()) <= 80 and bool(value.strip())


def _hard_wrap(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    current: list[str] = []
    for word in text.split():
        if current and len(" ".join(current + [word])) > limit:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces or [text]


def _overlap_units(units: list[str], index: int, overlap: int) -> list[str]:
    if overlap <= 0:
        return []
    picked: list[str] = []
    total = 0
    for unit in reversed(units[:index]):
        if total + len(unit) + 1 > overlap:
            break
        picked.append(unit)
        total += len(unit) + 1
    return list(reversed(picked))


def _render(prefix: str, units: list[str]) -> str:
    return prefix + "\n".join(units)


def _make_chunk(
    doc: LoadedDoc,
    *,
    chunk_type: ChunkType,
    field: str,
    section_title: str,
    text: str,
) -> Chunk:
    return Chunk(
        chunk_id=_chunk_id(doc.scheme_slug, field, text),
        chunk_type=chunk_type,
        scheme=doc.scheme,
        scheme_slug=doc.scheme_slug,
        field=field,
        section_title=section_title,
        text=text,
        source_url=doc.source_url,
        source_title=doc.source_title,
        page_type=doc.page_type,
        fetched_at=doc.fetched_at,
        char_len=len(text),
    )


def _chunk_id(scheme_slug: str, field: str, text: str) -> str:
    blob = f"{scheme_slug}|{field}|{text.strip()}".encode()
    return "sha1:" + hashlib.sha1(blob).hexdigest()[:20]


def _clean_value(value: str) -> str:
    text = value.replace("**", "").replace("\u00a0", " ").strip(" \t-")
    return re.sub(r"\s+", " ", text)


def _skip_section(title: str) -> bool:
    lowered = title.strip().lower()
    return any(noise in lowered for noise in _SKIP_SECTION_TITLES)


# ---------------------------------------------------------------------------
# stats + artifacts
# ---------------------------------------------------------------------------


def _build_stats(
    settings: Settings,
    docs: list[LoadedDoc],
    chunks: list[Chunk],
    *,
    duplicate_count: int,
    skipped_sections: list[dict[str, str]],
) -> dict[str, Any]:
    cfg = settings.chunking
    by_type = Counter(chunk.chunk_type for chunk in chunks)
    chars_by_type = {
        chunk_type: _char_summary(
            [chunk.char_len for chunk in chunks if chunk.chunk_type == chunk_type]
        )
        for chunk_type in ("fact_atom", "section_window")
    }

    coverage: dict[str, Any] = {}
    missing: list[dict[str, str]] = []
    for doc in docs:
        atoms = [chunk for chunk in chunks if chunk.scheme_slug == doc.scheme_slug and chunk.chunk_type == "fact_atom"]
        windows = [chunk for chunk in chunks if chunk.scheme_slug == doc.scheme_slug and chunk.chunk_type == "section_window"]
        field_counts = dict(sorted(Counter(chunk.field for chunk in atoms).items()))
        coverage[doc.scheme_slug] = {
            "scheme": doc.scheme,
            "page_type": doc.page_type,
            "source_url": doc.source_url,
            "fetched_at": doc.fetched_at,
            "atoms": len(atoms),
            "windows": len(windows),
            "fields": field_counts,
        }
        for field_key in required_fields(doc.page_type, doc.scheme_slug):
            if field_counts.get(field_key, 0) == 0:
                missing.append(
                    {
                        "scheme_slug": doc.scheme_slug,
                        "scheme": doc.scheme,
                        "field": field_key,
                    }
                )

    oversized = [
        chunk.chunk_id
        for chunk in chunks
        if (chunk.chunk_type == "fact_atom" and chunk.char_len > cfg.atom_max_chars)
        or (chunk.chunk_type == "section_window" and chunk.char_len > cfg.window_chars)
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunker_version": CHUNKER_VERSION,
        "strategy": cfg.strategy,
        "config_hash": settings.config_hash,
        "input": {
            "path": str(settings.paths.data_dir / "02_clean" / "documents.jsonl"),
            "documents": len(docs),
            "sections": sum(len(doc.sections) for doc in docs),
            "chars": sum(doc.char_len for doc in docs),
        },
        "counts": {
            "total": len(chunks),
            "fact_atom": by_type.get("fact_atom", 0),
            "section_window": by_type.get("section_window", 0),
        },
        "chars": {
            "fact_atom": chars_by_type["fact_atom"],
            "section_window": chars_by_type["section_window"],
            "all": _char_summary([chunk.char_len for chunk in chunks]),
        },
        "coverage": coverage,
        "required": {
            doc.scheme_slug: list(required_fields(doc.page_type, doc.scheme_slug))
            for doc in docs
        },
        "missing_required": missing,
        "duplicates": duplicate_count,
        "oversized": oversized,
        "skipped_sections": skipped_sections,
        "settings": {
            "atom_max_chars": cfg.atom_max_chars,
            "window_chars": cfg.window_chars,
            "window_overlap": cfg.window_overlap,
            "min_chunk_chars": cfg.min_chunk_chars,
        },
        "rationale": (
            "structure_aware_hybrid: fact_atom chunks keep one label/value fact "
            "with its source, because fixed-size windows split 'Exit load' from "
            "its value; section_window chunks add bounded, title-prefixed context "
            "for prose and ambiguous questions. Atoms are exempt from "
            "min_chunk_chars so terse facts ('Exit load: Nil') survive; the floor "
            "applies to windows only."
        ),
    }


def _char_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "p90": None, "max": None}
    ordered = sorted(values)
    p90_index = max(math.ceil(0.9 * len(ordered)) - 1, 0)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": round(statistics.fmean(ordered), 1),
        "p90": ordered[p90_index],
        "max": ordered[-1],
    }


def _write_artifacts(chunk_dir: Path, result: ChunkResult) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(asdict(chunk), ensure_ascii=False, sort_keys=True)
        for chunk in result.chunks
    ]
    body = "\n".join(lines)
    (chunk_dir / "chunks.jsonl").write_text(
        body + ("\n" if body else ""), encoding="utf-8"
    )
    (chunk_dir / "chunk_stats.json").write_text(
        json.dumps(result.stats, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


__all__ = ["CHUNKER_VERSION", "ChunkResult", "chunk_all", "chunk_document"]
