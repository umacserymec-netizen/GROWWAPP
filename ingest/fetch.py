"""Stage 2a: fetch the 5 briefed pages and keep an untouched snapshot.

Design rules
------------
- Only URLs present in ``sources/sources.csv`` are fetched. No link following.
- One failed URL never aborts the run; it is marked ``ok=false`` in the manifest.
- Raw bytes are saved before parsing so every later stage is reproducible.
- ``http_status`` / ``sha256`` / ``fetched_at`` are written back to
  ``sources.csv`` as the audit trail.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import date
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from config.settings import ConfigError, Settings
from core.types import ManifestEntry, SourceRow

RETRY_BACKOFF_SECONDS = (1.0, 3.0)
HTML_TYPES = ("text/html", "application/xhtml+xml")


def load_sources(csv_path: Path) -> list[SourceRow]:
    if not csv_path.exists():
        raise ConfigError(f"sources file not found: {csv_path}")
    rows: list[SourceRow] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                SourceRow(
                    id=(raw.get("id") or "").strip(),
                    role=(raw.get("role") or "").strip(),
                    scheme=(raw.get("scheme") or "").strip(),
                    url=(raw.get("url") or "").strip(),
                    page_type=(raw.get("page_type") or "scheme").strip(),  # type: ignore[arg-type]
                    priority=int((raw.get("priority") or "1").strip() or 1),
                    notes=(raw.get("notes") or "").strip(),
                )
            )
    if not rows:
        raise ConfigError(f"no rows in {csv_path}")
    return rows


def slug_for(url: str) -> str:
    """Stable file key. AMC pages get an ``amc-`` prefix to avoid collisions."""
    parts = [part for part in urlparse(url).path.split("/") if part]
    if not parts:
        return "root"
    slug = f"amc-{parts[-1]}" if "amc" in parts[:-1] else parts[-1]
    return re.sub(r"[^a-z0-9._-]+", "-", slug.lower()).strip("-")


def fetch_all(
    settings: Settings,
    *,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    today: str | None = None,
) -> list[ManifestEntry]:
    sources = load_sources(settings.paths.sources_csv)
    if len(sources) > settings.corpus.max_pages:
        raise ConfigError(
            f"sources.csv has {len(sources)} rows but corpus.max_pages="
            f"{settings.corpus.max_pages}; trim the list or raise the cap"
        )

    raw_dir = settings.paths.data_dir / "01_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetched_on = today or date.today().isoformat()

    own_client = client is None
    http = client or httpx.Client(
        headers={"User-Agent": settings.corpus.user_agent},
        timeout=settings.corpus.timeout_seconds,
        follow_redirects=True,
    )

    robots_cache: dict[str, RobotFileParser | None] = {}
    entries: list[ManifestEntry] = []

    try:
        for index, source in enumerate(sources):
            if index > 0 and settings.corpus.request_delay_seconds:
                sleep(settings.corpus.request_delay_seconds)

            entry = ManifestEntry(
                id=source.id,
                url=source.url,
                final_url=source.url,
                status=None,
                content_type="",
                bytes=0,
                sha256="",
                fetched_at=fetched_on,
                slug=slug_for(source.url),
                ok=False,
            )

            if settings.corpus.respect_robots and not _robots_allowed(
                http, source.url, settings, robots_cache
            ):
                entry.error = "robots_disallow"
                entries.append(entry)
                continue

            response, error = _get_with_retries(http, source.url, settings, sleep)
            if response is None:
                entry.error = error or "request_failed"
                entries.append(entry)
                continue

            entry.status = response.status_code
            entry.final_url = str(response.url)
            entry.content_type = response.headers.get("content-type", "")
            body = response.content
            entry.bytes = len(body)
            entry.sha256 = hashlib.sha256(body).hexdigest()

            if response.status_code != 200:
                entry.error = f"http_{response.status_code}"
            elif not any(kind in entry.content_type for kind in HTML_TYPES):
                entry.error = "not_html"
            elif not body:
                entry.error = "empty_body"
            else:
                entry.ok = True

            if settings.corpus.save_raw_html:
                (raw_dir / f"{entry.slug}.html").write_bytes(body)
                meta = asdict(entry) | {
                    "requested_url": source.url,
                    "headers": {
                        key: value
                        for key, value in response.headers.items()
                        if key.lower() in {"content-type", "content-length", "date", "server"}
                    },
                }
                (raw_dir / f"{entry.slug}.meta.json").write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
                )

            entries.append(entry)
    finally:
        if own_client:
            http.close()

    _write_manifest(raw_dir, entries)
    _update_sources_csv(settings.paths.sources_csv, entries)
    return entries


def _get_with_retries(
    client: httpx.Client,
    url: str,
    settings: Settings,
    sleep: Callable[[float], None],
) -> tuple[httpx.Response | None, str | None]:
    attempts = len(RETRY_BACKOFF_SECONDS) + 1
    last_error = "request_failed"
    for attempt in range(attempts):
        try:
            response = client.get(url)
        except httpx.TimeoutException:
            last_error = "timeout"
        except httpx.HTTPError as exc:
            last_error = f"network_error:{type(exc).__name__}"
        else:
            if response.status_code < 500 and response.status_code != 429:
                return response, None
            last_error = f"http_{response.status_code}"
            if attempt == attempts - 1:
                return response, last_error
        if attempt < len(RETRY_BACKOFF_SECONDS):
            sleep(RETRY_BACKOFF_SECONDS[attempt])
    return None, last_error


def _robots_allowed(
    client: httpx.Client,
    url: str,
    settings: Settings,
    cache: dict[str, RobotFileParser | None],
) -> bool:
    parsed = urlparse(url)
    host_key = f"{parsed.scheme}://{parsed.netloc}"
    if host_key not in cache:
        parser: RobotFileParser | None = RobotFileParser()
        try:
            response = client.get(f"{host_key}/robots.txt")
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            else:
                parser = None  # no readable robots -> treat as allowed
        except httpx.HTTPError:
            parser = None
        cache[host_key] = parser
    parser = cache[host_key]
    if parser is None:
        return True
    return parser.can_fetch(settings.corpus.user_agent, url)


def _write_manifest(raw_dir: Path, entries: list[ManifestEntry]) -> None:
    payload = [asdict(entry) for entry in sorted(entries, key=lambda item: item.id)]
    (raw_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _update_sources_csv(csv_path: Path, entries: list[ManifestEntry]) -> None:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    by_id = {entry.id: entry for entry in entries}
    for row in rows:
        entry = by_id.get((row.get("id") or "").strip())
        if entry is None:
            continue
        row["http_status"] = "" if entry.status is None else str(entry.status)
        row["sha256"] = entry.sha256
        row["fetched_at"] = entry.fetched_at

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["fetch_all", "load_sources", "slug_for"]
