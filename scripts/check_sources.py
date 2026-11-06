"""Structural check of ``sources/sources.csv``.

Confirms the source list is well-formed, only uses allow-listed official
domains, and does not exceed ``corpus.max_pages``. The fetch metadata columns
(``http_status`` / ``sha256`` / ``fetched_at``) are filled by
``python -m ingest.run fetch``; fact coverage is checked by
``data/02_clean/parse_report.json`` after ``python -m ingest.run parse``.

Usage:
    python scripts/check_sources.py
    python scripts/check_sources.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import ConfigError, load_settings  # noqa: E402

REQUIRED_COLUMNS = [
    "id",
    "role",
    "scheme",
    "url",
    "page_type",
    "priority",
    "http_status",
    "sha256",
    "fetched_at",
    "notes",
]
PAGE_TYPES = {"help", "scheme", "amc", "education", "process", "factsheet"}


def host_allowed(host: str, domains: list[str]) -> bool:
    host = host.lower()
    return any(host == d or host.endswith("." + d) for d in domains)


def check(csv_path: Path, allowed_domains: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    errors: list[str] = []

    if not csv_path.exists():
        return [f"sources file not found: {csv_path}"], []

    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames or []

    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        return [f"missing columns: {', '.join(missing)}"], rows

    seen_ids: set[str] = set()
    seen_urls: set[str] = set()

    for index, row in enumerate(rows, start=2):  # line 2 = first data row
        where = f"line {index} (id={row.get('id', '?')})"

        for column in REQUIRED_COLUMNS:
            if row.get(column) is None:
                errors.append(f"{where}: empty column '{column}'")

        row_id = (row.get("id") or "").strip()
        if not row_id:
            errors.append(f"{where}: missing id")
        elif row_id in seen_ids:
            errors.append(f"{where}: duplicate id '{row_id}'")
        seen_ids.add(row_id)

        url = (row.get("url") or "").strip()
        parsed = urlparse(url)
        if parsed.scheme != "https":
            errors.append(f"{where}: url must be https ({url!r})")
        if not parsed.netloc:
            errors.append(f"{where}: url has no host ({url!r})")
        elif not host_allowed(parsed.netloc, allowed_domains):
            errors.append(
                f"{where}: domain '{parsed.netloc}' is not in allowed_domains "
                f"({', '.join(allowed_domains)})"
            )
        if url in seen_urls:
            errors.append(f"{where}: duplicate url {url}")
        seen_urls.add(url)

        page_type = (row.get("page_type") or "").strip()
        if page_type not in PAGE_TYPES:
            errors.append(
                f"{where}: page_type '{page_type}' not in {sorted(PAGE_TYPES)}"
            )

        try:
            int((row.get("priority") or "").strip())
        except ValueError:
            errors.append(f"{where}: priority must be an integer")

    return errors, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the source list.")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args()

    try:
        settings = load_settings(
            args.config, ensure_dirs=False, require_provider_key=False
        )
    except ConfigError as exc:
        print(f"CONFIG ERROR\n{exc}", file=sys.stderr)
        return 2

    errors, rows = check(settings.paths.sources_csv, settings.corpus.allowed_domains)

    print(f"file      : {settings.paths.sources_csv}")
    print(f"rows      : {len(rows)}")
    print(f"domains   : {', '.join(settings.corpus.allowed_domains)}")
    print()
    print(f"{'id':<4} {'page_type':<10} {'status':<8} {'host':<22} role")
    print("-" * 78)
    for row in rows:
        host = urlparse(row.get("url") or "").netloc
        status = (row.get("http_status") or "").strip() or "-"
        print(
            f"{(row.get('id') or '?'):<4} {str(row.get('page_type') or '?'):<10} "
            f"{status:<8} {host:<22} {row.get('role') or '?'}"
        )

    print()
    if len(rows) > settings.corpus.max_pages:
        print(
            f"WARN: {len(rows)} rows exceeds corpus.max_pages "
            f"({settings.corpus.max_pages}); trim the list or raise the cap."
        )
    missing_meta = [row for row in rows if not (row.get("http_status") or "").strip()]
    if missing_meta:
        print(
            f"NOTE: {len(missing_meta)} row(s) have no http_status yet; "
            "run 'python -m ingest.run fetch'."
        )
    print(
        "NOTE: fact coverage is reported in "
        "data/02_clean/parse_report.json after 'python -m ingest.run parse'."
    )

    if errors:
        print(f"\nFAILED with {len(errors)} problem(s):", file=sys.stderr)
        for message in errors:
            print(f"  - {message}", file=sys.stderr)
        return 1

    print("\nOK: source list is structurally valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
