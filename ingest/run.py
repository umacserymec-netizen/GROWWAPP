"""Ingest CLI: ``python -m ingest.run <stage>``.

Stages are independent so a failure can be fixed without re-scraping::

    python -m ingest.run fetch      # 01_raw
    python -m ingest.run parse      # 02_clean page artifacts
    python -m ingest.run load       # 02_clean/documents.jsonl (validated)
    python -m ingest.run fetch parse load  # all three, in order
    python -m ingest.run chunk      # 03_chunks   (Stage 3)
    python -m ingest.run embed      # 04_embeddings (Stage 4)
    python -m ingest.run index      # 05_chroma   (Stage 5)
    python -m ingest.run all        # every implemented stage, in order

A ``data/.lock`` file prevents two runs from writing the same artifacts. It is
removed on exit; delete it manually only if a previous run crashed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

from config.settings import ConfigError, Settings, load_settings

STAGES = ("fetch", "parse", "load", "chunk", "embed", "index")
IMPLEMENTED = ("fetch", "parse", "load", "chunk", "embed", "index")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingest.run",
        description="Build the Groww MF FAQ corpus stage by stage.",
    )
    parser.add_argument(
        "stage",
        nargs="+",
        choices=STAGES + ("all",),
        help="one or more stages to run, in the order given (or 'all')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore an existing data/.lock (only after a crashed run)",
    )
    parser.add_argument(
        "--config", default=None, help="path to config.yaml (default: config/config.yaml)"
    )
    args = parser.parse_args(argv)
    if "all" in args.stage and len(args.stage) > 1:
        print("'all' cannot be combined with other stages.", file=sys.stderr)
        return 2
    run_all = args.stage == ["all"]
    requested = list(STAGES) if run_all else list(args.stage)

    try:
        settings = load_settings(args.config, require_provider_key=False)
    except ConfigError as exc:
        print(f"CONFIG ERROR\n{exc}", file=sys.stderr)
        return 2

    lock_path = settings.paths.data_dir / ".lock"
    if lock_path.exists() and not args.force:
        print(
            f"Another ingest appears to be running ({lock_path}).\n"
            "If no run is active, delete the file and retry.",
            file=sys.stderr,
        )
        return 3
    if lock_path.exists() and args.force:
        lock_path.unlink()

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        f"pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )

    started = time.perf_counter()
    try:
        stages = requested
        for stage in stages:
            if stage not in IMPLEMENTED:
                print(
                    f"[{stage}] not implemented yet "
                    f"(implemented: {', '.join(IMPLEMENTED)}).",
                    file=sys.stderr,
                )
                if run_all:
                    print("Stopped after the implemented stages.")
                return 4 if not run_all else 0
            code = _run_stage(stage, settings)
            if code != 0:
                return code
    finally:
        lock_path.unlink(missing_ok=True)
        elapsed = time.perf_counter() - started
        print(f"\ndone in {elapsed:.1f}s")

    return 0


def _run_stage(stage: str, settings: Settings) -> int:
    if stage == "fetch":
        return _fetch(settings)
    if stage == "parse":
        return _parse(settings)
    if stage == "load":
        return _load(settings)
    if stage == "chunk":
        return _chunk(settings)
    if stage == "embed":
        return _embed(settings)
    if stage == "index":
        return _index(settings)
    print(f"[{stage}] unknown stage", file=sys.stderr)
    return 2


def _fetch(settings: Settings) -> int:
    from ingest.fetch import fetch_all

    print(f"[fetch] {settings.paths.sources_csv}")
    print(f"[fetch] raw -> {settings.paths.data_dir / '01_raw'}")
    entries = fetch_all(settings)

    print()
    print(f"{'id':<4} {'status':<7} {'bytes':>9} {'ok':<4} slug")
    print("-" * 74)
    for entry in entries:
        status = "-" if entry.status is None else str(entry.status)
        print(
            f"{entry.id:<4} {status:<7} {entry.bytes:>9} "
            f"{'yes' if entry.ok else 'no':<4} {entry.slug}"
            + (f"  ({entry.error})" if entry.error else "")
        )

    failed = [entry for entry in entries if not entry.ok]
    print()
    print(f"fetched {len(entries) - len(failed)}/{len(entries)} pages")
    if failed:
        print(
            "WARN: failed pages are recorded in manifest.json; "
            "the parse stage will skip them.",
            file=sys.stderr,
        )
    return 0 if len(failed) < len(entries) else 1


def _parse(settings: Settings) -> int:
    from ingest.parse import parse_all

    print(f"[parse] raw <- {settings.paths.data_dir / '01_raw'}")
    report = parse_all(settings)

    print()
    print(f"{'id':<4} {'type':<7} {'chars':>7} {'sects':>5} {'ok':<4} page")
    print("-" * 78)
    for page in report["pages"]:
        print(
            f"{page['id']:<4} {page['page_type']:<7} {page['char_len']:>7} "
            f"{page['section_count']:>5} {'yes' if page['ok'] else 'no':<4} "
            f"{page['slug']}"
        )
        if not page["ok"]:
            print(f"     ! {page['reason']}")

    summary = report["summary"]
    print()
    print(
        f"parsed {summary['ok']}/{summary['pages']} pages, "
        f"{summary['total_chars']} chars, {summary['gaps']} fact gap(s)"
    )
    print(f"report -> {settings.paths.data_dir / '02_clean' / 'parse_report.json'}")

    if report["quarantined"]:
        print(
            f"WARN: {len(report['quarantined'])} page(s) quarantined; "
            "they must be fixed or documented before chunking.",
            file=sys.stderr,
        )
        return 1
    return 0


def _load(settings: Settings) -> int:
    from ingest.load import load_documents

    clean_dir = settings.paths.data_dir / "02_clean"
    print(f"[load] clean <- {clean_dir}")
    result = load_documents(settings)

    print()
    print(f"{'id':<4} {'type':<7} {'sects':>5} {'chars':>7} doc")
    print("-" * 78)
    for doc in result.documents:
        print(
            f"{doc.source_id:<4} {doc.page_type:<7} {doc.section_count:>5} "
            f"{doc.char_len:>7} {doc.doc_id}"
        )
    for item in result.rejected:
        print(f"     ! {item.source_id} {item.slug}: {item.reason}")
    for item in result.duplicates:
        print(f"     = {item.source_id} {item.slug}: {item.reason}")

    summary = result.payload["summary"]
    print()
    print(
        f"loaded {summary['loaded']} document(s), {summary['sections']} sections, "
        f"{summary['total_chars']} chars, {summary['rejected']} rejected, "
        f"{summary['duplicates']} duplicate(s)"
    )
    print(f"docs   -> {clean_dir / 'documents.jsonl'}")
    print(f"report -> {clean_dir / 'load_report.json'}")

    if not result.documents:
        print("ERROR: no documents passed the load gate.", file=sys.stderr)
        return 1
    if result.rejected:
        print(
            f"WARN: {len(result.rejected)} page(s) rejected; "
            "chunking will only see loaded documents.",
            file=sys.stderr,
        )
        return 1
    return 0


def _chunk(settings: Settings) -> int:
    from ingest.chunk import chunk_all

    chunk_dir = settings.paths.data_dir / "03_chunks"
    documents = settings.paths.data_dir / "02_clean" / "documents.jsonl"
    print(f"[chunk] documents <- {documents}")
    result = chunk_all(settings)
    stats = result.stats
    core = ("expense_ratio", "exit_load", "min_sip", "riskometer", "benchmark")

    print()
    print(
        f"{'scheme':<48} {'exp':>4} {'exit':>4} {'sip':>4} "
        f"{'risk':>4} {'bmk':>4} {'lock':>4} {'win':>4}"
    )
    print("-" * 86)
    for slug, coverage in stats["coverage"].items():
        if coverage["page_type"] != "scheme":
            continue
        counts = [coverage["fields"].get(field, 0) for field in core]
        lock = coverage["fields"].get("lock_in", 0)
        cells = " ".join(f"{count:>4}" for count in counts)
        print(
            f"{slug:<48} {cells} {lock:>4} {coverage['windows']:>4}"
        )

    amc = [
        (slug, coverage)
        for slug, coverage in stats["coverage"].items()
        if coverage["page_type"] == "amc"
    ]
    for slug, coverage in amc:
        fields = ", ".join(
            f"{name}={count}" for name, count in coverage["fields"].items()
        )
        print(f"{slug:<48} amc  {fields}  windows={coverage['windows']}")

    counts = stats["counts"]
    print()
    print(
        f"chunked {stats['input']['documents']} document(s) into "
        f"{counts['total']} chunk(s): {counts['fact_atom']} fact_atom, "
        f"{counts['section_window']} section_window, "
        f"{stats['duplicates']} duplicate(s)"
    )
    skipped = len(stats["skipped_sections"])
    if skipped:
        print(f"skipped {skipped} short/noise section(s)")
    print(f"chunks -> {chunk_dir / 'chunks.jsonl'}")
    print(f"stats  -> {chunk_dir / 'chunk_stats.json'}")

    missing = stats["missing_required"]
    if missing:
        print(
            f"ERROR: {len(missing)} required fact field(s) have no atom:",
            file=sys.stderr,
        )
        for item in missing:
            print(
                f"  ! {item['scheme_slug']}: {item['field']}",
                file=sys.stderr,
            )
        return 1
    if not result.chunks:
        print("ERROR: no chunks produced.", file=sys.stderr)
        return 1
    return 0


def _embed(settings: Settings) -> int:
    from core.embedding import EmbeddingError
    from ingest.embed import embed_all

    cfg = settings.embedding
    chunks_path = settings.paths.data_dir / "03_chunks" / "chunks.jsonl"
    out_dir = settings.paths.data_dir / "04_embeddings"
    print(f"[embed] chunks <- {chunks_path}")
    print(
        f"[embed] model  <- {cfg.model} "
        f"(revision={cfg.revision or 'latest'}, device={cfg.device})"
    )

    try:
        result = embed_all(settings)
    except ConfigError as exc:
        print(f"CONFIG ERROR\n{exc}", file=sys.stderr)
        return 2
    except EmbeddingError as exc:
        print(f"EMBEDDING ERROR\n{exc}", file=sys.stderr)
        print(
            "Nothing was written. Check the model name, network access, and "
            "'pip install -r requirements.txt'.",
            file=sys.stderr,
        )
        return 5

    manifest = result.manifest
    counts = manifest["counts"]["by_chunk_type"]
    print()
    print(
        f"embedded {manifest['count']} chunk(s) into "
        f"{manifest['dim']}-d float32 vectors "
        f"({counts.get('fact_atom', 0)} fact_atom, "
        f"{counts.get('section_window', 0)} section_window)"
    )
    print(f"shape  -> {manifest['shape'][0]} x {manifest['shape'][1]}")
    print(f"chunks -> {chunks_path}")
    print(f"vectors -> {out_dir / 'embeddings.npy'}")
    print(f"manifest -> {out_dir / 'embedding_manifest.json'}")
    if cfg.normalize:
        norms = manifest["vector_norms"]
        print(
            f"L2 norm -> min={norms['min']} max={norms['max']} "
            f"mean={norms['mean']}"
        )
    print(f"chunks sha256  -> {manifest['chunks_sha256']}")
    print(f"vectors sha256 -> {manifest['vectors_sha256']}")
    return 0


def _index(settings: Settings) -> int:
    from core.store import StoreError
    from ingest.index import index_all

    cfg = settings.vector_store
    print(f"[index] embeddings <- {settings.paths.data_dir / '04_embeddings'}")
    print(
        f"[index] collection <- {cfg.collection!r} "
        f"({cfg.distance}, reset_on_rebuild={cfg.reset_on_rebuild})"
    )
    print(f"[index] chroma     -> {cfg.persist_dir}")

    try:
        result = index_all(settings)
    except ConfigError as exc:
        print(f"CONFIG ERROR\n{exc}", file=sys.stderr)
        return 2
    except StoreError as exc:
        print(f"INDEX ERROR\n{exc}", file=sys.stderr)
        print(
            "The app will refuse to answer until the index is rebuilt.",
            file=sys.stderr,
        )
        return 6

    meta = result.meta
    print()
    print(
        f"indexed {meta['chunk_count']} chunk(s) into "
        f"collection {meta['collection']!r}"
    )
    print(
        f"model  -> {meta['embedding_model']} "
        f"(revision={meta['embedding_revision'] or 'latest'}, "
        f"dim={meta['embedding_dim']})"
    )
    print(
        f"fetched -> {meta['oldest_fetched_at']} .. {meta['newest_fetched_at']}"
    )
    print(f"collection count -> {result.collection_count}")
    print(f"index_meta -> {result.meta_path}")
    print(f"corpus_hash            -> {meta['corpus_hash']}")
    print(f"embedding_manifest_hash -> {meta['embedding_manifest_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
