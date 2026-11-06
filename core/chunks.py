"""Read and validate the chunk artifact shared by ingestion and retrieval.

``data/03_chunks/chunks.jsonl`` is the boundary between offline chunking and
online retrieval. Keeping the reader in ``core/`` means the retriever can load
the BM25 corpus without importing the ingestion package, while ingestion still
gets the same strict validation it had before.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.settings import ConfigError
from core.types import Chunk

CHUNK_FIELDS = (
    "chunk_id",
    "chunk_type",
    "scheme",
    "scheme_slug",
    "field",
    "section_title",
    "text",
    "source_url",
    "source_title",
    "page_type",
    "fetched_at",
    "char_len",
)
CHUNK_TYPES = {"fact_atom", "section_window"}
PAGE_TYPES = {"help", "scheme", "amc", "education", "process", "factsheet"}


def read_chunks(path: Path) -> list[Chunk]:
    """Parse and validate ``chunks.jsonl``; fail before using any row."""
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc
        chunk = _chunk_from_payload(payload, path, line_no)

        if chunk.chunk_id in seen:
            raise ConfigError(
                f"{path}:{line_no}: duplicate chunk_id {chunk.chunk_id!r} "
                f"(first seen on line {seen[chunk.chunk_id]}); "
                "re-run 'python -m ingest.run chunk'"
            )
        seen[chunk.chunk_id] = line_no
        chunks.append(chunk)

    return chunks


def _chunk_from_payload(payload: Any, path: Path, line_no: int) -> Chunk:
    if not isinstance(payload, dict):
        raise ConfigError(
            f"{path}:{line_no}: chunk row is not an object; "
            "re-run 'python -m ingest.run chunk'"
        )

    missing = [name for name in CHUNK_FIELDS if name not in payload]
    if missing:
        raise ConfigError(
            f"{path}:{line_no}: missing fields: {', '.join(missing)}; "
            "re-run 'python -m ingest.run chunk'"
        )

    chunk_type = str(payload["chunk_type"])
    if chunk_type not in CHUNK_TYPES:
        raise ConfigError(
            f"{path}:{line_no}: unknown chunk_type {chunk_type!r}; "
            "re-run 'python -m ingest.run chunk'"
        )
    page_type = str(payload["page_type"])
    if page_type not in PAGE_TYPES:
        raise ConfigError(
            f"{path}:{line_no}: unknown page_type {page_type!r}; "
            "re-run 'python -m ingest.run chunk'"
        )

    text = str(payload["text"])
    if not text.strip():
        raise ConfigError(
            f"{path}:{line_no}: empty chunk text; "
            "re-run 'python -m ingest.run chunk'"
        )

    chunk_id = str(payload["chunk_id"])
    if not chunk_id:
        raise ConfigError(
            f"{path}:{line_no}: empty chunk_id; "
            "re-run 'python -m ingest.run chunk'"
        )

    try:
        char_len = int(payload["char_len"])
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{path}:{line_no}: char_len is not an integer; "
            "re-run 'python -m ingest.run chunk'"
        ) from exc
    if char_len != len(text):
        raise ConfigError(
            f"{path}:{line_no}: char_len {char_len} != len(text) {len(text)}; "
            "re-run 'python -m ingest.run chunk'"
        )

    return Chunk(
        chunk_id=chunk_id,
        chunk_type=chunk_type,  # type: ignore[arg-type]
        scheme=str(payload["scheme"]),
        scheme_slug=str(payload["scheme_slug"]),
        field=str(payload["field"]),
        section_title=str(payload["section_title"]),
        text=text,
        source_url=str(payload["source_url"]),
        source_title=str(payload["source_title"]),
        page_type=page_type,  # type: ignore[arg-type]
        fetched_at=str(payload["fetched_at"]),
        char_len=char_len,
    )


__all__ = ["CHUNK_FIELDS", "CHUNK_TYPES", "PAGE_TYPES", "read_chunks"]
