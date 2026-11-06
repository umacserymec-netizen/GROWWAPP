"""Stage 5: load the embedding artifacts into a persistent Chroma collection.

Input is the pair written by ``ingest/embed.py`` plus the chunk file it came
from. Every hash is checked before Chroma is touched, so a stale or mismatched
vector matrix can never be indexed silently.

Artifacts
---------
``data/05_chroma/``                    Chroma persistence (sqlite + segments)
``data/05_chroma/index_meta.json``     model/dim/distance, counts, corpus +
                                       config + manifest hashes, fetched dates

The rebuild is destructive by design (``vector_store.reset_on_rebuild``): the
old ``index_meta.json`` is removed first, the collection is dropped and
recreated, rows are upserted in batches, and the new metadata is written last.
A crash therefore leaves the app in diagnostic mode instead of serving a
half-built index.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import ConfigError, Settings
from core.store import (
    INDEX_META_NAME,
    INDEX_SCHEMA_VERSION,
    ChromaStore,
    StoreError,
    load_index_meta,
    validate_index_meta,
)
from core.types import Chunk
from ingest.embed import load_chunks

INDEXER_VERSION = "index_v1"

_REQUIRED_MANIFEST_KEYS = (
    "model",
    "revision",
    "dim",
    "count",
    "shape",
    "chunk_ids",
    "chunk_ids_sha256",
    "chunks_sha256",
    "vectors_sha256",
    "config_hash",
    "index_fingerprint",
)


@dataclass(slots=True)
class IndexResult:
    meta: dict[str, Any] = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)
    vectors: np.ndarray | None = None
    meta_path: Path | None = None
    collection_count: int = 0


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def index_all(
    settings: Settings,
    *,
    store: ChromaStore | None = None,
    write: bool = True,
) -> IndexResult:
    """Validate embedding artifacts, rebuild Chroma, and write index_meta."""
    chunks_path = settings.paths.data_dir / "03_chunks" / "chunks.jsonl"
    vectors_path = settings.paths.data_dir / "04_embeddings" / "embeddings.npy"
    manifest_path = (
        settings.paths.data_dir / "04_embeddings" / "embedding_manifest.json"
    )
    meta_path = settings.vector_store.persist_dir / INDEX_META_NAME

    manifest, manifest_bytes = _read_manifest(manifest_path)
    if not chunks_path.exists():
        raise ConfigError(
            f"{chunks_path} not found; run 'python -m ingest.run chunk' first"
        )
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ConfigError(
            f"{chunks_path} has no chunks; run 'python -m ingest.run chunk'"
        )
    vectors = _read_vectors(vectors_path)
    _validate_inputs(
        settings,
        chunks,
        vectors,
        chunks_path,
        manifest,
        manifest_path,
    )

    active = store or ChromaStore.open(settings, create=True)
    if write:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.unlink(missing_ok=True)  # never leave the old meta on a failed rebuild

    try:
        if settings.vector_store.reset_on_rebuild:
            count = active.rebuild(
                chunks,
                vectors,
                batch_size=settings.vector_store.batch_size,
            )
        else:
            count = active.upsert(
                chunks,
                vectors,
                batch_size=settings.vector_store.batch_size,
            )
    except StoreError:
        raise
    except Exception as exc:  # pragma: no cover - defensive Chroma failure
        raise StoreError(f"could not rebuild Chroma collection: {exc}") from exc

    if count != len(chunks):
        raise StoreError(
            f"collection has {count} rows after indexing {len(chunks)} chunks"
        )

    meta = _build_index_meta(settings, chunks, vectors, manifest, manifest_bytes)
    result = IndexResult(
        meta=meta,
        chunks=chunks,
        vectors=vectors,
        meta_path=meta_path if write else None,
        collection_count=count,
    )

    if write:
        _atomic_write_json(meta_path, meta)
        persisted = load_index_meta(meta_path)
        validate_index_meta(settings, persisted, store=active)
        result.meta_path = meta_path

    return result


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def _read_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.exists():
        raise ConfigError(
            f"{path} not found; run 'python -m ingest.run embed' first"
        )
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"{path} is unreadable; run 'python -m ingest.run embed'"
        ) from exc
    if not isinstance(payload, dict):
        raise ConfigError(
            f"{path} is not a JSON object; run 'python -m ingest.run embed'"
        )
    missing = [key for key in _REQUIRED_MANIFEST_KEYS if key not in payload]
    if missing:
        raise ConfigError(
            f"{path} is missing keys: {', '.join(missing)}; "
            "run 'python -m ingest.run embed'"
        )
    return payload, raw


def _read_vectors(path: Path) -> np.ndarray:
    if not path.exists():
        raise ConfigError(
            f"{path} not found; run 'python -m ingest.run embed' first"
        )
    try:
        vectors = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ConfigError(
            f"{path} is not a readable .npy matrix; "
            "run 'python -m ingest.run embed'"
        ) from exc
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        raise ConfigError(
            f"{path} must be a 2-D matrix; run 'python -m ingest.run embed'"
        )
    if vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ConfigError(
            f"{path} is empty; run 'python -m ingest.run embed'"
        )
    if not np.isfinite(vectors).all():
        raise ConfigError(
            f"{path} contains NaN or infinite values; "
            "run 'python -m ingest.run embed'"
        )
    return vectors.astype(np.float32, copy=False)


def _validate_inputs(
    settings: Settings,
    chunks: list[Chunk],
    vectors: np.ndarray,
    chunks_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    expected_ids = manifest.get("chunk_ids")
    expected_shape = manifest.get("shape")

    problems: list[str] = []
    if manifest["model"] != settings.embedding.model:
        problems.append(
            f"manifest model {manifest['model']!r} != "
            f"config {settings.embedding.model!r}"
        )
    if manifest.get("revision") != settings.embedding.revision:
        problems.append(
            f"manifest revision {manifest.get('revision')!r} != "
            f"config {settings.embedding.revision!r}"
        )
    if manifest.get("index_fingerprint") != settings.index_fingerprint():
        problems.append("chunking/embedding/vector-store config changed")
    if manifest.get("count") != len(chunks):
        problems.append(
            f"manifest count {manifest.get('count')} != chunk count {len(chunks)}"
        )
    if expected_ids != chunk_ids:
        problems.append("manifest chunk_ids do not match chunks.jsonl")
    if manifest.get("chunk_ids_sha256") != _sha256_text("\n".join(chunk_ids)):
        problems.append("manifest chunk_ids_sha256 does not match chunks.jsonl")
    if manifest.get("chunks_sha256") != _sha256_bytes(chunks_path.read_bytes()):
        problems.append("chunks.jsonl changed after embedding")
    if manifest.get("vectors_sha256") != _sha256_bytes(
        vectors.tobytes(order="C")
    ):
        problems.append("embeddings.npy does not match its manifest hash")
    if manifest.get("dim") != int(vectors.shape[1]):
        problems.append(
            f"manifest dim {manifest.get('dim')} != vector width {vectors.shape[1]}"
        )
    if expected_shape != [int(vectors.shape[0]), int(vectors.shape[1])]:
        problems.append(
            f"manifest shape {expected_shape} != vector shape {list(vectors.shape)}"
        )

    if problems:
        detail = "; ".join(problems)
        raise ConfigError(
            f"{manifest_path} is out of date: {detail}; "
            "run 'python -m ingest.run embed' then retry"
        )


# ---------------------------------------------------------------------------
# metadata + artifacts
# ---------------------------------------------------------------------------


def _build_index_meta(
    settings: Settings,
    chunks: list[Chunk],
    vectors: np.ndarray,
    manifest: dict[str, Any],
    manifest_bytes: bytes,
) -> dict[str, Any]:
    cfg = settings.vector_store
    dates = sorted(chunk.fetched_at for chunk in chunks if chunk.fetched_at)
    corpus_hash = _sha256_text(
        "\n".join(sorted(chunk.chunk_id for chunk in chunks))
    )
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "indexer_version": INDEXER_VERSION,
        "collection": cfg.collection,
        "embedding_model": settings.embedding.model,
        "embedding_revision": settings.embedding.revision,
        "embedding_dim": int(vectors.shape[1]),
        "distance": cfg.distance,
        "chunk_count": len(chunks),
        "corpus_hash": corpus_hash,
        "config_hash": settings.config_hash,
        "index_fingerprint": settings.index_fingerprint(),
        "embedding_manifest_hash": _sha256_bytes(manifest_bytes),
        "chunks_sha256": manifest["chunks_sha256"],
        "vectors_sha256": manifest["vectors_sha256"],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "oldest_fetched_at": dates[0] if dates else "",
        "newest_fetched_at": dates[-1] if dates else "",
        "embedder_version": str(manifest.get("embedder_version") or ""),
        "text_template_version": str(manifest.get("text_template_version") or ""),
        "reset_on_rebuild": cfg.reset_on_rebuild,
        "batch_size": cfg.batch_size,
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


__all__ = ["INDEXER_VERSION", "IndexResult", "index_all"]
