"""Stage 4: embed every chunk and persist vectors for the index stage.

Input is ``data/03_chunks/chunks.jsonl`` only. Row *i* of the saved matrix always
corresponds to line *i* of that file, so the index can join embeddings to chunk
metadata without a second source of truth.

The text sent to the model is built by :func:`core.embedding.embedding_text`
(``scheme | field label or section title | text``); the chunk text stored on
disk is never rewritten. Vectors are L2-normalised when
``embedding.normalize`` is true, validated for shape/NaN/zero rows, and hashed
so a stale index can be detected before it answers a question.

Artifacts
---------
``data/04_embeddings/embeddings.npy``         float32 ``(count, dim)`` matrix
``data/04_embeddings/embedding_manifest.json`` model, template, shape, hashes,
                                               config hash, counts, created_at
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import ConfigError, Settings
from core.chunks import read_chunks
from core.embedding import (
    TEXT_TEMPLATE,
    TEXT_TEMPLATE_VERSION,
    Embedder,
    EmbeddingError,
    embedding_text,
)
from core.types import Chunk

EMBEDDER_VERSION = "embed_v1"


@dataclass(slots=True)
class EmbeddingResult:
    vectors: np.ndarray
    chunks: list[Chunk] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    vectors_path: Path | None = None
    manifest_path: Path | None = None


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def embed_all(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    write: bool = True,
) -> EmbeddingResult:
    """Read chunks, embed them, validate, and persist vectors + manifest."""
    chunks_path = settings.paths.data_dir / "03_chunks" / "chunks.jsonl"
    if not chunks_path.exists():
        raise ConfigError(
            f"{chunks_path} not found; run 'python -m ingest.run chunk' first"
        )

    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ConfigError(
            f"{chunks_path} has no chunks; run 'python -m ingest.run chunk'"
        )

    active = embedder if embedder is not None else Embedder(settings.embedding)
    vectors = active.docs(chunks)
    vectors = _validate_vectors(vectors, expected_rows=len(chunks))

    manifest = _build_manifest(settings, chunks, vectors, chunks_path, active)
    result = EmbeddingResult(vectors=vectors, chunks=chunks, manifest=manifest)

    if write:
        out_dir = settings.paths.data_dir / "04_embeddings"
        vectors_path = out_dir / "embeddings.npy"
        manifest_path = out_dir / "embedding_manifest.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_npy(vectors_path, vectors)
        _atomic_write_json(manifest_path, manifest)
        result.vectors_path = vectors_path
        result.manifest_path = manifest_path

    return result


def load_chunks(path: Path) -> list[Chunk]:
    """Backward-compatible name for :func:`core.chunks.read_chunks`."""
    return read_chunks(path)


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def _validate_vectors(vectors: np.ndarray, *, expected_rows: int) -> np.ndarray:
    if not isinstance(vectors, np.ndarray):
        raise EmbeddingError(f"expected a numpy array, got {type(vectors).__name__}")
    if vectors.ndim != 2:
        raise EmbeddingError(f"expected a 2-D matrix, got shape {vectors.shape}")
    if vectors.shape[0] != expected_rows:
        raise EmbeddingError(
            f"embedding row count {vectors.shape[0]} != chunk count {expected_rows}"
        )
    if vectors.shape[1] == 0:
        raise EmbeddingError("embedding matrix has zero columns")
    if not np.isfinite(vectors).all():
        raise EmbeddingError("embedding matrix contains NaN or infinite values")
    return vectors.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# manifest + artifacts
# ---------------------------------------------------------------------------


def _build_manifest(
    settings: Settings,
    chunks: list[Chunk],
    vectors: np.ndarray,
    chunks_path: Path,
    embedder: Embedder,
) -> dict[str, Any]:
    cfg = settings.embedding
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    norms = np.linalg.norm(vectors, axis=1)

    return {
        "stage": "embed",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedder_version": EMBEDDER_VERSION,
        "text_template_version": TEXT_TEMPLATE_VERSION,
        "text_template": TEXT_TEMPLATE,
        "model": cfg.model,
        "revision": cfg.revision,
        "device": cfg.device,
        "batch_size": cfg.batch_size,
        "normalized": cfg.normalize,
        "dtype": "float32",
        "dim": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
        "shape": [int(vectors.shape[0]), int(vectors.shape[1])],
        "chunk_order": (
            "row i == line i+1 of the input chunks.jsonl (file order preserved)"
        ),
        "chunk_ids": chunk_ids,
        "chunk_ids_sha256": _sha256_text("\n".join(chunk_ids)),
        "chunks_sha256": _sha256_bytes(chunks_path.read_bytes()),
        "vectors_sha256": _sha256_bytes(vectors.tobytes(order="C")),
        "config_hash": settings.config_hash,
        "index_fingerprint": settings.index_fingerprint(),
        "input": {
            "path": str(chunks_path),
            "bytes": chunks_path.stat().st_size,
            "chunks": len(chunks),
        },
        "counts": {
            "by_chunk_type": dict(
                sorted(Counter(chunk.chunk_type for chunk in chunks).items())
            ),
            "by_page_type": dict(
                sorted(Counter(chunk.page_type for chunk in chunks).items())
            ),
            "schemes": sorted({chunk.scheme_slug for chunk in chunks}),
        },
        "vector_norms": {
            "min": round(float(norms.min()), 6) if norms.size else None,
            "max": round(float(norms.max()), 6) if norms.size else None,
            "mean": round(float(norms.mean()), 6) if norms.size else None,
        },
        "artifacts": {
            "vectors": "data/04_embeddings/embeddings.npy",
            "manifest": "data/04_embeddings/embedding_manifest.json",
        },
        "embedder_dim": embedder.dim,
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write_npy(path: Path, vectors: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, vectors, allow_pickle=False)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


__all__ = [
    "EMBEDDER_VERSION",
    "EmbeddingResult",
    "embed_all",
    "embedding_text",
    "load_chunks",
]
