"""The one embedder used by both ingestion and query.

Ingest and query must never call different implementations, or the index and
the query vector can drift. This module wraps sentence-transformers behind a
small interface, normalises vectors, and validates them before they are saved.

The encoder is injectable so tests can run offline and fast; production code
loads ``sentence-transformers`` lazily on first use.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from config.settings import EmbeddingCfg
from core.fields import FIELD_SPECS
from core.types import Chunk

TEXT_TEMPLATE_VERSION = "embed_text_v1"
TEXT_TEMPLATE = "{scheme} | {field_label_or_section_title} | {text}"


class EmbeddingError(RuntimeError):
    """The embedder could not be loaded or produced unusable vectors."""


class Encoder(Protocol):
    """The part of ``SentenceTransformer`` this project relies on."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool = False,
    ) -> Any: ...


def embedding_text(chunk: Chunk) -> str:
    """Text sent to the model; the stored ``chunk.text`` is never rewritten."""
    if chunk.field and chunk.field in FIELD_SPECS:
        label = FIELD_SPECS[chunk.field].label
    else:
        label = chunk.section_title or "context"
    return f"{chunk.scheme} | {label} | {chunk.text}"


def load_encoder(cfg: EmbeddingCfg) -> Encoder:
    """Load the configured sentence-transformers model.

    Kept out of module import time so tests and the rest of ingestion do not
    need torch unless embedding actually runs.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise EmbeddingError(
            "sentence-transformers is not installed; run "
            "'pip install -r requirements.txt'"
        ) from exc

    try:
        return SentenceTransformer(
            cfg.model,
            revision=cfg.revision,
            device=cfg.device,
        )
    except Exception as exc:  # pragma: no cover - network/cache failure
        raise EmbeddingError(
            f"could not load embedding model {cfg.model!r}: {exc}"
        ) from exc


class Embedder:
    """Batch encoder with a fixed dimension for a run."""

    def __init__(self, cfg: EmbeddingCfg, *, encoder: Encoder | None = None) -> None:
        self.cfg = cfg
        self._encoder = encoder if encoder is not None else load_encoder(cfg)
        self._dim: int | None = None

    @property
    def dim(self) -> int | None:
        return self._dim

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into a ``(len(texts), dim)`` float32 matrix."""
        items = [str(text) for text in texts]
        if not items:
            return np.zeros((0, self._dim or 0), dtype=np.float32)

        raw = self._encoder.encode(
            items,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(raw, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2:
            raise EmbeddingError(f"expected a 2-D matrix, got shape {vectors.shape}")
        if vectors.shape[0] != len(items):
            raise EmbeddingError(
                f"encoder returned {vectors.shape[0]} rows for {len(items)} texts"
            )
        if vectors.shape[1] == 0:
            raise EmbeddingError("encoder returned zero-width vectors")
        if self._dim is not None and vectors.shape[1] != self._dim:
            raise EmbeddingError(
                f"embedding dimension changed from {self._dim} to {vectors.shape[1]}"
            )
        if not np.isfinite(vectors).all():
            raise EmbeddingError("encoder returned NaN or infinite values")

        if self.cfg.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if np.any(norms <= 0):
                raise EmbeddingError("encoder returned a zero vector")
            vectors = vectors / norms

        self._dim = int(vectors.shape[1])
        return vectors.astype(np.float32, copy=False)

    def docs(self, chunks: Sequence[Chunk]) -> np.ndarray:
        return self.encode_texts([embedding_text(chunk) for chunk in chunks])

    def query(self, question: str) -> np.ndarray:
        return self.encode_texts([question])[0]


__all__ = [
    "TEXT_TEMPLATE",
    "TEXT_TEMPLATE_VERSION",
    "Embedder",
    "EmbeddingError",
    "Encoder",
    "embedding_text",
    "load_encoder",
]
