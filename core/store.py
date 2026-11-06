"""Chroma adapter and index-drift guard.

This is the only module that talks to Chroma. Ingestion writes a collection
through :class:`ChromaStore`; the query app opens it through :func:`open_index`,
which refuses to serve a collection whose model, dimension, distance,
collection name, or index fingerprint no longer matches the live config.

Vectors are always supplied explicitly. Chroma's own embedding function stays
``None`` so it can never re-embed chunks with a different model than
``data/04_embeddings``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import Settings, VectorStoreCfg
from core.types import Chunk, RetrievedChunk

INDEX_SCHEMA_VERSION = "index_v1"
INDEX_META_NAME = "index_meta.json"


class StoreError(RuntimeError):
    """Chroma is unavailable or returned an unusable result."""


class IndexStale(StoreError):
    """The persisted index is missing or does not match the live config."""


@dataclass(slots=True)
class IndexMeta:
    schema_version: str
    collection: str
    embedding_model: str
    embedding_revision: str | None
    embedding_dim: int
    distance: str
    chunk_count: int
    corpus_hash: str
    config_hash: str
    index_fingerprint: str
    embedding_manifest_hash: str
    chunks_sha256: str
    vectors_sha256: str
    created_at: str
    oldest_fetched_at: str
    newest_fetched_at: str
    embedder_version: str = ""
    text_template_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IndexMeta:
        required = (
            "schema_version",
            "collection",
            "embedding_model",
            "embedding_dim",
            "distance",
            "chunk_count",
            "corpus_hash",
            "config_hash",
            "index_fingerprint",
            "embedding_manifest_hash",
            "chunks_sha256",
            "vectors_sha256",
            "created_at",
            "oldest_fetched_at",
            "newest_fetched_at",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise IndexStale(
                f"{INDEX_META_NAME} is missing keys: {', '.join(missing)}; "
                "run 'python -m ingest.run index'"
            )
        return cls(
            schema_version=str(payload["schema_version"]),
            collection=str(payload["collection"]),
            embedding_model=str(payload["embedding_model"]),
            embedding_revision=payload.get("embedding_revision"),
            embedding_dim=int(payload["embedding_dim"]),
            distance=str(payload["distance"]),
            chunk_count=int(payload["chunk_count"]),
            corpus_hash=str(payload["corpus_hash"]),
            config_hash=str(payload["config_hash"]),
            index_fingerprint=str(payload["index_fingerprint"]),
            embedding_manifest_hash=str(payload["embedding_manifest_hash"]),
            chunks_sha256=str(payload["chunks_sha256"]),
            vectors_sha256=str(payload["vectors_sha256"]),
            created_at=str(payload["created_at"]),
            oldest_fetched_at=str(payload["oldest_fetched_at"]),
            newest_fetched_at=str(payload["newest_fetched_at"]),
            embedder_version=str(payload.get("embedder_version") or ""),
            text_template_version=str(payload.get("text_template_version") or ""),
        )


# ---------------------------------------------------------------------------
# metadata + discovery
# ---------------------------------------------------------------------------


def flat_metadata(chunk: Chunk) -> dict[str, str | int]:
    """Chroma metadata must be flat scalars; keep the fields the query uses."""
    return {
        "chunk_type": chunk.chunk_type,
        "scheme": chunk.scheme,
        "scheme_slug": chunk.scheme_slug,
        "field": chunk.field,
        "section_title": chunk.section_title,
        "page_type": chunk.page_type,
        "source_url": chunk.source_url,
        "source_title": chunk.source_title,
        "fetched_at": chunk.fetched_at,
        "char_len": chunk.char_len,
    }


def _import_chromadb() -> Any:
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise StoreError(
            "chromadb is not installed; run 'pip install -r requirements.txt'"
        ) from exc
    return chromadb


def open_client(cfg: VectorStoreCfg) -> Any:
    """Open a persistent Chroma client, creating the directory if needed."""
    chromadb = _import_chromadb()
    cfg.persist_dir.mkdir(parents=True, exist_ok=True)
    try:
        return chromadb.PersistentClient(path=str(cfg.persist_dir))
    except Exception as exc:  # pragma: no cover - disk/permission failure
        raise StoreError(
            f"could not open Chroma at {cfg.persist_dir}: {exc}"
        ) from exc


def _collection_metadata(cfg: VectorStoreCfg) -> dict[str, str]:
    return {"hnsw:space": cfg.distance}


def _chroma_where(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """Chroma wants one operator per ``where``; ``$and`` combines keys."""
    if not where:
        return None
    if len(where) == 1:
        return where
    return {"$and": [{key: value} for key, value in where.items()]}


class ChromaStore:
    """Thin wrapper around one Chroma collection."""

    def __init__(self, client: Any, collection: Any, cfg: VectorStoreCfg) -> None:
        self.client = client
        self.collection = collection
        self.cfg = cfg

    @classmethod
    def open(cls, settings: Settings, *, create: bool, client: Any = None) -> ChromaStore:
        cfg = settings.vector_store
        client = client if client is not None else open_client(cfg)
        try:
            if create:
                collection = client.get_or_create_collection(
                    name=cfg.collection,
                    embedding_function=None,
                    metadata=_collection_metadata(cfg),
                )
            else:
                collection = client.get_collection(
                    name=cfg.collection,
                    embedding_function=None,
                )
        except Exception as exc:
            if create:
                raise StoreError(
                    f"could not create collection {cfg.collection!r}: {exc}"
                ) from exc
            raise IndexStale(
                f"collection {cfg.collection!r} is missing or unreadable; "
                "run 'python -m ingest.run index'"
            ) from exc
        return cls(client, collection, cfg)

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception as exc:  # pragma: no cover - Chroma failure
            raise StoreError(f"could not count collection: {exc}") from exc

    def reset(self) -> None:
        """Drop and recreate the collection so a rebuild cannot leave old rows."""
        names: set[str] = set()
        try:
            for item in self.client.list_collections():
                names.add(item if isinstance(item, str) else item.name)
        except Exception as exc:  # pragma: no cover - Chroma failure
            raise StoreError(f"could not list Chroma collections: {exc}") from exc

        if self.cfg.collection in names:
            try:
                self.client.delete_collection(name=self.cfg.collection)
            except Exception as exc:  # pragma: no cover - Chroma failure
                raise StoreError(
                    f"could not reset collection {self.cfg.collection!r}: {exc}"
                ) from exc

        try:
            self.collection = self.client.get_or_create_collection(
                name=self.cfg.collection,
                embedding_function=None,
                metadata=_collection_metadata(self.cfg),
            )
        except Exception as exc:  # pragma: no cover - Chroma failure
            raise StoreError(
                f"could not recreate collection {self.cfg.collection!r}: {exc}"
            ) from exc

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        *,
        batch_size: int = 256,
    ) -> int:
        """Write rows in batches; vectors must already match chunks row-for-row."""
        if len(chunks) != int(vectors.shape[0]):
            raise StoreError(
                f"chunk count {len(chunks)} != vector rows {vectors.shape[0]}"
            )
        if batch_size < 1:
            raise StoreError("vector_store.batch_size must be >= 1")

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            matrix = vectors[start : start + batch_size]
            try:
                self.collection.upsert(
                    ids=[chunk.chunk_id for chunk in batch],
                    documents=[chunk.text for chunk in batch],
                    metadatas=[flat_metadata(chunk) for chunk in batch],
                    embeddings=matrix.tolist(),
                )
            except Exception as exc:  # pragma: no cover - Chroma failure
                raise StoreError(
                    f"could not upsert rows {start}-{start + len(batch)}: {exc}"
                ) from exc
        return self.count()

    def rebuild(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        *,
        batch_size: int = 256,
    ) -> int:
        self.reset()
        return self.upsert(chunks, vectors, batch_size=batch_size)

    def query(
        self,
        embedding: np.ndarray,
        *,
        where: dict[str, Any] | None = None,
        n_results: int = 10,
        include_embeddings: bool = False,
    ) -> list[RetrievedChunk]:
        """Nearest neighbours, with cosine similarity = 1 - Chroma distance."""
        if n_results < 1:
            raise StoreError("n_results must be >= 1")
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")
        try:
            payload = self.collection.query(
                query_embeddings=[embedding.astype(float).tolist()],
                n_results=n_results,
                where=_chroma_where(where),
                include=include,
            )
        except Exception as exc:  # pragma: no cover - Chroma failure
            raise StoreError(f"Chroma query failed: {exc}") from exc

        ids = (payload.get("ids") or [[]])[0]
        documents = (payload.get("documents") or [[]])[0]
        metadatas = (payload.get("metadatas") or [[]])[0]
        distances = (payload.get("distances") or [[]])[0]
        embeddings = (payload.get("embeddings") or []) if include_embeddings else []
        vectors = embeddings[0] if embeddings else []

        results: list[RetrievedChunk] = []
        for index, (chunk_id, text, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances, strict=False)
        ):
            score = 1.0 - float(distance) if distance is not None else 0.0
            vector = None
            if index < len(vectors) and vectors[index] is not None:
                vector = np.asarray(vectors[index], dtype=np.float32).tolist()
            results.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    text=str(text or ""),
                    score=score,
                    metadata=dict(metadata or {}),
                    source="dense",
                    embedding=vector,
                )
            )
        return results


def load_index_meta(path: Path) -> IndexMeta:
    if not path.exists():
        raise IndexStale(
            f"{path} not found; run 'python -m ingest.run index'"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexStale(
            f"{path} is unreadable; run 'python -m ingest.run index'"
        ) from exc
    if not isinstance(payload, dict):
        raise IndexStale(
            f"{path} is not a JSON object; run 'python -m ingest.run index'"
        )
    return IndexMeta.from_dict(payload)


def validate_index_meta(
    settings: Settings,
    meta: IndexMeta,
    *,
    store: ChromaStore | None = None,
    expected_dim: int | None = None,
) -> None:
    """Raise :class:`IndexStale` with every mismatch, not just the first."""
    cfg = settings.vector_store
    problems: list[str] = []

    if meta.schema_version != INDEX_SCHEMA_VERSION:
        problems.append(
            f"schema {meta.schema_version!r} != {INDEX_SCHEMA_VERSION!r}"
        )
    if meta.collection != cfg.collection:
        problems.append(f"collection {meta.collection!r} != {cfg.collection!r}")
    if meta.embedding_model != settings.embedding.model:
        problems.append(
            f"model {meta.embedding_model!r} != {settings.embedding.model!r}"
        )
    if meta.embedding_revision != settings.embedding.revision:
        problems.append(
            f"revision {meta.embedding_revision!r} != "
            f"{settings.embedding.revision!r}"
        )
    if meta.distance != cfg.distance:
        problems.append(f"distance {meta.distance!r} != {cfg.distance!r}")
    if meta.index_fingerprint != settings.index_fingerprint():
        problems.append("chunking/embedding/vector-store config changed")
    if expected_dim is not None and meta.embedding_dim != expected_dim:
        problems.append(
            f"dimension {meta.embedding_dim} != expected {expected_dim}"
        )
    if store is not None:
        live_count = store.count()
        if live_count != meta.chunk_count:
            problems.append(
                f"collection count {live_count} != metadata {meta.chunk_count}"
            )

    if problems:
        raise IndexStale(
            "index is stale: " + "; ".join(problems)
            + "; run 'python -m ingest.run index'"
        )


def open_index(
    settings: Settings,
    *,
    client: Any = None,
    expected_dim: int | None = None,
) -> ChromaStore:
    """Return a verified store, or raise :class:`IndexStale`."""
    meta_path = settings.vector_store.persist_dir / INDEX_META_NAME
    meta = load_index_meta(meta_path)
    store = ChromaStore.open(settings, create=False, client=client)
    validate_index_meta(settings, meta, store=store, expected_dim=expected_dim)
    return store


__all__ = [
    "INDEX_META_NAME",
    "INDEX_SCHEMA_VERSION",
    "ChromaStore",
    "IndexMeta",
    "IndexStale",
    "StoreError",
    "flat_metadata",
    "load_index_meta",
    "open_client",
    "open_index",
    "validate_index_meta",
]
