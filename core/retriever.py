"""Online retrieval: resolve → filter → dense → MMR → lexical fallback → gate.

The retriever is deliberately boring and inspectable. Every decision is written
to ``data/06_retrieval/<timestamp>_<qid>.json``: intent signals, the resolved
plan, each filter level's top score, the dense candidates, the MMR picks, any
BM25 fallback, and the final gate. Raw embeddings are never written to a trace.

No LLM is called anywhere in this module. A failed gate is a complete result:
the answer layer turns it into the not-found template.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from config.settings import ConfigError, Settings
from core.chunks import read_chunks
from core.embedding import Embedder
from core.ids import new_qid
from core.lexical import LexicalIndex
from core.resolve import build_plan
from core.store import (
    INDEX_META_NAME,
    StoreError,
    load_index_meta,
    open_index,
)
from core.types import IntentResult, QueryPlan, RetrievalResult, RetrievedChunk


class QueryEmbedder(Protocol):
    def query(self, question: str) -> np.ndarray: ...


class CandidateStore(Protocol):
    def query(
        self,
        embedding: np.ndarray,
        *,
        where: dict[str, Any] | None = None,
        n_results: int = 10,
        include_embeddings: bool = False,
    ) -> list[RetrievedChunk]: ...


class Retriever:
    """Reusable retrieval session over one verified Chroma index."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: QueryEmbedder,
        store: CandidateStore,
        lexical: LexicalIndex | None = None,
        index_dim: int | None = None,
        write_trace: bool = True,
        trace_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.lexical = lexical
        self.index_dim = index_dim
        self.write_trace = write_trace
        self.trace_dir = trace_dir or (settings.paths.data_dir / "06_retrieval")

    @classmethod
    def open(
        cls,
        settings: Settings,
        *,
        embedder: QueryEmbedder | None = None,
        client: Any = None,
        write_trace: bool = True,
    ) -> Retriever:
        """Load the real embedder, verified index, and BM25 corpus once."""
        active_embedder = embedder if embedder is not None else Embedder(
            settings.embedding
        )
        meta = load_index_meta(settings.vector_store.persist_dir / INDEX_META_NAME)
        store = open_index(settings, client=client)

        chunks_path = settings.paths.data_dir / "03_chunks" / "chunks.jsonl"
        if not chunks_path.exists():
            raise ConfigError(
                f"{chunks_path} not found; run 'python -m ingest.run chunk' first"
            )
        lexical = LexicalIndex(read_chunks(chunks_path))
        return cls(
            settings,
            embedder=active_embedder,
            store=store,
            lexical=lexical,
            index_dim=meta.embedding_dim,
            write_trace=write_trace,
        )

    # -- public API ------------------------------------------------------
    def retrieve(self, question: str) -> RetrievalResult:
        question = " ".join((question or "").split())
        qid = new_qid()
        plan, intent = build_plan(question)
        result = RetrievalResult(plan=plan, qid=qid)
        notes = plan.notes

        levels: list[dict[str, Any]] = []
        dense: list[dict[str, Any]] = []
        lexical: list[dict[str, Any]] = []
        lexical_used = False
        best_score = 0.0
        candidates: list[RetrievedChunk] = []
        selected: list[RetrievedChunk] = []

        if intent.label != "fact":
            result.filter_level = f"intent:{intent.label}"
            result.trace_path = self._write_trace(
                qid=qid,
                question=question,
                plan=plan,
                intent=intent,
                filter_level=result.filter_level,
                levels=levels,
                dense=dense,
                lexical=lexical,
                lexical_used=lexical_used,
                selected=selected,
                best_score=best_score,
                gate_passed=False,
                notes=notes,
            )
            return result

        if len(plan.scheme_candidates) > 1:
            result.filter_level = "ambiguous_scheme"
            notes.append("ask_user_which_scheme")
            result.trace_path = self._write_trace(
                qid=qid,
                question=question,
                plan=plan,
                intent=intent,
                filter_level=result.filter_level,
                levels=levels,
                dense=dense,
                lexical=lexical,
                lexical_used=lexical_used,
                selected=selected,
                best_score=best_score,
                gate_passed=False,
                notes=notes,
            )
            return result

        # A field such as ``fund_manager`` exists in the taxonomy but this
        # five-page corpus has no atom for it. Falling back to ``{scheme}`` or
        # a global pass would answer a different fact, so fail closed.
        if (
            plan.field
            and self.lexical is not None
            and not self.lexical.has_field(plan.field)
        ):
            result.filter_level = "unsupported_field"
            notes.append(f"no_chunks_for_field:{plan.field}")
            result.trace_path = self._write_trace(
                qid=qid,
                question=question,
                plan=plan,
                intent=intent,
                filter_level=result.filter_level,
                levels=levels,
                dense=dense,
                lexical=lexical,
                lexical_used=lexical_used,
                selected=selected,
                best_score=best_score,
                gate_passed=False,
                notes=notes,
            )
            return result

        vector = self._embed(question)
        retrieval = self.settings.retrieval
        # A topic-filtered lookup only has to clear the topic threshold. Without
        # a resolved topic the whole corpus is in play, so demand a much closer
        # semantic match before answering.
        threshold = (
            retrieval.min_score if plan.scheme_slug else retrieval.global_min_score
        )
        accepted: tuple[dict[str, Any] | None, str, list[RetrievedChunk]] | None = None
        chosen: tuple[dict[str, Any] | None, str, list[RetrievedChunk]] | None = None

        for where, label in self._levels(plan):
            found = list(
                self.store.query(
                    vector,
                    where=where or None,
                    n_results=retrieval.candidate_k,
                    include_embeddings=True,
                )
            )
            top = max((chunk.score for chunk in found), default=0.0)
            levels.append(
                {
                    "level": label,
                    "where": where or {},
                    "top_score": round(top, 6),
                    "candidate_count": len(found),
                }
            )
            dense.extend(_trace_chunk(chunk) for chunk in found)
            chosen = (where, label, found)
            if top >= threshold:
                accepted = chosen
                break

        if accepted is not None:
            chosen = accepted
        if chosen is not None:
            _, label, candidates = chosen
            result.filter_level = label
            best_score = max(
                (chunk.score for chunk in candidates), default=0.0
            )

        # BM25 fallback: only when no dense level cleared the gate. It keeps
        # the last metadata filter, so a named scheme can never be widened to
        # the whole corpus.
        if (
            retrieval.lexical_fallback
            and self.lexical is not None
            and plan.scheme_slug is not None
            and best_score < threshold
        ):
            lexical_hits = self.lexical.search(
                question,
                where=(chosen[0] if chosen else None),
                limit=retrieval.candidate_k,
            )
            lexical = [_trace_chunk(chunk) for chunk in lexical_hits]
            if lexical_hits:
                lexical_used = True
                candidates = lexical_hits
                best_score = max(chunk.score for chunk in lexical_hits)
                result.filter_level = (
                    f"{result.filter_level}+lexical"
                    if result.filter_level != "none"
                    else "global+lexical"
                )

        if candidates:
            if lexical_used:
                selected = [
                    replace(chunk, source="lexical")
                    for chunk in candidates[: retrieval.top_k]
                ]
            else:
                selected = mmr_select(
                    candidates,
                    top_k=retrieval.top_k,
                    lambda_=retrieval.mmr_lambda,
                )
            if selected:
                best_score = max(best_score, selected[0].score)

        gate_passed = best_score >= threshold
        if not gate_passed:
            notes.append("gate_failed")

        result.candidates = candidates
        result.selected = selected
        result.best_score = round(best_score, 6)
        result.gate_passed = gate_passed
        result.trace_path = self._write_trace(
            qid=qid,
            question=question,
            plan=plan,
            intent=intent,
            filter_level=result.filter_level,
            levels=levels,
            dense=dense,
            lexical=lexical,
            lexical_used=lexical_used,
            selected=selected,
            best_score=best_score,
            gate_passed=gate_passed,
            notes=notes,
        )
        return result

    # -- internals -------------------------------------------------------
    def _embed(self, question: str) -> np.ndarray:
        try:
            vector = np.asarray(self.embedder.query(question), dtype=np.float32)
        except Exception as exc:  # pragma: no cover - model/runtime failure
            raise StoreError(f"could not embed the question: {exc}") from exc
        if vector.ndim != 1 or vector.size == 0:
            raise StoreError("embedder returned an empty query vector")
        if not np.isfinite(vector).all():
            raise StoreError("embedder returned NaN or infinite values")
        if self.index_dim is not None and vector.shape[0] != self.index_dim:
            raise StoreError(
                f"query vector has {vector.shape[0]} dims but the index has "
                f"{self.index_dim}"
            )
        return vector

    @staticmethod
    def _levels(plan: QueryPlan) -> list[tuple[dict[str, Any] | None, str]]:
        if plan.scheme_slug:
            levels: list[tuple[dict[str, Any] | None, str]] = []
            if plan.field:
                levels.append(
                    (
                        {
                            "scheme_slug": plan.scheme_slug,
                            "field": plan.field,
                        },
                        "scheme+field",
                    )
                )
            levels.append(({"scheme_slug": plan.scheme_slug}, "scheme"))
            return levels
        # No scheme: a single global pass. ``plan.ambiguous_scheme`` still
        # tells the answer layer to ask which scheme before answering.
        return [(None, "global")]

    def _write_trace(
        self,
        *,
        qid: str,
        question: str,
        plan: QueryPlan,
        intent: IntentResult,
        filter_level: str,
        levels: list[dict[str, Any]],
        dense: list[dict[str, Any]],
        lexical: list[dict[str, Any]],
        lexical_used: bool,
        selected: list[RetrievedChunk],
        best_score: float,
        gate_passed: bool,
        notes: list[str],
    ) -> str:
        if not self.write_trace or not self.settings.logging.log_retrieval:
            return ""

        now = datetime.now(timezone.utc)
        payload = {
            "qid": qid,
            "timestamp": now.isoformat(timespec="seconds"),
            "question": question,
            "intent": {
                "label": intent.label,
                "confidence": intent.confidence,
                "signals": list(intent.signals),
            },
            "plan": {
                "intent": plan.intent,
                "scheme_slug": plan.scheme_slug,
                "scheme": plan.scheme,
                "field": plan.field,
                "filters": dict(plan.filters),
                "scheme_candidates": list(plan.scheme_candidates),
                "field_candidates": list(plan.field_candidates),
                "ambiguous_scheme": plan.ambiguous_scheme,
                "ambiguous_field": plan.ambiguous_field,
                "notes": list(plan.notes),
            },
            "filter_level": filter_level,
            "levels": levels,
            "dense": dense,
            "mmr": [
                {
                    "chunk_id": chunk.chunk_id,
                    "score": round(chunk.score, 6),
                }
                for chunk in selected
                if chunk.source == "mmr"
            ],
            "lexical_used": lexical_used,
            "lexical": lexical,
            "gate": {
                "passed": gate_passed,
                "best_score": round(best_score, 6),
                "min_score": (
                    self.settings.retrieval.min_score
                    if plan.scheme_slug
                    else self.settings.retrieval.global_min_score
                ),
            },
            "selected": [_trace_chunk(chunk) for chunk in selected],
            "notes": notes,
        }

        self.trace_dir.mkdir(parents=True, exist_ok=True)
        stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
        path = self.trace_dir / f"{stamp}_{qid}.json"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
        return str(path)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _trace_chunk(chunk: RetrievedChunk) -> dict[str, Any]:
    """Trace view of a chunk: never includes the raw embedding."""
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "score": round(chunk.score, 6),
        "source": chunk.source,
        "metadata": dict(chunk.metadata),
    }


def _cosine(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    if a.shape != b.shape:
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def mmr_select(
    candidates: list[RetrievedChunk],
    *,
    top_k: int,
    lambda_: float = 0.7,
) -> list[RetrievedChunk]:
    """Maximal Marginal Relevance, λ * relevance - (1-λ) * redundancy."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return [replace(chunk, source="mmr") for chunk in candidates]
    if any(chunk.embedding is None for chunk in candidates):
        # Without vectors we cannot measure redundancy; preserve dense order.
        return [replace(chunk, source="mmr") for chunk in candidates[:top_k]]

    selected: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < top_k:
        best_index = remaining[0]
        best_value = float("-inf")
        for index in remaining:
            candidate = candidates[index]
            redundancy = 0.0
            if selected:
                redundancy = max(
                    _cosine(candidate.embedding, candidates[other].embedding)
                    for other in selected
                )
            value = lambda_ * candidate.score - (1.0 - lambda_) * redundancy
            if value > best_value:
                best_value = value
                best_index = index
        selected.append(best_index)
        remaining.remove(best_index)

    return [replace(candidates[index], source="mmr") for index in selected]


__all__ = [
    "CandidateStore",
    "QueryEmbedder",
    "Retriever",
    "mmr_select",
]
