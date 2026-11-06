"""BM25 fallback over the same chunks that were embedded.

Dense vectors are good at paraphrase ("cost of the fund") but weak on exact
tokens such as ``1%``, ``12 months`` or ``NIFTY 100``. This module keeps a
small in-memory BM25 index over ``chunks.jsonl`` so the retriever can catch
those cases without a second model.

Scores are normalised by the best hit for the query, which keeps the gate's
``min_score`` meaningful for both dense and lexical results. Metadata filters
(``scheme_slug`` / ``field``) are applied exactly like Chroma's ``where``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from core.store import flat_metadata
from core.text import normalise_search_text
from core.types import Chunk, RetrievedChunk

_TOKEN_RE = re.compile(r"[0-9a-z]+")

#: Question words and glue that carry no retrieval signal.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "being",
        "between",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "done",
        "for",
        "from",
        "get",
        "give",
        "had",
        "has",
        "have",
        "he",
        "her",
        "here",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "know",
        "many",
        "may",
        "me",
        "might",
        "mine",
        "more",
        "most",
        "much",
        "must",
        "my",
        "need",
        "of",
        "on",
        "or",
        "our",
        "over",
        "please",
        "shall",
        "she",
        "should",
        "show",
        "so",
        "some",
        "tell",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "under",
        "very",
        "want",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)


def tokenize(value: str) -> list[str]:
    """Lower-case alphanumeric tokens with question-word stopwords removed."""
    return [
        token
        for token in _TOKEN_RE.findall(normalise_search_text(value))
        if token not in STOPWORDS
    ]


def _document_text(chunk: Chunk) -> str:
    field_words = chunk.field.replace("_", " ") if chunk.field else ""
    return " ".join(
        part
        for part in (
            chunk.scheme,
            chunk.section_title,
            field_words,
            chunk.text,
        )
        if part
    )


class LexicalIndex:
    """In-memory BM25 index with the same flat metadata as Chroma."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks: list[Chunk] = list(chunks)
        self.metadata: list[dict[str, Any]] = [
            flat_metadata(chunk) for chunk in self.chunks
        ]
        self.corpus: list[list[str]] = [
            tokenize(_document_text(chunk)) for chunk in self.chunks
        ]
        self._fields = {chunk.field for chunk in self.chunks if chunk.field}

        if self.chunks:
            from rank_bm25 import BM25Plus

            self._bm25 = BM25Plus(self.corpus)
        else:  # pragma: no cover - guarded by callers, kept for safety
            self._bm25 = None

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def fields(self) -> set[str]:
        """Fields that actually have at least one chunk."""
        return set(self._fields)

    def has_field(self, field: str | None) -> bool:
        return bool(field) and field in self._fields

    def search(
        self,
        query: str,
        *,
        where: dict[str, Any] | None = None,
        limit: int = 24,
    ) -> list[RetrievedChunk]:
        """Return up to ``limit`` hits with scores normalised to the top hit."""
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if not self.chunks or self._bm25 is None:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        raw_scores = [float(score) for score in self._bm25.get_scores(tokens)]
        wanted = set(tokens)
        indices = [
            index
            for index, score in enumerate(raw_scores)
            if score > 0
            and wanted.intersection(self.corpus[index])
            and _matches_where(self.metadata[index], where)
        ]
        if not indices:
            return []

        indices.sort(key=lambda index: (-raw_scores[index], index))
        indices = indices[:limit]
        best = raw_scores[indices[0]]
        if best <= 0:  # pragma: no cover - impossible after the score filter
            return []

        results: list[RetrievedChunk] = []
        for index in indices:
            chunk = self.chunks[index]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=raw_scores[index] / best,
                    metadata=dict(self.metadata[index]),
                    source="lexical",
                )
            )
        return results

    def raw_scores(self, query: str) -> list[float]:
        """Expose unnormalised scores for tests and traces."""
        if not self.chunks or self._bm25 is None:
            return []
        return [float(score) for score in self._bm25.get_scores(tokenize(query))]


def _matches_where(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    for key, expected in where.items():
        actual = metadata.get(key)
        if isinstance(expected, dict):
            # Chroma supports operators such as {"$in": [...]}; mirror the
            # common equality/containment forms here.
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$ne" in expected and actual == expected["$ne"]:
                return False
        elif actual != expected:
            return False
    return True


__all__ = ["LexicalIndex", "STOPWORDS", "tokenize"]
