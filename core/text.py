"""Small text helpers shared by intent detection, resolution, and BM25.

Everything is deliberately dependency-free and deterministic so query planning
can be unit-tested without models or a vector store.
"""

from __future__ import annotations

import re

_NON_WORD = re.compile(r"[^0-9a-z]+")
_SPACES = re.compile(r"\s+")


def normalise_search_text(value: str) -> str:
    """Lower-case, flatten punctuation, and collapse spaces.

    ``KYC Fees`` and ``kyc fees`` become the same searchable string, so aliases
    can be written naturally without regex punctuation traps.
    """
    text = (value or "").replace("\u00a0", " ").lower()
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Word-boundary pattern for an already-normalised phrase."""
    return re.compile(r"(?<![0-9a-z])" + re.escape(phrase) + r"(?![0-9a-z])")


def find_phrase(text: str, phrase: str) -> int | None:
    """Return the start offset of ``phrase`` in normalised ``text``."""
    match = phrase_pattern(phrase).search(text)
    return match.start() if match else None


def contains_phrase(text: str, phrase: str) -> bool:
    return find_phrase(text, phrase) is not None


def first_phrase(text: str, phrases: tuple[str, ...]) -> str | None:
    """Return the first (by list order) phrase present in ``text``."""
    for phrase in phrases:
        if contains_phrase(text, normalise_search_text(phrase)):
            return phrase
    return None


__all__ = [
    "contains_phrase",
    "find_phrase",
    "first_phrase",
    "normalise_search_text",
    "phrase_pattern",
]
