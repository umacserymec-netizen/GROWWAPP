"""Validator regressions for help prose and citation rules."""

from __future__ import annotations

from core.types import RetrievedChunk
from core.validate import validate_answer

URL = "https://groww.in/help/payments/payments/why-is-my-available-to-invest-not-the-same-as-my-available-to-withdraw-1"
CITATION = f"[Source: Groww Help]({URL})"


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="sha1:test",
        text="Any amount added is available to invest immediately.",
        score=1.0,
        metadata={"source_url": URL, "source_title": "Groww Help"},
    )


def test_plain_returns_to_balance_is_not_a_performance_claim() -> None:
    verdict = validate_answer(
        f"Money returns to your balance after one working day. {CITATION}",
        [_chunk()],
        allowed_domains=["groww.in"],
        max_sentences=3,
    )
    assert verdict.ok, verdict.codes


def test_percentage_return_is_rejected() -> None:
    verdict = validate_answer(
        f"The fund gave 12% returns last year. {CITATION}",
        [_chunk()],
        allowed_domains=["groww.in"],
        max_sentences=3,
    )
    assert not verdict.ok
    assert "performance_figures" in verdict.codes


def test_citation_must_come_from_context() -> None:
    verdict = validate_answer(
        "Opening a demat account is free. [Source: Other](https://groww.in/help/other)",
        [_chunk()],
        allowed_domains=["groww.in"],
        max_sentences=3,
    )
    assert not verdict.ok
    assert "url_not_in_context" in verdict.codes
