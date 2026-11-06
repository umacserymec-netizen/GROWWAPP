"""Deterministic intent router for the Groww help assistant.

Rules run in a fixed order so a refusal is explainable and repeatable:
advice → performance → out-of-scope → fact.  How-to questions are *in scope*
because the corpus is official Groww help content, so "how do I change my
password" is a normal factual lookup rather than a refusal.

The router never guesses: when a rule matches, the matched phrase is returned as
a trace signal. PII is intentionally not detected here; it is a separate
pre-LLM guard.
"""

from __future__ import annotations

from core.text import first_phrase, normalise_search_text
from core.types import IntentResult

#: Attempts to influence an investment/trading decision.
ADVICE_PHRASES: tuple[str, ...] = (
    "should i",
    "should we",
    "should you",
    "should one",
    "good to buy",
    "worth buying",
    "buy now",
    "buy this",
    "sell now",
    "sell this",
    "when to sell",
    "switch to",
    "switch from",
    "hold or sell",
    "keep holding",
    "recommend",
    "best",
    "worth investing",
    "good investment",
    "good stock",
    "good fund",
    "good for me",
    "suitable for me",
    "is it safe",
    "safe bet",
    "guaranteed",
    "good time",
    "my portfolio",
    "review my",
    "how much should",
    "which stock",
    "which fund",
    "which scheme",
    "which is better",
    "better than",
    "target price",
    "multibagger",
)

#: Return / CAGR / prediction requests. Refused without numbers.
PERFORMANCE_PHRASES: tuple[str, ...] = (
    "returns",
    "return on",
    "return from",
    "return of",
    "return percentage",
    "fund return",
    "expected return",
    "expected returns",
    "what return",
    "how much return",
    "cagr",
    "xirr",
    "performance",
    "outperform",
    "underperform",
    "nav history",
    "annualised return",
    "annualized return",
    "absolute return",
    "1 year return",
    "3 year return",
    "5 year return",
    "since inception return",
    "past return",
    "how much will i earn",
    "how much can i earn",
    "future value",
    "profit",
    "predict",
    "go up",
    "go down",
    "market timing",
)

#: Procedural phrasings. Kept as a trace signal; help topics answer them.
HOWTO_PHRASES: tuple[str, ...] = (
    "how do i",
    "how to",
    "steps to",
    "step by step",
    "where can i",
    "where do i",
    "process for",
    "procedure",
    "guide me",
    "walk me through",
)

#: Topics outside the Groww platform-help corpus (mutual funds and other apps).
OUT_OF_SCOPE_PHRASES: tuple[str, ...] = (
    "hdfc",
    "mutual fund",
    "mutual funds",
    "systematic investment plan",
    "sip",
    "nav",
    "net asset value",
    "expense ratio",
    "exit load",
    "aum",
    "asset under management",
    "elss",
    "folio",
    "redemption",
    "redeem",
    "lumpsum",
    "mutual fund scheme",
    "zerodha",
    "upstox",
    "angel one",
    "5paisa",
    "paytm money",
    "indmoney",
    "icici direct",
    "kotak securities",
    "crypto",
    "bitcoin",
    "forex",
    "futures",
    "options trading",
)


def detect_out_of_scope(text: str) -> str | None:
    """Return the first phrase that is outside Groww platform help."""
    return first_phrase(text, OUT_OF_SCOPE_PHRASES)


def detect_intent(
    question: str,
    *,
    topic_found: bool = False,
    out_of_scope: str | None = None,
    **_legacy: object,
) -> IntentResult:
    """Classify ``question`` using the documented precedence."""
    text = normalise_search_text(question)

    for label, phrases in (
        ("advice", ADVICE_PHRASES),
        ("performance", PERFORMANCE_PHRASES),
    ):
        hit = first_phrase(text, phrases)
        if hit is not None:
            return IntentResult(
                label=label,  # type: ignore[arg-type]
                confidence=0.95,
                signals=[f"{label}:{hit.strip()}"],
            )

    if out_of_scope:
        return IntentResult(
            label="out_of_scope",
            confidence=0.9,
            signals=[f"out_of_scope:{out_of_scope}"],
        )

    howto = first_phrase(text, HOWTO_PHRASES)
    signals = ["fact:howto" if howto else ("fact:topic" if topic_found else "fact:global")]
    if howto:
        signals.append(f"howto:{howto.strip()}")
    return IntentResult(
        label="fact",
        confidence=0.8 if topic_found else 0.6,
        signals=signals,
    )


__all__ = [
    "ADVICE_PHRASES",
    "HOWTO_PHRASES",
    "OUT_OF_SCOPE_PHRASES",
    "PERFORMANCE_PHRASES",
    "detect_intent",
    "detect_out_of_scope",
]
