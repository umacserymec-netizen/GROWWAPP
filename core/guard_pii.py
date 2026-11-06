"""Input-only PII guard.

The assistant never needs PAN, Aadhaar, account, OTP, email, or phone data.
This module scans the *question* before any retrieval or provider call, and
returns a redacted copy. The redacted copy is the only version allowed into a
log or an answer record when a hit occurs; the raw question is dropped.

The patterns deliberately target identifiers, not ordinary finance text:
``1,00,000``, ``12 months`` and ``0.55%`` must stay clean.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.types import PiiScan

#: Ordered so the most specific identifier wins the "kinds" list.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "pan",
        re.compile(r"(?<![A-Z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Z0-9])", re.IGNORECASE),
    ),
    (
        "aadhaar",
        re.compile(
            r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)"
        ),
    ),
    (
        "email",
        re.compile(
            r"(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![A-Z0-9.-])",
            re.IGNORECASE,
        ),
    ),
    (
        "otp",
        re.compile(r"\botp\b[^\d\n]{0,16}\d{6}\b", re.IGNORECASE),
    ),
    (
        "otp",
        re.compile(r"\b\d{6}\b[^\n]{0,12}\botp\b", re.IGNORECASE),
    ),
    (
        "account",
        re.compile(
            r"\b(?:a/c|acct|account)\b[^\d\n]{0,24}\d{9,18}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "phone",
        re.compile(
            r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"
        ),
    ),
)

#: Long bare digit runs are only treated as account-like when a nearby word
#: says so (handled by the ``account`` pattern); this guard catches a 12+ run
#: with no separators only if it looks like an account, not an amount.
_LONG_DIGITS = re.compile(r"(?<!\d)\d{12,19}(?!\d)")


@dataclass(slots=True)
class PiiDecision:
    """Result of the input scan, including a safe version of the text."""

    scan: PiiScan
    safe_text: str = ""
    spans: list[tuple[int, int]] = field(default_factory=list)


def scan_pii(text: str) -> PiiDecision:
    """Return every identifier hit and a fully redacted copy of ``text``."""
    question = text or ""
    spans: list[tuple[int, int]] = []
    kinds: list[str] = []

    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(question):
            spans.append(match.span())
            if kind not in kinds:
                kinds.append(kind)

    for match in _LONG_DIGITS.finditer(question):
        # A 12-digit Aadhaar is already covered; an 18-digit account may not
        # have the word "account" next to it. Avoid 12-month style values by
        # requiring 13+ digits here.
        if len(re.sub(r"\D", "", match.group(0))) >= 13:
            spans.append(match.span())
            if "account" not in kinds:
                kinds.append("account")

    if not spans:
        return PiiDecision(scan=PiiScan(hit=False, kinds=[], redacted=question), safe_text=question)

    merged = _merge_spans(spans)
    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        parts.append(question[cursor:start])
        parts.append("[redacted]")
        cursor = end
    parts.append(question[cursor:])
    redacted = "".join(parts)

    return PiiDecision(
        scan=PiiScan(hit=True, kinds=kinds, redacted=redacted),
        safe_text=redacted,
        spans=merged,
    )


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(spans)
    merged: list[list[int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


__all__ = ["PiiDecision", "scan_pii"]
