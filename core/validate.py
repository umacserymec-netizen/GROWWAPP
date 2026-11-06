"""Deterministic answer validation (no model, no network).

The validator is the last line of defence before text reaches the UI. It checks
the contract in ``ARCHITECTURE.md`` §6.4 and returns codes, not free-form model
feedback. On failure the caller gets one repair attempt; a second failure is
handled by ``core/fallback.py``.

The freshness line is never trusted from the model. :func:`append_freshness`
removes anything the model wrote and appends the date from chunk metadata, so a
hallucinated date cannot survive.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import datetime
from urllib.parse import urlsplit

from core.types import RetrievedChunk, Verdict

FRESHNESS_PREFIX = "Last updated from sources:"
MAX_ANSWER_CHARS = 600

_MD_LINK = re.compile(r"!?\[([^\]]*)\]\(([^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\)")
_BARE_URL = re.compile(r"https?://[^\s<>\]\)]+", re.IGNORECASE)
_FRESHNESS = re.compile(
    r"(?im)(?:\s*[·|]\s*)?\bLast updated(?:\s+from\s+sources)?\s*:[^\n]*"
)
_ABBREVIATIONS = re.compile(
    r"\b(e\.g|i\.e|vs|Mr|Mrs|Ms|Dr|Rs|No|Ltd|Pvt|Inc|Co|Corp|approx|viz)\.",
    re.IGNORECASE,
)
_INITIAL = re.compile(r"\b([A-Z])\.")
_DECIMAL = re.compile(r"(\d)\.(\d)")

_ADVICE = re.compile(
    r"\b(?:should|shouldn't|recommend|recommends|recommended|recommendation|"
    r"best|guaranteed|guarantee|safe\s+bet|will\s+give|advise|advice)\b",
    re.IGNORECASE,
)
_PERFORMANCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:cagr|xirr|annualised|annualized|absolute\s+return|fund\s+return|"
        r"past\s+return|expected\s+return|return\s+on\s+investment)\b"
        r"[^.\n]{0,40}\d",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d+(?:\.\d+)?\s*%[^.\n]{0,30}\b(?:returns?|cagr|xirr|profit|gain)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d+\s*[- ]?(?:year|yr)s?\s+(?:returns?|performance)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\breturns?\b[^.\n]{0,24}\d+(?:\.\d+)?\s*%",
        re.IGNORECASE,
    ),
)


# ---------------------------------------------------------------------------
# freshness
# ---------------------------------------------------------------------------


def format_source_date(value: str | None) -> str:
    """Render an ISO ``fetched_at`` as ``01 Nov 2026`` (PRD §6.1)."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.strftime("%d %b %Y")
    except ValueError:
        pass
    for fmt in ("%d %b %Y", "%d %B %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%d %b %Y")
        except ValueError:
            continue
    return raw


def strip_freshness(text: str) -> str:
    """Remove any model-written freshness line before validation/rendering."""
    return _FRESHNESS.sub("", text or "").strip()


def append_freshness(text: str, fetched_at: str | None, *, show: bool = True) -> str:
    body = strip_freshness(text).strip()
    if not show:
        return body
    date = format_source_date(fetched_at)
    if not date:
        return body
    line = f"{FRESHNESS_PREFIX} {date}"
    return f"{body} · {line}" if body else line


# ---------------------------------------------------------------------------
# sentence splitting
# ---------------------------------------------------------------------------


def _mask_url(match: re.Match[str]) -> str:
    """Replace a URL but keep its sentence-ending punctuation."""
    url = match.group(0)
    trailing = ""
    while url and url[-1] in ".,;:!?":
        trailing = url[-1] + trailing
        url = url[:-1]
    return f"URL{trailing}"


def split_sentences(text: str) -> list[str]:
    """Abbreviation-aware split that ignores URLs and decimal points."""
    clean = strip_freshness(text)
    clean = _MD_LINK.sub("LINK", clean)
    clean = _BARE_URL.sub(_mask_url, clean)
    clean = _ABBREVIATIONS.sub(r"\1<abbr>", clean)
    clean = _INITIAL.sub(r"\1<abbr>", clean)
    clean = _DECIMAL.sub(r"\1<dot>\2", clean)
    parts = re.split(r"(?<=[.!?])\s+|\n+", clean)
    return [
        part.strip()
        for part in parts
        if re.search(r"[A-Za-z0-9\u0900-\u097F]", part or "")
    ]


# ---------------------------------------------------------------------------
# citations
# ---------------------------------------------------------------------------


def extract_urls(text: str) -> list[str]:
    found: list[str] = []
    for match in _MD_LINK.finditer(text or ""):
        found.append(match.group(2).strip())
    for match in _BARE_URL.finditer(text or ""):
        url = match.group(0).rstrip(".,;:!?)]}'\"")
        if url:
            found.append(url)
    # Preserve order, drop exact duplicates.
    seen: set[str] = set()
    ordered: list[str] = []
    for url in found:
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _host_allowed(host: str, allowed_domains: Iterable[str]) -> bool:
    if not host:
        return False
    for domain in allowed_domains:
        candidate = domain.lower().strip(".")
        if host == candidate or host.endswith("." + candidate):
            return True
    return False


def _url_in_context(url: str, chunks: Sequence[RetrievedChunk]) -> bool:
    target = url.rstrip("/")
    for chunk in chunks:
        source = str((chunk.metadata or {}).get("source_url") or "").rstrip("/")
        if source and source == target:
            return True
    return False


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def validate_answer(
    text: str,
    chunks: Sequence[RetrievedChunk],
    *,
    allowed_domains: Sequence[str],
    max_sentences: int = 3,
    require_citation: bool = True,
    field: str | None = None,
) -> Verdict:
    """Validate a model draft; return codes the repair prompt can quote."""
    del field  # kept for the documented call signature / future field rules
    clean = strip_freshness(text)
    codes: list[str] = []
    problems: list[str] = []

    def fail(code: str, message: str) -> None:
        if code not in codes:
            codes.append(code)
            problems.append(message)

    if not clean:
        fail("empty_answer", "Answer is empty.")

    if "NOT_IN_CONTEXT" in clean.upper():
        fail(
            "not_in_context_token",
            "Use the not-found path instead of returning the sentinel to the user.",
        )

    urls = extract_urls(clean)
    if require_citation:
        if len(urls) != 1:
            fail(
                "url_count",
                f"Exactly one citation URL is required; found {len(urls)}.",
            )
        else:
            url = urls[0]
            host = _host(url)
            if not _host_allowed(host, allowed_domains):
                fail(
                    "url_host",
                    f"Citation host '{host or '<none>'}' is not in the allowed domain list.",
                )
            if not _url_in_context(url, chunks):
                fail(
                    "url_not_in_context",
                    "Citation URL must be one of the selected source URLs.",
                )

    if _ADVICE.search(clean):
        fail(
            "advice_language",
            "Remove advice words (should, recommend, best, guaranteed, safe bet).",
        )

    performance_text = re.sub(
        r"total\s+return\s+index", "TotalReturnIndex", clean, flags=re.IGNORECASE
    )
    if any(pattern.search(performance_text) for pattern in _PERFORMANCE_PATTERNS):
        fail(
            "performance_figures",
            "Do not state, calculate, or compare returns or performance figures.",
        )

    sentences = split_sentences(clean)
    if max_sentences >= 1 and len(sentences) > max_sentences:
        fail(
            "too_many_sentences",
            f"Use at most {max_sentences} sentences including the citation; "
            f"found {len(sentences)}.",
        )

    if len(clean) > MAX_ANSWER_CHARS:
        fail(
            "too_long",
            f"Keep the answer under {MAX_ANSWER_CHARS} characters.",
        )

    return Verdict(ok=not codes, text=clean, problems=problems, codes=codes)


__all__ = [
    "FRESHNESS_PREFIX",
    "MAX_ANSWER_CHARS",
    "append_freshness",
    "extract_urls",
    "format_source_date",
    "split_sentences",
    "strip_freshness",
    "validate_answer",
]
