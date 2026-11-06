"""Deterministic Groww help-topic resolution for the facts-only retriever.

The corpus is five official Groww help articles. A question is mapped to at most
one help topic with case-insensitive, punctuation-flat, longest-alias-first
matching, so a phrase such as "demat account opening charges" cannot also match
the shorter stock-delivery topic.

Help articles are prose, so there are no label/value fact fields to resolve.
``field`` stays ``None`` and retrieval is filtered by topic slug only. When no
topic alias matches, the plan falls through to a global pass and the gate decides
whether the corpus actually covers the question.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.intent import detect_intent, detect_out_of_scope
from core.text import normalise_search_text, phrase_pattern
from core.types import IntentResult, QueryPlan

#: Generic official page used when a refusal needs a citation.
HELP_HOME_SLUG = "groww-help-home"
HELP_HOME_TITLE = "Groww Help"
HELP_HOME_URL = "https://groww.in/help"

#: Slugs match the last path segment of each help URL (``ingest.fetch.slug_for``).
TOPIC_NAMES: dict[str, str] = {
    "do-you-charge-for-processing-kyc": "Do you charge for processing KYC?",
    "how-to-change-password-on-groww": "How to change password on Groww?",
    "what-are-account-opening-charges": "What are account opening charges?",
    "when-will-the-stocks-be-delivered-to-my-demat-account-2": (
        "When will the stocks be delivered to my demat account?"
    ),
    "why-is-my-available-to-invest-not-the-same-as-my-available-to-withdraw-1": (
        "Why is my 'Available to Invest' not the same as my 'Available to Withdraw'?"
    ),
}

#: Natural ways users refer to each help topic. Longest alias wins.
TOPIC_ALIASES: dict[str, tuple[str, ...]] = {
    "do-you-charge-for-processing-kyc": (
        "do you charge for processing kyc",
        "charge for processing kyc",
        "charges for processing kyc",
        "processing kyc",
        "kyc charges",
        "kyc charge",
        "kyc fees",
        "kyc fee",
        "kyc cost",
        "kyc free",
        "is kyc free",
        "kyc",
    ),
    "how-to-change-password-on-groww": (
        "how to change password on groww",
        "how do i change my password",
        "how can i change my password",
        "change password on groww",
        "change my password",
        "change password",
        "reset my password",
        "reset password",
        "forgot password",
        "update password",
        "password",
    ),
    "what-are-account-opening-charges": (
        "what are account opening charges",
        "account opening charges",
        "account opening charge",
        "account opening fees",
        "account opening fee",
        "demat account opening charges",
        "demat account opening charge",
        "demat account opening",
        "demat opening",
        "open a demat account",
        "open demat account",
        "opening a demat account",
        "opening demat account",
        "account opening",
        "open an account",
        "open account",
        "free demat account",
    ),
    "when-will-the-stocks-be-delivered-to-my-demat-account-2": (
        "when will the stocks be delivered to my demat account",
        "when will stocks be delivered",
        "when will my stocks reach",
        "when will my stocks reach my demat account",
        "stocks reach my demat",
        "reach my demat account",
        "when will i get my stocks",
        "when will i receive my stocks",
        "when do i get my stocks",
        "stocks be delivered",
        "stocks delivered",
        "stock delivery",
        "delivery of stocks",
        "delivery to my demat account",
        "delivered to my demat account",
        "stocks in my demat account",
        "stocks in my demat",
        "demat delivery",
        "stock settlement",
        "settlement of stocks",
        "t plus 1",
        "t 1",
    ),
    "why-is-my-available-to-invest-not-the-same-as-my-available-to-withdraw-1": (
        "why is my available to invest not the same as my available to withdraw",
        "available to invest not the same as available to withdraw",
        "available to invest vs available to withdraw",
        "available to invest",
        "available to withdraw",
        "available for withdrawal",
        "available balance",
        "withdrawable balance",
        "withdraw my money",
        "withdraw money",
        "withdraw funds",
        "withdraw the amount",
        "withdraw amount",
        "when can i withdraw",
        "why can t i withdraw",
        "unsettled funds",
        "funds available",
        "available funds",
        "withdrawal",
        "settlement",
    ),
}


@dataclass(frozen=True, slots=True)
class AliasMatch:
    key: str
    alias: str
    start: int
    end: int


@dataclass(slots=True)
class Resolution:
    schemes: list[str] = field(default_factory=list)
    scheme_matches: list[AliasMatch] = field(default_factory=list)


def _build_alias_map(aliases: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """Map normalised alias -> canonical key, first key wins."""
    table: dict[str, str] = {}
    for key, values in aliases.items():
        for value in values:
            alias = normalise_search_text(value)
            if alias:
                table.setdefault(alias, key)
    return table


_TOPIC_ALIAS_MAP = _build_alias_map(TOPIC_ALIASES)


def _find_matches(text: str, alias_map: dict[str, str]) -> list[AliasMatch]:
    """All non-overlapping alias hits, longest alias first at each position."""
    found: list[AliasMatch] = []
    for alias, key in alias_map.items():
        for match in phrase_pattern(alias).finditer(text):
            found.append(
                AliasMatch(
                    key=key,
                    alias=alias,
                    start=match.start(),
                    end=match.end(),
                )
            )

    # Longest alias wins; ties are resolved by earlier position, then name.
    found.sort(key=lambda item: (-(item.end - item.start), item.start, item.alias))
    accepted: list[AliasMatch] = []
    for candidate in found:
        if any(
            candidate.start < item.end and item.start < candidate.end
            for item in accepted
        ):
            continue
        accepted.append(candidate)
    accepted.sort(key=lambda item: (item.start, item.end))
    return accepted


def match_topics(question: str) -> list[AliasMatch]:
    """Return non-overlapping help-topic alias hits, in question order."""
    return _find_matches(normalise_search_text(question), _TOPIC_ALIAS_MAP)


def _distinct(matches: list[AliasMatch]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in matches:
        if item.key not in seen:
            seen.add(item.key)
            ordered.append(item.key)
    return ordered


def resolve(question: str) -> Resolution:
    """Resolve help topics without deciding intent."""
    topic_matches = match_topics(question)
    return Resolution(
        schemes=_distinct(topic_matches),
        scheme_matches=topic_matches,
    )


def build_plan(question: str) -> tuple[QueryPlan, IntentResult]:
    """Resolve a question and classify it into the fixed intent order."""
    text = normalise_search_text(question)
    resolution = resolve(question)
    topics = resolution.schemes

    out_of_scope = detect_out_of_scope(text)
    intent = detect_intent(
        question,
        topic_found=bool(topics),
        out_of_scope=out_of_scope,
    )

    topic_slug = topics[0] if len(topics) == 1 else None

    filters: dict[str, str] = {}
    if topic_slug:
        filters["scheme_slug"] = topic_slug

    notes: list[str] = []
    if len(topics) > 1:
        notes.append("multiple_topics:" + ",".join(topics))
    elif not topics:
        notes.append("no_topic")
    if out_of_scope:
        notes.append(f"out_of_scope:{out_of_scope}")

    plan = QueryPlan(
        question=question,
        intent=intent.label,
        scheme_slug=topic_slug,
        scheme=TOPIC_NAMES.get(topic_slug) if topic_slug else None,
        field=None,
        filters=filters,
        scheme_candidates=topics,
        field_candidates=[],
        ambiguous_scheme=len(topics) > 1,
        ambiguous_field=False,
        notes=notes,
    )
    return plan, intent


__all__ = [
    "AliasMatch",
    "HELP_HOME_SLUG",
    "HELP_HOME_TITLE",
    "HELP_HOME_URL",
    "Resolution",
    "TOPIC_ALIASES",
    "TOPIC_NAMES",
    "build_plan",
    "match_topics",
    "resolve",
]
