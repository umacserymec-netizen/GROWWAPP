"""Canonical fact fields shared by chunking, resolution, and answer templates.

One table, one vocabulary. A label seen on a page is normalised and looked up
here; the returned :class:`FieldSpec` decides the atom's metadata ``field`` and
the human label used in answer text. Keeping this in ``core/`` means the
retriever and the chunker can never drift apart.

The Groww Help corpus is prose, not label/value fund facts, so help pages are
deliberately excluded from every :class:`FieldSpec` page-type list. Help topics
produce ``section_window`` chunks only; this table remains for the legacy
scheme/AMC adapters and shared helper functions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.types import PageType


@dataclass(frozen=True, slots=True)
class FieldSpec:
    key: str
    label: str
    aliases: tuple[str, ...]
    page_types: tuple[PageType, ...] = ("scheme", "amc")
    numeric: bool = False


FIELD_SPECS: dict[str, FieldSpec] = {
    "expense_ratio": FieldSpec(
        "expense_ratio",
        "Expense ratio",
        ("expense ratio", "total expense ratio", "ter", "expense ratio (direct)"),
    ),
    "exit_load": FieldSpec(
        "exit_load",
        "Exit load",
        ("exit load", "exit load (%)"),
    ),
    "min_sip": FieldSpec(
        "min_sip",
        "Minimum SIP",
        (
            "min. for sip",
            "min sip",
            "minimum sip",
            "minimum sip investment",
            "sip",
        ),
        numeric=True,
    ),
    "min_investment": FieldSpec(
        "min_investment",
        "Minimum investment",
        (
            "min. for 1st investment",
            "min investment",
            "minimum investment",
            "minimum lumpsum investment",
            "min. for lumpsum investment",
            "minimum lumpsum",
            "min. lumpsum",
        ),
        numeric=True,
    ),
    "lock_in": FieldSpec(
        "lock_in",
        "Lock-in",
        ("lock-in", "lock in", "lock-in period", "lock in period", "lockin"),
    ),
    "riskometer": FieldSpec(
        "riskometer",
        "Riskometer",
        ("riskometer", "risk level", "risk"),
    ),
    "benchmark": FieldSpec(
        "benchmark",
        "Benchmark",
        ("benchmark", "fund benchmark", "benchmark index"),
    ),
    "aum": FieldSpec(
        "aum",
        "AUM",
        (
            "aum",
            "fund size (aum)",
            "fund size",
            "asset under management",
            "asset under management(aum)",
        ),
        numeric=True,
    ),
    "nav": FieldSpec(
        "nav",
        "NAV",
        ("nav", "latest nav"),
        numeric=True,
    ),
    "rating": FieldSpec("rating", "Rating", ("rating", "groww rating"), numeric=True),
    "category": FieldSpec("category", "Category", ("category", "scheme category")),
    "fund_manager": FieldSpec(
        "fund_manager",
        "Fund manager",
        ("fund manager", "fund managers"),
        page_types=("scheme",),
    ),
    "launch_date": FieldSpec(
        "launch_date",
        "Launch date",
        ("launch date",),
    ),
    "scheme_count": FieldSpec(
        "scheme_count",
        "Number of schemes",
        ("no of schemes", "number of schemes", "schemes"),
        page_types=("amc",),
        numeric=True,
    ),
    "amc_age": FieldSpec(
        "amc_age",
        "AMC age",
        ("amc age", "age"),
        page_types=("amc",),
        numeric=True,
    ),
    "incorporation_date": FieldSpec(
        "incorporation_date",
        "Incorporation date",
        ("amc incorporation date", "date of incorporation"),
        page_types=("amc",),
    ),
    "amc_name": FieldSpec(
        "amc_name",
        "AMC name",
        ("amc", "mutual fund name", "asset management company name"),
    ),
}

#: Fields every scheme page must contribute at least one atom for.
REQUIRED_SCHEME_FIELDS: tuple[str, ...] = (
    "expense_ratio",
    "exit_load",
    "min_sip",
    "riskometer",
    "benchmark",
)

#: ELSS-style scheme pages would add a statutory lock-in; this help-only corpus
#: has no scheme pages, so the per-slug table is intentionally empty.
REQUIRED_SCHEME_FIELDS_BY_SLUG: dict[str, tuple[str, ...]] = {}

#: AMC pages carry fund-house facts instead of scheme facts.
REQUIRED_AMC_FIELDS: tuple[str, ...] = ("aum", "incorporation_date")


def required_fields(page_type: str, scheme_slug: str) -> tuple[str, ...]:
    """Return the fields the chunker must cover for one loaded document.

    Help pages are prose topics with no required label/value atoms. Returning an
    empty tuple keeps the chunk coverage gate meaningful for help pages instead
    of failing every document for missing scheme fields.
    """
    if page_type == "help":
        return ()
    if page_type == "amc":
        return REQUIRED_AMC_FIELDS
    extra = REQUIRED_SCHEME_FIELDS_BY_SLUG.get(scheme_slug, ())
    return REQUIRED_SCHEME_FIELDS + extra


def normalise_label(value: str) -> str:
    """Lower-case and flatten a page label so lookups are stable."""
    text = value.replace("\u00a0", " ").replace("**", "")
    text = text.strip(" \t-*_")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" :#").lower()


_LABEL_INDEX: dict[str, FieldSpec] = {}
for _spec in FIELD_SPECS.values():
    for _alias in (_spec.key, _spec.label, *_spec.aliases):
        _LABEL_INDEX.setdefault(normalise_label(_alias), _spec)


def match_field(label: str, page_type: str | None = None) -> FieldSpec | None:
    """Resolve a page label to its canonical field (longest/exact match only)."""
    spec = _LABEL_INDEX.get(normalise_label(label))
    if spec is None:
        return None
    if page_type is not None and page_type not in spec.page_types:
        return None
    return spec


def field_labels() -> dict[str, str]:
    """Canonical field -> human label, for prompts and fallback templates."""
    return {key: spec.label for key, spec in FIELD_SPECS.items()}


def field_synonyms() -> dict[str, tuple[str, ...]]:
    """Canonical field -> accepted labels, for the query resolver (Stage 6)."""
    return {key: spec.aliases for key, spec in FIELD_SPECS.items()}


__all__ = [
    "FIELD_SPECS",
    "REQUIRED_AMC_FIELDS",
    "REQUIRED_SCHEME_FIELDS",
    "REQUIRED_SCHEME_FIELDS_BY_SLUG",
    "FieldSpec",
    "field_labels",
    "field_synonyms",
    "match_field",
    "normalise_label",
    "required_fields",
]
