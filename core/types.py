"""Shared dataclasses for every stage.

Plain dataclasses (not pydantic) so they serialize with ``dataclasses.asdict``
and stay cheap to build in tests. Field names are the contract used by the
JSON artifacts under ``data/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Literal

Intent = Literal["fact", "howto", "advice", "performance", "pii", "out_of_scope"]
PageType = Literal["help", "scheme", "amc", "education", "process", "factsheet"]
ChunkType = Literal["fact_atom", "section_window"]


@dataclass(slots=True)
class Section:
    title: str
    body: str
    level: int = 2


@dataclass(slots=True)
class SourceRow:
    id: str
    role: str
    scheme: str
    url: str
    page_type: PageType
    priority: int = 1
    notes: str = ""


@dataclass(slots=True)
class ManifestEntry:
    id: str
    url: str
    final_url: str
    status: int | None
    content_type: str
    bytes: int
    sha256: str
    fetched_at: str
    slug: str
    ok: bool
    error: str | None = None


@dataclass(slots=True)
class CleanDoc:
    scheme: str
    scheme_slug: str
    source_title: str
    source_url: str
    page_type: PageType
    fetched_at: str
    sections: list[Section]
    raw_sha256: str = ""
    char_len: int = 0


@dataclass(slots=True)
class LoadedDoc:
    """A clean document that passed the Stage-2c load gate."""

    doc_id: str
    source_id: str
    scheme: str
    scheme_slug: str
    source_title: str
    source_url: str
    page_type: PageType
    fetched_at: str
    sections: list[Section]
    raw_sha256: str = ""
    content_sha256: str = ""
    char_len: int = 0

    @property
    def section_count(self) -> int:
        return len(self.sections)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    chunk_type: ChunkType
    scheme: str
    scheme_slug: str
    field: str
    section_title: str
    text: str
    source_url: str
    source_title: str
    page_type: PageType
    fetched_at: str
    char_len: int = 0


@dataclass(slots=True)
class IntentResult:
    label: Intent
    confidence: float
    signals: list[str] = dc_field(default_factory=list)


@dataclass(slots=True)
class QueryPlan:
    question: str
    intent: Intent
    scheme_slug: str | None = None
    scheme: str | None = None
    field: str | None = None
    filters: dict[str, str] = dc_field(default_factory=dict)
    scheme_candidates: list[str] = dc_field(default_factory=list)
    field_candidates: list[str] = dc_field(default_factory=list)
    ambiguous_scheme: bool = False
    ambiguous_field: bool = False
    notes: list[str] = dc_field(default_factory=list)


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = dc_field(default_factory=dict)
    source: Literal["dense", "lexical", "mmr"] = "dense"
    embedding: list[float] | None = dc_field(
        default=None, repr=False, compare=False
    )


@dataclass(slots=True)
class RetrievalResult:
    plan: QueryPlan
    candidates: list[RetrievedChunk] = dc_field(default_factory=list)
    selected: list[RetrievedChunk] = dc_field(default_factory=list)
    best_score: float = 0.0
    gate_passed: bool = False
    filter_level: str = "none"
    trace_path: str = ""
    qid: str = ""


@dataclass(slots=True)
class PiiScan:
    hit: bool
    kinds: list[str] = dc_field(default_factory=list)
    redacted: str = ""


@dataclass(slots=True)
class Verdict:
    ok: bool
    text: str
    problems: list[str] = dc_field(default_factory=list)
    codes: list[str] = dc_field(default_factory=list)


@dataclass(slots=True)
class AnswerRecord:
    qid: str
    question: str
    status: Literal["answered", "refused", "not_found", "error"]
    answer_text: str
    citation_url: str | None
    last_updated: str
    intent: Intent = "fact"
    scheme: str | None = None
    field: str | None = None
    chunk_ids: list[str] = dc_field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    validation: dict[str, Any] = dc_field(default_factory=dict)
    provider: str | None = None
    model: str | None = None
    prompt_version: str = "answer_v1"
    latency_ms: dict[str, int] = dc_field(default_factory=dict)
    trace_id: str | None = None
    # Step-8 inspector fields. ``question`` always stores the safe version: on
    # a PII hit it contains only the redacted text, never the raw value.
    question_redacted: str | None = None
    citation_title: str | None = None
    trace_path: str | None = None
    prompt: dict[str, str] = dc_field(default_factory=dict)
    raw_answer: str = ""
    repair_answer: str = ""
    fallback_used: bool = False
    fallback_reason: str | None = None
    provider_attempts: list[dict[str, Any]] = dc_field(default_factory=list)
    error_code: str | None = None
