"""Online answer orchestrator: guard → plan → retrieve → gate → generate → validate.

This is the only module the UI/API needs. Every branch returns an
:class:`~core.types.AnswerRecord`, including refusals and index/runtime errors,
so the caller never sees a raw exception or a partial answer.

Key guarantees
--------------
* PII is scanned before retrieval, so a hit never reaches Chroma or a provider.
* Advice/performance/out-of-scope intents short-circuit with a citation.
* A failed retrieval gate never calls the LLM.
* A model draft is validated in code; one repair is allowed, then an extractive
  answer is built from the best retrieved help passage.
* With no provider key the same pipeline runs offline using that extractive path.
* Every answer is persisted to ``data/07_answers/<qid>.json`` for the inspector.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from config.settings import Settings, load_settings
from core.chunks import read_chunks
from core.fallback import build_extractive_answer
from core.guard_pii import PiiDecision, scan_pii
from core.ids import new_qid
from core.llm import LLMError, RetryingClient, build_retrying_client
from core.prompt import PROMPT_VERSION, build_prompt, build_repair_prompt
from core.resolve import (
    HELP_HOME_SLUG,
    HELP_HOME_TITLE,
    HELP_HOME_URL,
    TOPIC_NAMES,
)
from core.retriever import Retriever
from core.types import (
    AnswerRecord,
    Chunk,
    PiiScan,
    QueryPlan,
    RetrievalResult,
    RetrievedChunk,
)
from core.validate import append_freshness, validate_answer

_SCOPE = (
    "KYC, password, account opening charges, stock delivery, and "
    "available balance help"
)


class AnswerRetriever(Protocol):
    def retrieve(self, question: str) -> RetrievalResult: ...


@dataclass(frozen=True, slots=True)
class Citation:
    scheme_slug: str
    scheme: str
    title: str
    url: str
    fetched_at: str


@dataclass(slots=True)
class _Run:
    """Mutable bookkeeping for one request."""

    qid: str
    question: str
    started: float
    latency: dict[str, int]
    attempts: list[dict[str, Any]]
    prompt: dict[str, str]
    trace_path: str | None = None
    raw_answer: str = ""
    repair_answer: str = ""
    fallback_reason: str | None = None
    error_code: str | None = None


class AnswerEngine:
    """Reusable answer session over one retriever and one LLM client."""

    def __init__(
        self,
        settings: Settings,
        retriever: AnswerRetriever,
        *,
        llm: RetryingClient | None = None,
        chunks: Sequence[Chunk] | None = None,
        answer_dir: Path | None = None,
        prompt_dir: Path | None = None,
        write_answer: bool = True,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.llm = llm
        self.chunks = list(chunks or [])
        self.answer_dir = Path(answer_dir or settings.logging.trace_dir)
        self.prompt_dir = Path(prompt_dir or settings.paths.prompt_dir)
        self.write_answer = write_answer
        self.citations = _build_citations(self.chunks)
        self.home = self.citations.get(HELP_HOME_SLUG) or Citation(
            scheme_slug=HELP_HOME_SLUG,
            scheme=HELP_HOME_TITLE,
            title=HELP_HOME_TITLE,
            url=HELP_HOME_URL,
            fetched_at="",
        )

    # -- construction ----------------------------------------------------
    @classmethod
    def open(
        cls,
        settings: Settings,
        *,
        retriever: AnswerRetriever | None = None,
        llm: RetryingClient | None | object = ...,
        use_llm: bool = True,
        chunks: Sequence[Chunk] | None = None,
        answer_dir: Path | None = None,
        write_answer: bool = True,
    ) -> AnswerEngine:
        active_retriever = retriever if retriever is not None else Retriever.open(settings)
        if chunks is None:
            lexical = getattr(active_retriever, "lexical", None)
            chunks = list(getattr(lexical, "chunks", []) or [])
            if not chunks:
                path = settings.paths.data_dir / "03_chunks" / "chunks.jsonl"
                if path.exists():
                    chunks = read_chunks(path)
        active_llm = (
            build_retrying_client(settings)
            if llm is Ellipsis and use_llm
            else (llm if llm is not Ellipsis else None)
        )
        return cls(
            settings,
            active_retriever,
            llm=active_llm,
            chunks=chunks,
            answer_dir=answer_dir,
            write_answer=write_answer,
        )

    @property
    def llm_available(self) -> bool:
        return self.llm is not None

    # -- public API ------------------------------------------------------
    def ask(self, question: str) -> AnswerRecord:
        started = time.perf_counter()
        latency: dict[str, int] = {}
        attempts: list[dict[str, Any]] = []
        question = " ".join((question or "").split())
        if len(question) > self.settings.app.max_input_chars:
            question = question[: self.settings.app.max_input_chars].rstrip()

        # 1. PII guard: before retrieval, before any provider call.
        guard_started = time.perf_counter()
        decision = (
            scan_pii(question)
            if self.settings.pii.enabled
            else _clean_scan(question)
        )
        latency["guard"] = _ms(guard_started)
        safe_question = decision.safe_text

        if decision.scan.hit:
            run = _Run(
                qid=new_qid(),
                question=safe_question,
                started=started,
                latency=latency,
                attempts=attempts,
                prompt={},
            )
            return self._persist(
                self._pii_record(run, decision.scan.kinds)
            )

        # 2. Retrieval (which also resolves the plan and writes the trace).
        retrieval_started = time.perf_counter()
        try:
            result = self.retriever.retrieve(safe_question)
        except Exception:  # never leak a raw exception to the UI
            latency["retrieval"] = _ms(retrieval_started)
            run = _Run(
                qid=new_qid(),
                question=safe_question,
                started=started,
                latency=latency,
                attempts=attempts,
                prompt={},
                error_code="E_RETRIEVAL",
            )
            return self._persist(self._runtime_error_record(run))
        latency["retrieval"] = _ms(retrieval_started)

        run = _Run(
            qid=result.qid or new_qid(),
            question=safe_question,
            started=started,
            latency=latency,
            attempts=attempts,
            prompt={},
            trace_path=result.trace_path or None,
        )
        plan = result.plan
        intent = plan.intent

        # 3. Fail-closed routes that must not call the LLM.
        if intent == "advice":
            return self._persist(
                self._refusal_record(
                    run,
                    kind="advice",
                    reason="advice",
                    citation=self._citation_for(plan.scheme_slug),
                )
            )
        if intent == "performance":
            return self._persist(
                self._refusal_record(
                    run,
                    kind="performance",
                    reason="performance",
                    citation=self._citation_for(plan.scheme_slug),
                )
            )
        if intent == "out_of_scope":
            return self._persist(
                self._not_found_record(
                    run,
                    reason=_out_of_scope_reason(plan),
                    citation=self.home,
                    plan=plan,
                )
            )

        # fact
        if result.filter_level == "ambiguous_scheme":
            return self._persist(
                self._ambiguous_topic_record(run, result)
            )
        if result.filter_level == "unsupported_field":
            return self._persist(
                self._not_found_record(
                    run,
                    reason="unsupported_field",
                    citation=self._citation_for(plan.scheme_slug),
                    plan=plan,
                )
            )
        if not result.gate_passed or not result.selected:
            return self._persist(
                self._not_found_record(
                    run,
                    reason="no_context",
                    citation=self._citation_for(plan.scheme_slug),
                    plan=plan,
                )
            )

        field = plan.field
        if field is None and plan.field_candidates:
            field = plan.field_candidates[0]

        context = _topic_chunks(result.selected, plan.scheme_slug)
        if plan.ambiguous_field and field:
            narrowed = [
                chunk
                for chunk in context
                if str((chunk.metadata or {}).get("field") or "") == field
            ]
            if narrowed:
                context = narrowed
        if not context:
            return self._persist(
                self._not_found_record(
                    run,
                    reason="no_context",
                    citation=self._citation_for(plan.scheme_slug),
                    plan=plan,
                )
            )

        # 4. Prompt + generation.
        prompt_started = time.perf_counter()
        prompt = build_prompt(
            safe_question,
            context,
            prompt_dir=self.prompt_dir,
        )
        run.latency["prompt"] = _ms(prompt_started)
        run.prompt = {
            "version": prompt.version,
            "system": prompt.system,
            "user": prompt.user,
        }

        if self.llm is None:
            return self._persist(
                self._extractive_record(
                    run,
                    context=context,
                    field=field,
                    reason="no_provider",
                )
            )

        llm_started = time.perf_counter()
        try:
            response = self.llm.complete(
                prompt.messages,
                temperature=self.settings.llm.temperature,
                max_tokens=self.settings.llm.max_tokens,
                timeout_s=self.settings.llm.timeout_seconds,
            )
        except LLMError as exc:
            run.latency["llm"] = _ms(llm_started)
            run.attempts.extend({**item, "phase": "draft"} for item in exc.attempts)
            run.error_code = "E_LLM_TIMEOUT" if "timed out" in str(exc) else "E_LLM_ERROR"
            return self._persist(
                self._extractive_record(
                    run,
                    context=context,
                    field=field,
                    reason="llm_error",
                )
            )
        run.latency["llm"] = _ms(llm_started)
        run.attempts.extend({**item, "phase": "draft"} for item in response.attempts)
        run.raw_answer = response.text

        if _is_not_in_context(response.text):
            return self._persist(
                self._not_found_record(
                    run,
                    reason="model_not_in_context",
                    citation=self.home,
                    plan=plan,
                )
            )

        validate_started = time.perf_counter()
        verdict = self._validate(response.text, context, field)
        run.latency["validate"] = _ms(validate_started)
        if verdict.ok:
            return self._persist(
                self._generated_record(
                    run,
                    text=verdict.text,
                    citation=_citation_from_chunk(context[0]),
                    plan=plan,
                    field=field,
                    context=context,
                    response=response,
                    verdict={"ok": True, "codes": [], "problems": []},
                )
            )

        # 5. One repair pass, then the extractive fallback.
        repair_started = time.perf_counter()
        repair_prompt = build_repair_prompt(
            safe_question,
            context,
            response.text,
            verdict.problems,
            prompt_dir=self.prompt_dir,
        )
        run.prompt["repair_user"] = repair_prompt.user
        try:
            repair = self.llm.complete(
                repair_prompt.messages,
                temperature=self.settings.llm.temperature,
                max_tokens=self.settings.llm.max_tokens,
                timeout_s=self.settings.llm.timeout_seconds,
            )
        except LLMError as exc:
            run.latency["llm"] = run.latency.get("llm", 0) + _ms(repair_started)
            run.attempts.extend({**item, "phase": "repair"} for item in exc.attempts)
            run.error_code = "E_LLM_TIMEOUT" if "timed out" in str(exc) else "E_LLM_ERROR"
            return self._persist(
                self._extractive_record(
                    run,
                    context=context,
                    field=field,
                    reason="repair_failed",
                )
            )
        run.latency["llm"] = run.latency.get("llm", 0) + _ms(repair_started)
        run.attempts.extend({**item, "phase": "repair"} for item in repair.attempts)
        run.repair_answer = repair.text

        if _is_not_in_context(repair.text):
            return self._persist(
                self._not_found_record(
                    run,
                    reason="model_not_in_context",
                    citation=self.home,
                    plan=plan,
                )
            )

        repair_started = time.perf_counter()
        repaired = self._validate(repair.text, context, field)
        run.latency["validate"] = run.latency.get("validate", 0) + _ms(repair_started)
        if repaired.ok:
            return self._persist(
                self._generated_record(
                    run,
                    text=repaired.text,
                    citation=_citation_from_chunk(context[0]),
                    plan=plan,
                    field=field,
                    context=context,
                    response=repair,
                    verdict={
                        "ok": True,
                        "codes": [],
                        "problems": [],
                        "repaired": True,
                        "first_codes": verdict.codes,
                    },
                )
            )

        return self._persist(
            self._extractive_record(
                run,
                context=context,
                field=field,
                reason="validation_failed",
                validation={
                    "ok": False,
                    "codes": repaired.codes,
                    "problems": repaired.problems,
                },
            )
        )

    # -- record builders -------------------------------------------------
    def _pii_record(self, run: _Run, kinds: list[str]) -> AnswerRecord:
        citation = self.home
        body = (
            "Please don't share PAN, Aadhaar, account numbers, OTPs, emails, or "
            "phone numbers — I don't need them and won't store them. I can answer "
            "Groww platform help questions only."
        )
        text = self._with_citation(body, citation)
        return self._record(
            run,
            status="refused",
            answer_text=text,
            citation=citation,
            intent="pii",
            refused=True,
            refusal_reason="pii",
            validation={"ok": False, "codes": ["pii_blocked"], "kinds": kinds},
            question_redacted=run.question,
        )

    def _runtime_error_record(self, run: _Run) -> AnswerRecord:
        citation = self.home
        body = "I couldn't complete that request right now. Please try again."
        return self._record(
            run,
            status="error",
            answer_text=self._with_citation(body, citation),
            citation=citation,
            intent="fact",
            error_code=run.error_code,
            validation={"ok": False, "codes": ["runtime_error"]},
        )

    def _refusal_record(
        self,
        run: _Run,
        *,
        kind: str,
        reason: str,
        citation: Citation,
    ) -> AnswerRecord:
        if kind == "advice":
            body = (
                "I can only share facts from official Groww help pages, so I "
                "can't advise on investing, trading, or which stock to choose. "
                "The official help page is here:"
            )
        else:
            body = (
                "I don't compute or compare returns or performance, and those "
                "figures aren't in the pages I cover. The official help page is here:"
            )
        return self._record(
            run,
            status="refused",
            answer_text=self._with_citation(body, citation),
            citation=citation,
            intent=kind,
            refused=True,
            refusal_reason=reason,
            validation={"ok": False, "codes": [reason]},
        )

    def _not_found_record(
        self,
        run: _Run,
        *,
        reason: str,
        citation: Citation,
        plan: QueryPlan | None = None,
    ) -> AnswerRecord:
        lead = (
            "I couldn't find that in the official Groww help pages in my scope "
            f"({_SCOPE}). The official help page is here:"
        )
        return self._record(
            run,
            status="not_found",
            answer_text=self._with_citation(lead, citation),
            citation=citation,
            intent=(plan.intent if plan is not None else "fact"),
            refusal_reason=reason,
            validation={"ok": False, "codes": [reason]},
        )

    def _ambiguous_topic_record(
        self, run: _Run, result: RetrievalResult
    ) -> AnswerRecord:
        candidates = result.plan.scheme_candidates
        names = [TOPIC_NAMES.get(slug, slug) for slug in candidates]
        body = (
            "Please pick one Groww help topic, because I answer one topic at a "
            f"time: {', '.join(names)}. The Groww Help home page is here:"
        )
        return self._record(
            run,
            status="not_found",
            answer_text=self._with_citation(body, self.home),
            citation=self.home,
            intent="fact",
            refusal_reason="ambiguous_topic",
            validation={"ok": False, "codes": ["ambiguous_topic"]},
        )

    def _generated_record(
        self,
        run: _Run,
        *,
        text: str,
        citation: Citation,
        plan: QueryPlan,
        field: str | None,
        context: Sequence[RetrievedChunk],
        response: Any,
        verdict: dict[str, Any],
    ) -> AnswerRecord:
        final = append_freshness(
            text,
            citation.fetched_at,
            show=self.settings.answer.show_last_updated,
        )
        return self._record(
            run,
            status="answered",
            answer_text=final,
            citation=citation,
            intent="fact",
            scheme=plan.scheme,
            field=field,
            chunk_ids=[chunk.chunk_id for chunk in context],
            validation=verdict,
            provider=response.provider,
            model=response.model,
            prompt_version=PROMPT_VERSION,
        )

    def _extractive_record(
        self,
        run: _Run,
        *,
        context: Sequence[RetrievedChunk],
        field: str | None,
        reason: str,
        validation: dict[str, Any] | None = None,
    ) -> AnswerRecord:
        extractive = build_extractive_answer(context, field=field)
        if extractive is None:
            return self._not_found_record(
                run,
                reason="no_context",
                citation=self._citation_for(None),
            )
        citation = _citation_from_chunk(extractive.chunk)
        final = append_freshness(
            extractive.text,
            citation.fetched_at,
            show=self.settings.answer.show_last_updated,
        )
        verdict = validate_answer(
            extractive.text,
            context,
            allowed_domains=self.settings.corpus.allowed_domains,
            max_sentences=self.settings.answer.max_sentences,
            require_citation=self.settings.answer.require_citation,
            field=extractive.field,
        )
        if not verdict.ok:
            return self._not_found_record(
                run,
                reason="fallback_invalid",
                citation=self._citation_for(None),
            )
        run.fallback_reason = reason
        payload = {
            "ok": True,
            "codes": [],
            "problems": [],
            "fallback": True,
            "fallback_reason": reason,
        }
        if validation:
            payload["first_attempt"] = validation
        return self._record(
            run,
            status="answered",
            answer_text=final,
            citation=citation,
            intent="fact",
            scheme=citation.scheme or None,
            field=extractive.field,
            chunk_ids=[extractive.chunk.chunk_id],
            validation=payload,
            fallback_used=True,
            fallback_reason=reason,
            prompt_version=PROMPT_VERSION,
        )

    # -- helpers ---------------------------------------------------------
    def _record(
        self,
        run: _Run,
        *,
        status: str,
        answer_text: str,
        citation: Citation,
        intent: str,
        scheme: str | None = None,
        field: str | None = None,
        chunk_ids: Sequence[str] | None = None,
        refused: bool = False,
        refusal_reason: str | None = None,
        validation: dict[str, Any] | None = None,
        provider: str | None = None,
        model: str | None = None,
        prompt_version: str = PROMPT_VERSION,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
        error_code: str | None = None,
        question_redacted: str | None = None,
    ) -> AnswerRecord:
        run.latency["total"] = _ms(run.started)
        return AnswerRecord(
            qid=run.qid,
            question=run.question,
            status=status,  # type: ignore[arg-type]
            answer_text=answer_text,
            citation_url=citation.url or None,
            last_updated=citation.fetched_at,
            intent=intent,  # type: ignore[arg-type]
            scheme=scheme,
            field=field,
            chunk_ids=list(chunk_ids or []),
            refused=refused,
            refusal_reason=refusal_reason,
            validation=validation or {},
            provider=provider,
            model=model,
            prompt_version=prompt_version,
            latency_ms=dict(run.latency),
            trace_id=run.qid if run.trace_path else None,
            question_redacted=question_redacted,
            citation_title=citation.title or None,
            trace_path=run.trace_path,
            prompt=dict(run.prompt),
            raw_answer=run.raw_answer,
            repair_answer=run.repair_answer,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            provider_attempts=list(run.attempts),
            error_code=error_code or run.error_code,
        )

    def _with_citation(self, body: str, citation: Citation) -> str:
        if not citation.url:
            text = body.rstrip()
        else:
            text = f"{body.rstrip()} [Source: {citation.title}]({citation.url})"
        return append_freshness(
            text,
            citation.fetched_at,
            show=self.settings.answer.show_last_updated,
        )

    def _citation_for(self, scheme_slug: str | None) -> Citation:
        if scheme_slug and scheme_slug in self.citations:
            return self.citations[scheme_slug]
        return self.home

    def _validate(
        self,
        text: str,
        context: Sequence[RetrievedChunk],
        field: str | None,
    ):
        return validate_answer(
            text,
            context,
            allowed_domains=self.settings.corpus.allowed_domains,
            max_sentences=self.settings.answer.max_sentences,
            require_citation=self.settings.answer.require_citation,
            field=field,
        )

    def _persist(self, record: AnswerRecord) -> AnswerRecord:
        if not self.write_answer:
            return record
        try:
            self.answer_dir.mkdir(parents=True, exist_ok=True)
            path = self.answer_dir / f"{record.qid}.json"
            tmp = path.with_name(path.name + ".tmp")
            payload = asdict(record)
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError:
            # An answer must still render even if the demo disk is read-only.
            pass
        return record


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _clean_scan(question: str) -> PiiDecision:
    return PiiDecision(
        scan=PiiScan(hit=False, kinds=[], redacted=question),
        safe_text=question,
    )


def _build_citations(chunks: Sequence[Chunk]) -> dict[str, Citation]:
    found: dict[str, Citation] = {}
    for chunk in chunks:
        if chunk.scheme_slug in found:
            continue
        found[chunk.scheme_slug] = Citation(
            scheme_slug=chunk.scheme_slug,
            scheme=chunk.scheme,
            title=chunk.source_title,
            url=chunk.source_url,
            fetched_at=chunk.fetched_at,
        )
    return found


def _citation_from_chunk(chunk: RetrievedChunk) -> Citation:
    meta = chunk.metadata or {}
    return Citation(
        scheme_slug=str(meta.get("scheme_slug") or ""),
        scheme=str(meta.get("scheme") or ""),
        title=str(meta.get("source_title") or ""),
        url=str(meta.get("source_url") or ""),
        fetched_at=str(meta.get("fetched_at") or ""),
    )


def _topic_chunks(
    chunks: Sequence[RetrievedChunk], scheme_slug: str | None
) -> list[RetrievedChunk]:
    if not scheme_slug:
        return list(chunks)
    matched = [
        chunk
        for chunk in chunks
        if str((chunk.metadata or {}).get("scheme_slug") or "") == scheme_slug
    ]
    return matched or list(chunks)


def _is_not_in_context(text: str) -> bool:
    return (text or "").strip().strip("`\"'").upper() == "NOT_IN_CONTEXT"


def _out_of_scope_reason(plan: QueryPlan) -> str:
    for note in plan.notes:
        if note.startswith("out_of_scope:"):
            return "out_of_scope"
    return "out_of_scope"


def answer(
    question: str,
    *,
    settings: Settings | None = None,
    engine: AnswerEngine | None = None,
) -> AnswerRecord:
    """Convenience entry point used by scripts/tests; the UI keeps an engine."""
    if engine is not None:
        return engine.ask(question)
    active = settings or load_settings(require_provider_key=False)
    return AnswerEngine.open(active).ask(question)


__all__ = [
    "AnswerEngine",
    "Citation",
    "answer",
]
