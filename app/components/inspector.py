"""Inspector pane: reads persisted stage artifacts, never recomputes them.

The UI never performs retrieval or formatting here. It opens
``data/06_retrieval/<ts>_<qid>.json``, ``data/07_answers/<qid>.json``, and
``data/03_chunks/chunks.jsonl`` exactly as the pipeline wrote them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import streamlit as st

from config.settings import Settings
from core.types import AnswerRecord
from core.validate import format_source_date, strip_freshness

_CITATION = re.compile(r"\s*\[Source:[^\]]*\]\([^)]+\)")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def load_answer(answer_dir: Path, qid: str) -> dict[str, Any] | None:
    return read_json(Path(answer_dir) / f"{qid}.json")


def find_trace(trace_dir: Path, qid: str) -> dict[str, Any] | None:
    directory = Path(trace_dir)
    if not directory.exists():
        return None
    matches = sorted(directory.glob(f"*_{qid}.json"))
    if not matches:
        return None
    return read_json(matches[-1])


def load_chunks(chunks_path: Path, chunk_ids: list[str]) -> list[dict[str, Any]]:
    wanted = set(chunk_ids)
    if not wanted:
        return []
    found: list[dict[str, Any]] = []
    try:
        with Path(chunks_path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item.get("chunk_id") in wanted:
                    found.append(item)
    except (OSError, json.JSONDecodeError):
        return []
    return found


def answer_body(record: AnswerRecord) -> str:
    """Strip the citation/freshness tail so the card can render them once."""
    body = _CITATION.sub("", strip_freshness(record.answer_text)).strip()
    return body or record.answer_text


def render_inspector(record: AnswerRecord, settings: Settings) -> None:
    """Collapsed-by-default five-tab trace of the request."""
    if not settings.debug.inspector:
        return

    answer_dir = Path(settings.logging.trace_dir)
    trace_dir = Path(settings.paths.data_dir) / "06_retrieval"
    chunks_path = Path(settings.paths.data_dir) / "03_chunks" / "chunks.jsonl"
    persisted = load_answer(answer_dir, record.qid) or {}
    trace = find_trace(trace_dir, record.qid)
    selected = list((trace or {}).get("selected") or [])
    chunk_ids = [
        str(item.get("chunk_id"))
        for item in selected
        if item.get("chunk_id")
    ] or list(record.chunk_ids)
    chunks = load_chunks(chunks_path, chunk_ids)

    with st.expander("🔍 Stage inspector", expanded=False):
        st.caption(
            f"qid `{record.qid}` · status `{record.status}` · "
            f"intent `{record.intent}` · "
            f"trace `{record.trace_path or 'not written'}`"
        )
        tabs = st.tabs(
            [
                "Sources",
                "Chunks",
                "Retrieval scores",
                "Prompt",
                "Raw vs validated answer",
            ]
        )

        with tabs[0]:
            if selected:
                for item in selected:
                    meta = item.get("metadata") or {}
                    title = meta.get("source_title") or "Official page"
                    url = meta.get("source_url") or ""
                    st.markdown(f"**[{title}]({url})**" if url else f"**{title}**")
                    st.caption(
                        f"{meta.get('scheme', '')} · field `{meta.get('field') or 'n/a'}` · "
                        f"fetched {meta.get('fetched_at', '')} · "
                        f"chunk score `{item.get('score', '')}`"
                    )
            elif record.citation_url:
                st.markdown(
                    f"**[{record.citation_title or 'Official page'}]"
                    f"({record.citation_url})**"
                )
                st.caption("No retrieval chunks on this path (guard or refusal).")
            else:
                st.caption("No source on this record.")

        with tabs[1]:
            if not chunks:
                st.caption("No chunks selected for this request.")
            for item in chunks:
                meta = item.get("metadata") or {}
                scheme = meta.get("scheme") or item.get("scheme", "")
                st.markdown(
                    f"`{item.get('chunk_id', '')}` · {scheme} · "
                    f"`{item.get('field') or 'n/a'}` · {item.get('chunk_type', '')}"
                )
                st.code(str(item.get("text") or ""), language=None)

        with tabs[2]:
            if not trace:
                st.caption("No retrieval trace for this path.")
            else:
                st.markdown(
                    f"**Filter level:** `{trace.get('filter_level', '')}`  \n"
                    f"**Gate:** `{trace.get('gate', {})}`"
                )
                st.write("Filter levels")
                for level in trace.get("levels") or []:
                    st.write(
                        f"- `{level.get('level')}` · top `{level.get('top_score')}` "
                        f"· {level.get('candidate_count')} candidates · "
                        f"where `{level.get('where')}`"
                    )
                st.write("MMR picks")
                for item in trace.get("mmr") or []:
                    st.write(f"- `{item.get('chunk_id')}` · score `{item.get('score')}`")
                st.write("Top dense candidates")
                for item in (trace.get("dense") or [])[:10]:
                    st.write(
                        f"- `{item.get('score')}` · `{item.get('chunk_id')}` · "
                        f"{(item.get('text') or '')[:140]}"
                    )
                if trace.get("lexical_used"):
                    st.write("BM25 fallback")
                    for item in (trace.get("lexical") or [])[:10]:
                        st.write(
                            f"- `{item.get('score')}` · `{item.get('chunk_id')}` · "
                            f"{(item.get('text') or '')[:140]}"
                        )

        with tabs[3]:
            prompt = (persisted.get("prompt") if persisted else None) or record.prompt
            if not prompt:
                st.caption("No prompt was built on this path.")
            else:
                st.caption(f"prompt version `{prompt.get('version', '')}`")
                st.text_area(
                    "System prompt",
                    value=str(prompt.get("system", "")),
                    height=240,
                    disabled=True,
                    key=f"inspector-{record.qid}-system-prompt",
                )
                st.text_area(
                    "User prompt (context + question)",
                    value=str(prompt.get("user", "")),
                    height=260,
                    disabled=True,
                    key=f"inspector-{record.qid}-user-prompt",
                )
                if prompt.get("repair_user"):
                    st.text_area(
                        "Repair prompt",
                        value=str(prompt.get("repair_user", "")),
                        height=180,
                        disabled=True,
                        key=f"inspector-{record.qid}-repair-prompt",
                    )

        with tabs[4]:
            raw = (persisted.get("raw_answer") if persisted else "") or record.raw_answer
            repair = (persisted.get("repair_answer") if persisted else "") or record.repair_answer
            validation = (
                persisted.get("validation") if persisted else None
            ) or record.validation
            st.caption(
                f"provider `{record.provider or '-'}` · model `{record.model or '-'}` · "
                f"fallback `{record.fallback_used}` · "
                f"latency `{record.latency_ms}`"
            )
            st.text_area(
                "Raw model output",
                value=str(raw or ""),
                height=120,
                disabled=True,
                key=f"inspector-{record.qid}-raw-answer",
            )
            if repair:
                st.text_area(
                    "Repair output",
                    value=str(repair),
                    height=120,
                    disabled=True,
                    key=f"inspector-{record.qid}-repair-answer",
                )
            st.text_area(
                "Final validated answer",
                value=record.answer_text,
                height=140,
                disabled=True,
                key=f"inspector-{record.qid}-final-answer",
            )
            st.json(validation or {})
            if record.last_updated:
                st.caption(
                    "Freshness from source metadata: "
                    f"{format_source_date(record.last_updated)}"
                )


__all__ = [
    "answer_body",
    "find_trace",
    "load_answer",
    "load_chunks",
    "read_json",
    "render_inspector",
]
