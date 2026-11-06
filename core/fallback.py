"""Extractive fallback used when the model is unavailable or invalid.

The fallback never paraphrases. For help topics it copies up to two sentences
from the best retrieved article passage, verbatim, and cites that passage's own
page. If no usable passage is available the caller uses the not-found template.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from core.fields import field_labels
from core.types import RetrievedChunk

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ORDERED_STEP = re.compile(r"^\d+\.\s+")


@dataclass(slots=True)
class ExtractiveAnswer:
    text: str
    chunk: RetrievedChunk
    field: str
    value: str


def select_fact_chunk(
    chunks: Sequence[RetrievedChunk], field: str | None
) -> RetrievedChunk | None:
    """Prefer a fact atom for the asked field, then any atom, then top chunk."""
    if not chunks:
        return None
    for chunk in chunks:
        meta = chunk.metadata or {}
        if field and meta.get("field") == field and meta.get("chunk_type") == "fact_atom":
            return chunk
    for chunk in chunks:
        if (chunk.metadata or {}).get("chunk_type") == "fact_atom":
            return chunk
    return chunks[0]


def _one_sentence(value: str) -> str:
    clean = " ".join(value.split()).strip()
    if not clean:
        return clean
    # Keep a verbatim first sentence, not a long window.
    for separator in (". ", "! ", "? "):
        if separator in clean:
            clean = clean.split(separator, 1)[0]
            break
    clean = clean.rstrip(" .;:")
    return clean


def _value_from_text(text: str, field: str | None) -> str:
    clean = " ".join((text or "").split()).strip()
    if ":" in clean:
        label, value = clean.split(":", 1)
        expected = field_labels().get(field or "", "")
        # Use the right-hand side when the left side is recognisably a label.
        if expected and label.strip().lower() == expected.lower():
            clean = value
        elif len(label) <= 48 and not label.strip().lower().startswith(("http", "note")):
            clean = value
    return _one_sentence(clean)


def _help_units(text: str, section_title: str) -> list[tuple[str, str]]:
    """Split a help passage into ``(kind, text)`` units, keeping steps intact."""
    body = (text or "").strip()
    prefix = f"{section_title} - "
    if section_title and body.startswith(prefix):
        body = body[len(prefix) :]

    units: list[tuple[str, str]] = []
    for raw_line in body.splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        if _ORDERED_STEP.match(line):
            units.append(("step", _ORDERED_STEP.sub("", line, count=1)))
            continue
        units.extend(
            ("text", part.strip())
            for part in _SENTENCE_SPLIT.split(line)
            if part.strip()
        )
    return units


def _help_body(text: str, section_title: str) -> str:
    """Copy the intro plus every ordered step as one semicolon-joined sentence."""
    units = _help_units(text, section_title)
    steps = [body for kind, body in units if kind == "step"]
    if steps:
        lead = [body for kind, body in units if kind == "text"][:1]
        joined = "; ".join(step.rstrip(" .;:") for step in steps)
        if lead:
            return f"{lead[0].rstrip(' :')}: {joined}."
        return f"{joined}."
    prose = [body for _, body in units]
    return " ".join(prose[:2]).strip()


def build_extractive_answer(
    chunks: Sequence[RetrievedChunk],
    *,
    field: str | None,
) -> ExtractiveAnswer | None:
    chunk = select_fact_chunk(chunks, field)
    if chunk is None:
        return None
    meta = chunk.metadata or {}
    url = str(meta.get("source_url") or "")
    title = str(meta.get("source_title") or "")
    if not url or not title:
        return None

    citation = f"[Source: {title}]({url})"

    if str(meta.get("page_type") or "") == "help":
        value = _help_body(chunk.text, str(meta.get("section_title") or ""))
        if not value:
            return None
        return ExtractiveAnswer(
            text=f"{value} {citation}",
            chunk=chunk,
            field="",
            value=value,
        )

    scheme = str(meta.get("scheme") or "")
    active_field = field or str(meta.get("field") or "")
    label = field_labels().get(active_field, "Fact")
    value = _value_from_text(chunk.text, active_field)
    if not value:
        return None

    if scheme:
        body = f"{label} for {scheme}: {value}."
    else:
        body = f"{label}: {value}."

    return ExtractiveAnswer(
        text=f"{body} {citation}",
        chunk=chunk,
        field=active_field,
        value=value,
    )


__all__ = [
    "ExtractiveAnswer",
    "build_extractive_answer",
    "select_fact_chunk",
]
