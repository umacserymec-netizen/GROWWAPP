"""Versioned prompt assembly for the answer stage.

The system prompt is a file (``prompts/answer_v1.md``) so a regression can be
tied to a prompt version. Context blocks follow the contract in
``ARCHITECTURE.md`` §6.3 and contain only retrieved public text plus the
citation metadata the validator later re-checks.

Retrieved text is data, never instructions. Markdown links and bare URLs inside
chunk text are removed here as a second line of defence (the chunker already
strips them), so the only URL in the prompt is the canonical ``source_url``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from config.settings import PROJECT_ROOT
from core.llm.base import Message
from core.types import RetrievedChunk

PROMPT_VERSION = "answer_v1"
PROMPT_FILENAME = "answer_v1.md"
DEFAULT_PROMPT_DIR = PROJECT_ROOT / "prompts"

_MD_LINK = re.compile(r"!?\[([^\]]*)\]\(([^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\)")
_BARE_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


@dataclass(slots=True)
class PromptBundle:
    version: str
    system: str
    user: str
    messages: list[Message]
    context: str

    @property
    def char_len(self) -> int:
        return len(self.system) + len(self.user)


def load_system_prompt(prompt_dir: str | Path | None = None) -> str:
    directory = Path(prompt_dir) if prompt_dir is not None else DEFAULT_PROMPT_DIR
    path = directory / PROMPT_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"prompt template not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"prompt template is empty: {path}")
    return text


def sanitise_chunk_text(text: str) -> str:
    """Keep visible words; drop injected links/tags and collapse whitespace."""
    clean = _MD_LINK.sub(r"\1", text or "")
    clean = _BARE_URL.sub("", clean)
    clean = _TAG.sub(" ", clean)
    return " ".join(clean.split())


def render_context(chunks: Sequence[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        field = str(meta.get("field") or "n/a")
        topic = str(meta.get("scheme") or "")
        title = str(meta.get("source_title") or "")
        url = str(meta.get("source_url") or "")
        fetched_at = str(meta.get("fetched_at") or "")
        blocks.append(
            "\n".join(
                (
                    f"[{index}] chunk_id={chunk.chunk_id} field={field} topic={topic}",
                    f"text: {sanitise_chunk_text(chunk.text)}",
                    f"source_title: {title}",
                    f"source_url: {url}",
                    f"fetched_at: {fetched_at}",
                )
            )
        )
    return "\n\n".join(blocks)


def build_prompt(
    question: str,
    chunks: Sequence[RetrievedChunk],
    *,
    prompt_dir: str | Path | None = None,
) -> PromptBundle:
    system = load_system_prompt(prompt_dir)
    context = render_context(chunks)
    user = (
        "CONTEXT:\n"
        "<context>\n"
        f"{context if context else '(empty)'}\n"
        "</context>\n\n"
        "QUESTION:\n"
        f"{question.strip()}\n"
    )
    return PromptBundle(
        version=PROMPT_VERSION,
        system=system,
        user=user,
        messages=[Message(role="system", content=system), Message(role="user", content=user)],
        context=context,
    )


def build_repair_prompt(
    question: str,
    chunks: Sequence[RetrievedChunk],
    draft: str,
    problems: Sequence[str],
    *,
    prompt_dir: str | Path | None = None,
) -> PromptBundle:
    """One repair attempt: same context, explicit deterministic violations."""
    base = build_prompt(question, chunks, prompt_dir=prompt_dir)
    issues = "\n".join(f"- {problem}" for problem in problems)
    user = (
        f"{base.user}\n"
        "Your previous draft was rejected by the validator for:\n"
        f"{issues}\n\n"
        "Rejected draft:\n"
        f"{draft.strip()}\n\n"
        "Return a corrected answer that follows every rule. If the answer is "
        "not in CONTEXT, reply exactly NOT_IN_CONTEXT."
    )
    return PromptBundle(
        version=base.version,
        system=base.system,
        user=user,
        messages=[
            Message(role="system", content=base.system),
            Message(role="user", content=user),
        ],
        context=base.context,
    )


__all__ = [
    "DEFAULT_PROMPT_DIR",
    "PROMPT_FILENAME",
    "PROMPT_VERSION",
    "PromptBundle",
    "build_prompt",
    "build_repair_prompt",
    "load_system_prompt",
    "render_context",
    "sanitise_chunk_text",
]
