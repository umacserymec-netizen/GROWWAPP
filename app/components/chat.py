"""Chat bubbles and source cards for the Streamlit UI.

Answers are rendered from the validated ``AnswerRecord`` only. Technical stage
artifacts stay on disk for development; they are not shown to the visitor.
"""

from __future__ import annotations

import html
import re
from urllib.parse import urlparse

import streamlit as st

from app.components.theme import LINK_ICON, compact_html
from config.settings import Settings
from core.types import AnswerRecord
from core.validate import format_source_date, strip_freshness

_CITATION = re.compile(r"\s*\[Source:[^\]]*\]\([^)]+\)")


def answer_body(record: AnswerRecord) -> str:
    """Strip the citation/freshness tail so the card can render them once."""
    body = _CITATION.sub("", strip_freshness(record.answer_text)).strip()
    return body or record.answer_text


def _text(value: str) -> str:
    return html.escape(value).replace("\n", "<br />")


def _url_label(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc:
        return f"{parsed.netloc}{parsed.path.rstrip('/')}"
    return url


def _source_block(record: AnswerRecord) -> str:
    if not record.citation_url:
        return ""
    url = html.escape(record.citation_url, quote=True)
    label = html.escape(_url_label(record.citation_url))
    title = html.escape(record.citation_title or "Official page", quote=True)
    updated = ""
    if record.last_updated:
        updated = (
            '<div class="mf-updated">Last updated from sources: '
            f"{html.escape(format_source_date(record.last_updated))}</div>"
        )
    return f"""
      <div class="mf-source-row">
        <span class="mf-source-row-icon">{LINK_ICON}</span>
        <div class="mf-source-row-body">
          <span class="mf-source-label">Source</span>
          <a class="mf-source-link" href="{url}" target="_blank"
             rel="noopener noreferrer"
             aria-label="Open official source: {title}">{label}</a>
        </div>
      </div>
      {updated}
    """


def render_user_message(text: str) -> None:
    st.markdown(
        '<div class="mf-msg mf-msg-user">'
        f'<div class="mf-bubble-user">{_text(text)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _note_copy(record: AnswerRecord) -> tuple[str, str]:
    if record.status == "refused" and record.intent == "advice":
        return (
            "I can only share factual information, not investment advice",
            "Try asking about KYC charges, changing your password, account "
            "opening charges, stock delivery, or available balance.",
        )
    if record.status == "refused":
        return (
            "I can only share factual information from the official Groww help pages",
            answer_body(record),
        )
    if record.status == "not_found":
        return (
            "I couldn't find that in the official Groww help pages",
            answer_body(record),
        )
    return (
        "Something went wrong",
        answer_body(record) or "Please try that question again.",
    )


def render_assistant_message(record: AnswerRecord, settings: Settings) -> None:
    if record.status == "answered":
        key = f"mf-answer-{record.qid}"
        with st.container(key=key):
            st.markdown(
                compact_html(
                    f"""
                    <div class="mf-answer-top">
                      <span class="mf-badge">
                        <span class="mf-badge-dot" aria-hidden="true"></span>
                        Official source
                      </span>
                    </div>
                    <p class="mf-answer-text">{_text(answer_body(record))}</p>
                    {_source_block(record)}
                    """
                ),
                unsafe_allow_html=True,
            )
        return

    title, body = _note_copy(record)
    with st.container(key=f"mf-note-{record.qid}"):
        st.markdown(
            compact_html(
                f"""
                <p class="mf-note-title">{_text(title)}</p>
                <p class="mf-note-text">{_text(body)}</p>
                {_source_block(record)}
                """
            ),
            unsafe_allow_html=True,
        )


__all__ = ["answer_body", "render_assistant_message", "render_user_message"]
