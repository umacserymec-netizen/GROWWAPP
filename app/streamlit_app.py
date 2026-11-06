"""Streamlit chat UI for the facts-only Groww help assistant.

Run from the repository root::

    streamlit run app/streamlit_app.py

The page owns conversation state only. Every answer comes from
``core.answer.AnswerEngine``; the inspector reads the persisted ``data/``
artifacts and never recomputes retrieval.
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components.chat import (  # noqa: E402
    render_assistant_message,
    render_user_message,
)
from app.components.theme import (  # noqa: E402
    inject_css,
    load_sources,
    render_footer,
    render_hero,
    render_navbar,
    render_sidebar_disclaimer,
    render_sources_panel,
)
from config.settings import Settings, load_settings  # noqa: E402
from core.answer import AnswerEngine  # noqa: E402
from core.store import INDEX_META_NAME, load_index_meta  # noqa: E402
from core.types import AnswerRecord  # noqa: E402

EXAMPLES = (
    "Do you charge for KYC verification?",
    "How do I change my Groww password?",
    "When will my stocks reach my demat account?",
)


def _bridge_streamlit_secrets() -> None:
    """Expose Streamlit Cloud secrets to ``config.load_settings``.

    Community Cloud normally maps secrets to environment variables as well,
    but this keeps local ``.env`` and cloud secrets on the same code path.
    """
    try:
        secrets = {str(key): str(value) for key, value in st.secrets.items()}
    except Exception:
        return
    for key in (
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MISTRAL_API_KEY",
        "LLM_PROVIDER",
        "LLM_MODEL",
    ):
        if key not in os.environ and key in secrets:
            os.environ[key] = secrets[key]


@st.cache_resource(show_spinner="Loading the verified index…")
def get_engine() -> AnswerEngine:
    _bridge_streamlit_secrets()
    settings = load_settings(require_provider_key=False)
    return AnswerEngine.open(settings)


def _source_summary(settings: Settings) -> tuple[str, int, bool]:
    """Return newest fetched date, chunk count, stale flag from index metadata."""
    try:
        meta = load_index_meta(
            Path(settings.vector_store.persist_dir) / INDEX_META_NAME
        )
    except Exception:
        return "", 0, False
    newest = meta.newest_fetched_at
    stale = False
    try:
        fetched = datetime.fromisoformat(newest.replace("Z", "+00:00")).date()
        stale = (date.today() - fetched).days > settings.debug.index_stale_days
    except ValueError:
        pass
    return newest, meta.chunk_count, stale


def _submit(question: str, max_chars: int) -> None:
    question = question.strip()[:max_chars]
    if not question:
        return
    st.session_state.groww_help_messages.append(
        {"role": "user", "text": question}
    )
    st.session_state.groww_help_pending = question
    st.rerun()


def _error_record(question: str) -> AnswerRecord:
    return AnswerRecord(
        qid=f"err_{uuid.uuid4().hex[:6]}",
        question=question,
        status="error",
        answer_text=(
            "Something went wrong while reading the official Groww help pages. "
            "Please try that question again."
        ),
        citation_url=None,
        last_updated="",
        intent="fact",
    )


def _render_messages(engine: AnswerEngine) -> None:
    """Render the transcript, never letting one bad record hide the input."""
    for item in st.session_state.groww_help_messages:
        try:
            if item.get("role") == "user":
                render_user_message(str(item.get("text", "")))
            elif item.get("record") is not None:
                render_assistant_message(item["record"], engine.settings)
        except Exception:
            continue


def main() -> None:
    st.set_page_config(
        page_title="Ask a Groww platform help question",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    inject_css()
    render_navbar()
    render_sidebar_disclaimer()

    try:
        engine = get_engine()
    except Exception:
        st.error(
            "This demo is starting up. Please refresh the page in a moment."
        )
        render_footer()
        st.stop()
        return

    if "groww_help_messages" not in st.session_state:
        st.session_state.groww_help_messages = []
    if "groww_help_pending" not in st.session_state:
        st.session_state.groww_help_pending = None

    max_chars = engine.settings.app.max_input_chars
    with st.container(key="mf-layout"):
        main_col, sources_col = st.columns([800, 320], gap="large")

    with main_col:
        render_hero()
        with st.container(key="mf-chips"):
            columns = st.columns(3, gap="small")
            for index, (column, example) in enumerate(
                zip(columns, EXAMPLES, strict=True)
            ):
                with column:
                    if st.button(
                        example,
                        key=f"mf-example-{index}",
                        use_container_width=True,
                    ):
                        _submit(example, max_chars)
        _render_messages(engine)
        typing_slot = st.empty()

    with sources_col:
        _, _, stale = _source_summary(engine.settings)
        render_sources_panel(
            sources=load_sources(engine.settings.paths.sources_csv),
        )
        if stale:
            st.warning(
                "These official pages may have changed since this demo was "
                "last updated. Please check the source links for the latest "
                "information."
            )

    typed = st.chat_input(
        "Ask a factual question about Groww...",
        max_chars=max_chars,
    )
    if typed:
        _submit(typed, max_chars)

    pending = st.session_state.groww_help_pending
    if pending:
        st.session_state.groww_help_pending = None
        with typing_slot:
            with st.spinner("Reading the official pages…"):
                try:
                    record = engine.ask(str(pending))
                except Exception:
                    record = _error_record(str(pending))
        st.session_state.groww_help_messages.append(
            {"role": "assistant", "record": record}
        )
        st.rerun()

    render_footer()


if __name__ == "__main__":
    main()
