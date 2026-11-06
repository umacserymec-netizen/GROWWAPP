"""Visual shell for the Streamlit app.

Palette, navbar, hero, source rail, and footer live here so the page module
stays focused on chat state. This module renders only HTML/CSS; it never
performs retrieval or formatting of answers.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

import streamlit as st

DISCLAIMER = "Facts-only. No investment advice."

GREEN = "#00D09C"
VIOLET = "#5367FF"
INK = "#44475B"
MUTED = "#7C7E8C"
BORDER = "#E8E8E8"
TINT = "#E5F9F4"

LINK_ICON = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true">'
    '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>'
    '<path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>'
    "</svg>"
)

CSS = """
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");

:root {
  --mf-green: #00D09C;
  --mf-violet: #5367FF;
  --mf-ink: #44475B;
  --mf-muted: #7C7E8C;
  --mf-bg: #FFFFFF;
  --mf-card: #F8F8F8;
  --mf-border: #E8E8E8;
  --mf-tint: #E5F9F4;
  --mf-shadow: 0 1px 2px rgba(68, 71, 91, 0.04), 0 8px 24px rgba(68, 71, 91, 0.04);
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--mf-bg);
  color: var(--mf-ink);
}

html, body, .stApp, button, input, textarea, [data-testid="stMarkdownContainer"] {
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"],
#MainMenu {
  display: none !important;
}

[data-testid="stMain"] { background: var(--mf-bg); }
[data-testid="stMainBlockContainer"] {
  max-width: 1160px;
  padding: 0 16px 190px !important;
}

/* ---------- navbar ---------- */
.mf-nav {
  display: flex;
  align-items: center;
  gap: 28px;
  min-height: 64px;
  border-bottom: 1px solid var(--mf-border);
  margin-bottom: 8px;
}
.mf-brand {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  color: var(--mf-ink);
  font-size: 18px;
  font-weight: 700;
  letter-spacing: -0.01em;
  text-decoration: none;
  white-space: nowrap;
}
.mf-brand:hover { color: var(--mf-ink); }
.mf-logo {
  display: grid;
  place-items: center;
  width: 30px;
  height: 30px;
  border-radius: 9px;
  background: var(--mf-green);
  color: var(--mf-ink);
  font-size: 15px;
  font-weight: 700;
  line-height: 1;
}
.mf-links {
  display: flex;
  align-items: center;
  gap: 26px;
}
.mf-links a {
  color: var(--mf-muted);
  border-bottom: 2px solid transparent;
  padding: 7px 1px;
  font-size: 14px;
  font-weight: 500;
  text-decoration: none;
  transition: color 120ms ease, border-color 120ms ease;
}
.mf-links a:hover,
.mf-links a:focus-visible {
  color: var(--mf-ink);
  border-bottom-color: var(--mf-green);
}
.mf-sources-btn {
  margin-left: auto;
  border: 1px solid var(--mf-green);
  border-radius: 8px;
  padding: 8px 18px;
  color: var(--mf-ink);
  font-size: 14px;
  font-weight: 600;
  text-decoration: none;
  transition: background-color 120ms ease;
}
.mf-sources-btn:hover,
.mf-sources-btn:focus-visible {
  background: var(--mf-tint);
  color: var(--mf-ink);
}

/* ---------- hero + example chips ---------- */
.mf-hero { padding: 30px 0 18px; }
.mf-hero h1 {
  margin: 0 0 8px;
  color: var(--mf-ink);
  font-size: 32px;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.2;
}
.mf-hero p {
  margin: 0;
  color: var(--mf-muted);
  font-size: 16px;
  line-height: 1.55;
}

.st-key-mf-chips { margin: 2px 0 24px; }
.st-key-mf-chips [data-testid="stHorizontalBlock"] {
  align-items: stretch;
  gap: 12px;
}
.st-key-mf-chips [data-testid="stColumn"] { display: flex; }
.st-key-mf-chips [data-testid="stButton"] { width: 100%; }
.st-key-mf-chips button {
  width: 100%;
  min-height: 66px;
  align-items: flex-start;
  justify-content: flex-start;
  border: 1px solid rgba(0, 208, 156, 0.35);
  border-radius: 12px;
  background: var(--mf-tint);
  box-shadow: none;
  color: var(--mf-ink);
  padding: 12px 15px;
  font-size: 14px;
  font-weight: 500;
  line-height: 1.45;
  text-align: left;
  white-space: normal;
  transition: border-color 120ms ease, box-shadow 120ms ease;
}
.st-key-mf-chips button p {
  font-size: 14px;
  line-height: 1.45;
  text-align: left;
}
.st-key-mf-chips button:hover {
  border-color: var(--mf-green);
  background: var(--mf-tint);
  color: var(--mf-ink);
  box-shadow: 0 2px 8px rgba(68, 71, 91, 0.06);
}
.st-key-mf-chips button:focus-visible {
  outline: 2px solid var(--mf-violet);
  outline-offset: 2px;
}

/* ---------- layout: chat + sources ---------- */
.st-key-mf-layout [data-testid="stHorizontalBlock"] {
  align-items: flex-start;
  gap: 38px;
}
.st-key-mf-layout [data-testid="stColumn"]:first-child { max-width: 800px; }

.mf-sources {
  position: sticky;
  top: 18px;
  background: var(--mf-card);
  border: 1px solid var(--mf-border);
  border-radius: 12px;
  padding: 20px;
}
.mf-panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 6px;
}
.mf-panel-head h2 {
  margin: 0;
  color: var(--mf-ink);
  font-size: 15px;
  font-weight: 600;
}
.mf-chip {
  border-radius: 999px;
  background: var(--mf-tint);
  color: var(--mf-ink);
  padding: 3px 9px;
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
}
.mf-panel-sub {
  margin: 0 0 14px;
  color: var(--mf-muted);
  font-size: 12px;
  line-height: 1.5;
}
.mf-source-list { display: flex; flex-direction: column; gap: 8px; }
.mf-source-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  border: 1px solid var(--mf-border);
  border-radius: 10px;
  background: var(--mf-bg);
  padding: 11px 12px;
  color: var(--mf-ink);
  text-decoration: none;
  transition: border-color 120ms ease, box-shadow 120ms ease;
}
.mf-source-card:hover,
.mf-source-card:focus-visible {
  border-color: var(--mf-green);
  box-shadow: 0 2px 8px rgba(68, 71, 91, 0.06);
  color: var(--mf-ink);
}
.mf-source-title {
  display: block;
  font-size: 13px;
  font-weight: 600;
  line-height: 1.35;
}
.mf-source-meta {
  display: block;
  margin-top: 2px;
  color: var(--mf-muted);
  font-size: 11px;
}
.mf-link-icon {
  display: inline-flex;
  flex-shrink: 0;
  color: var(--mf-violet);
}

.mf-about {
  margin-top: 20px;
  border-top: 1px solid var(--mf-border);
  padding-top: 16px;
}
.mf-about h3 {
  margin: 0 0 6px;
  color: var(--mf-ink);
  font-size: 13px;
  font-weight: 600;
}
.mf-about p {
  margin: 0 0 8px;
  color: var(--mf-muted);
  font-size: 11.5px;
  line-height: 1.55;
}
.mf-about .mf-about-meta { font-size: 11px; }
.mf-about .mf-about-disclaimer {
  margin: 10px 0 0;
  color: var(--mf-ink);
  font-size: 11.5px;
  font-weight: 600;
}

/* ---------- chat messages ---------- */
.mf-msg { margin: 0 0 18px; }
.mf-msg-user { display: flex; justify-content: flex-end; }
.mf-bubble-user {
  max-width: 78%;
  border-radius: 16px 16px 4px 16px;
  background: var(--mf-green);
  color: var(--mf-ink);
  padding: 12px 16px;
  box-shadow: 0 1px 2px rgba(68, 71, 91, 0.06);
  font-size: 15px;
  line-height: 1.5;
  overflow-wrap: anywhere;
}
.mf-msg-assistant { margin-bottom: 20px; }

[class*="st-key-mf-answer-"],
[class*="st-key-mf-note-"] {
  border: 1px solid var(--mf-border);
  border-radius: 12px;
  box-shadow: var(--mf-shadow);
  padding: 18px 20px 14px;
  margin-bottom: 20px;
}
[class*="st-key-mf-answer-"] { background: var(--mf-bg); }
[class*="st-key-mf-note-"] { background: var(--mf-card); }

.mf-answer-top {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}
.mf-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 999px;
  background: var(--mf-tint);
  color: var(--mf-ink);
  padding: 3px 9px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.01em;
}
.mf-badge-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--mf-green);
}
.mf-answer-text {
  margin: 0 0 16px;
  color: var(--mf-ink);
  font-size: 15.5px;
  line-height: 1.65;
}
.mf-note-title {
  margin: 0 0 6px;
  color: var(--mf-ink);
  font-size: 15.5px;
  font-weight: 600;
  line-height: 1.5;
}
.mf-note-text {
  margin: 0 0 16px;
  color: var(--mf-muted);
  font-size: 14px;
  line-height: 1.6;
}
.mf-source-row {
  display: flex;
  align-items: flex-start;
  gap: 9px;
  border-top: 1px solid var(--mf-border);
  padding-top: 12px;
}
.mf-source-row-icon { display: inline-flex; flex-shrink: 0; margin-top: 1px; color: var(--mf-violet); }
.mf-source-row-body { min-width: 0; }
.mf-source-label {
  display: block;
  margin-bottom: 2px;
  color: var(--mf-muted);
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.mf-source-link {
  color: var(--mf-ink);
  font-size: 12.5px;
  line-height: 1.45;
  overflow-wrap: anywhere;
  text-decoration: underline;
  text-decoration-color: var(--mf-green);
  text-underline-offset: 3px;
}
.mf-source-link:hover,
.mf-source-link:focus-visible { color: var(--mf-violet); }
.mf-updated {
  margin-top: 9px;
  color: var(--mf-muted);
  font-size: 11.5px;
}

[class*="st-key-mf-answer-"] [data-testid="stExpander"],
[class*="st-key-mf-note-"] [data-testid="stExpander"] { margin-top: 10px; }
[class*="st-key-mf-answer-"] [data-testid="stExpander"] details,
[class*="st-key-mf-note-"] [data-testid="stExpander"] details {
  border: 1px solid var(--mf-border);
  border-radius: 8px;
  background: var(--mf-bg);
}
[class*="st-key-mf-answer-"] [data-testid="stExpander"] summary,
[class*="st-key-mf-note-"] [data-testid="stExpander"] summary { font-size: 12px; }

/* ---------- pinned composer + footer ---------- */
[data-testid="stBottom"] { bottom: 40px !important; }
[data-testid="stBottomBlockContainer"] {
  max-width: 1160px;
  padding: 10px 344px 12px 16px !important;
  background: var(--mf-bg);
}
[data-testid="stChatInput"] {
  max-width: 800px;
  border: 1px solid var(--mf-border);
  border-radius: 24px;
  background: var(--mf-bg);
  box-shadow: 0 6px 20px rgba(68, 71, 91, 0.07);
  padding: 3px 5px 3px 7px;
}
[data-testid="stChatInput"] textarea {
  color: var(--mf-ink);
  font-size: 15px;
}
[data-testid="stChatInput"] textarea::placeholder { color: #9a9cab; }
[data-testid="stChatInput"] button {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: var(--mf-green);
  color: var(--mf-ink);
}
[data-testid="stChatInput"] button:hover { background: var(--mf-tint); color: var(--mf-ink); }
[data-testid="stChatInput"] button:focus-visible {
  outline: 2px solid var(--mf-violet);
  outline-offset: 2px;
}

.mf-footer {
  position: fixed;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1000000;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 40px;
  border-top: 1px solid var(--mf-border);
  background: var(--mf-card);
  color: var(--mf-ink);
  font-size: 12px;
  font-weight: 500;
}

/* ---------- responsive ---------- */
@media (max-width: 1024px) {
  [data-testid="stBottomBlockContainer"] { padding-right: 16px !important; }
  .st-key-mf-layout [data-testid="stHorizontalBlock"] { flex-direction: column; gap: 20px; }
  .st-key-mf-layout [data-testid="stColumn"] {
    width: 100% !important;
    max-width: 100% !important;
    flex: 1 1 auto !important;
  }
  .mf-sources { position: static; }
  .mf-hero h1 { font-size: 28px; }
}

@media (max-width: 720px) {
  [data-testid="stMainBlockContainer"] { padding-left: 12px !important; padding-right: 12px !important; }
  .mf-nav { gap: 12px; }
  .mf-links { display: none; }
  .mf-hero { padding-top: 22px; }
  .mf-hero h1 { font-size: 24px; }
  .st-key-mf-chips [data-testid="stHorizontalBlock"] { flex-direction: column; }
  .mf-bubble-user { max-width: 90%; }
  .mf-sources { padding: 16px; }
}
"""


def compact_html(markup: str) -> str:
    """Collapse indentation so Markdown never treats the block as code."""
    return " ".join(
        line.strip() for line in markup.strip().splitlines() if line.strip()
    )


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def render_navbar() -> None:
    st.markdown(
        compact_html(
            """
            <header class="mf-nav">
              <a class="mf-brand" href="#mf-home" aria-label="Groww Help home">
                <span class="mf-logo" aria-hidden="true">G</span>
                <span>Groww Help</span>
              </a>
              <nav class="mf-links" aria-label="Primary">
                <a href="#mf-home">Home</a>
                <a href="#mf-sources">Sources</a>
                <a href="#mf-about">About</a>
              </nav>
              <a class="mf-sources-btn" href="#mf-sources">Sources</a>
            </header>
            """
        ),
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        compact_html(
            """
            <section class="mf-hero" id="mf-home">
              <h1>Ask a Groww platform help question</h1>
              <p>I answer from official Groww help pages, with the source link.</p>
            </section>
            """
        ),
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        f'<footer class="mf-footer" role="contentinfo">{DISCLAIMER}</footer>',
        unsafe_allow_html=True,
    )


def render_sidebar_disclaimer() -> None:
    """Sidebar disclaimer required by IMPLEMENTATION §6.6.

    The sidebar starts collapsed; this content is available from the collapse
    control for anyone who wants the scope and privacy notes.
    """
    with st.sidebar:
        st.markdown("### About this demo")
        st.markdown(
            "**Facts-only. No investment advice.**\n\n"
            "This is an educational demo. It answers from a small set of "
            "official Groww help pages and may lag behind those pages."
        )
        st.markdown(
            "It does not collect PAN, Aadhaar, account numbers, OTPs, emails, "
            "or phone numbers. It does not give advice, returns, or "
            "recommendations."
        )
        st.caption("Official sources only · one source link per answer")


def load_sources(path: str | Path) -> list[dict[str, Any]]:
    """Read the briefed source list (never a network call)."""
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except (OSError, csv.Error):
        return []
    return sorted(rows, key=lambda row: str(row.get("id", "")))


def short_source_title(row: dict[str, Any]) -> str:
    return str(row.get("scheme") or "Official Groww help page")


def render_sources_panel(*, sources: list[dict[str, Any]]) -> None:
    cards: list[str] = []
    for row in sources:
        title = html.escape(short_source_title(row))
        url = html.escape(str(row.get("url") or ""), quote=True)
        if not url:
            continue
        cards.append(
            compact_html(
                f"""
                <a class="mf-source-card" href="{url}" target="_blank"
                   rel="noopener noreferrer"
                   aria-label="Open {title} on groww.in">
                  <span>
                    <span class="mf-source-title">{title}</span>
                    <span class="mf-source-meta">Official Groww page</span>
                  </span>
                  <span class="mf-link-icon">{LINK_ICON}</span>
                </a>
                """
            )
        )

    st.markdown(
        compact_html(
            f"""
            <aside class="mf-sources" id="mf-sources" aria-label="Sources used">
              <div class="mf-panel-head">
                <h2>Sources used</h2>
                <span class="mf-chip">{len(cards)} pages</span>
              </div>
              <p class="mf-panel-sub">Answers come only from these official Groww help pages.</p>
              <div class="mf-source-list">{''.join(cards)}</div>
              <section class="mf-about" id="mf-about">
                <h3>About</h3>
                <p>
                  A facts-only demo for Groww account, KYC, charges, order,
                  demat, and payment help. It does not collect PAN, Aadhaar,
                  account numbers, OTPs, emails, or phone numbers, and it does
                  not give advice, returns, or recommendations.
                </p>
                <p class="mf-about-disclaimer">{DISCLAIMER}</p>
              </section>
            </aside>
            """
        ),
        unsafe_allow_html=True,
    )


__all__ = [
    "DISCLAIMER",
    "inject_css",
    "load_sources",
    "render_footer",
    "render_hero",
    "render_navbar",
    "render_sidebar_disclaimer",
    "render_sources_panel",
    "short_source_title",
]
