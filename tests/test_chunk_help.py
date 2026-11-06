"""Chunking keeps each numbered how-to step with its number."""

from __future__ import annotations

from config.settings import load_settings
from core.types import LoadedDoc, Section
from ingest.chunk import chunk_document

TITLE = "How to change password on Groww?"
URL = "https://groww.in/help/my-account/ma-account-details/how-to-change-password-on-groww"


def test_ordered_steps_stay_on_one_line() -> None:
    settings = load_settings(require_provider_key=False)
    body = (
        "To change your password:\n"
        "1. Go to the 'Profile' section in the app.\n"
        "2. Click on the 'Settings' icon.\n"
        "3. Select 'Change Password'."
    )
    doc = LoadedDoc(
        doc_id="doc-test",
        source_id="2",
        scheme=TITLE,
        scheme_slug="how-to-change-password-on-groww",
        source_title=TITLE,
        source_url=URL,
        page_type="help",
        fetched_at="2026-11-06",
        sections=[Section(title=TITLE, body=body)],
    )
    chunks, _ = chunk_document(doc, settings)
    assert len(chunks) == 1
    text = chunks[0].text
    assert "1. Go to the 'Profile' section in the app." in text
    assert "2. Click on the 'Settings' icon." in text
    assert "3. Select 'Change Password'." in text
    assert "\n1.\n" not in text
