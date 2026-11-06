"""Extractive help fallback: verbatim prose, all how-to steps kept."""

from __future__ import annotations

from core.fallback import build_extractive_answer
from core.types import RetrievedChunk

URL = "https://groww.in/help/my-account/ma-account-details/how-to-change-password-on-groww"
TITLE = "How to change password on Groww?"


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="sha1:test",
        text=text,
        score=0.9,
        metadata={
            "page_type": "help",
            "chunk_type": "section_window",
            "section_title": TITLE,
            "source_title": TITLE,
            "source_url": URL,
            "scheme": TITLE,
            "scheme_slug": "how-to-change-password-on-groww",
            "field": "",
        },
    )


def test_how_to_steps_are_joined_into_one_sentence() -> None:
    text = (
        f"{TITLE} - To change your password:\n"
        "1. Go to the 'Profile' section in the app.\n"
        "2. Click on the 'Settings' icon.\n"
        "3. Select 'Change Password'.\n"
        "4. Enter your new password.\n"
        "5. Click on 'Send OTP' to complete the process."
    )
    answer = build_extractive_answer([_chunk(text)], field=None)
    assert answer is not None
    body = answer.text.split("[Source:", 1)[0]
    assert body.count(".") == 1  # intro + semicolon-joined steps is one sentence
    assert "Send OTP" in body
    assert "Select 'Change Password'" in body
    assert "; " in body
    assert answer.text.count("https://groww.in/help") == 1


def test_prose_answer_copies_two_sentences_verbatim() -> None:
    text = (
        f"{TITLE} - Any amount added to your Groww account is Available to "
        "Invest immediately. However, it becomes Available to Withdraw only "
        "after 1 working day. A third sentence must not appear."
    )
    answer = build_extractive_answer([_chunk(text)], field=None)
    assert answer is not None
    assert "Available to Invest immediately" in answer.text
    assert "after 1 working day" in answer.text
    assert "third sentence" not in answer.text
