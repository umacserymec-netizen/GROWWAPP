"""Topic resolution and intent routing for the Groww help corpus."""

from __future__ import annotations

from core.intent import detect_intent
from core.resolve import build_plan, match_topics

DELIVERY = "when-will-the-stocks-be-delivered-to-my-demat-account-2"
PASSWORD = "how-to-change-password-on-groww"


def test_stock_delivery_aliases_resolve_the_topic() -> None:
    for question in (
        "When will my stocks reach my demat account?",
        "When will my stocks reach?",
        "When do I get my stocks in my demat?",
    ):
        assert [item.key for item in match_topics(question)] == [DELIVERY]


def test_hdfc_question_does_not_match_a_groww_topic() -> None:
    assert match_topics("What is the expense ratio of HDFC Mid Cap Fund?") == []


def test_how_to_questions_are_in_scope_facts() -> None:
    plan, intent = build_plan("How do I change my Groww password?")
    assert intent.label == "fact"
    assert plan.scheme_slug == PASSWORD
    assert plan.field is None


def test_advice_is_refused_before_topic_matching() -> None:
    _, intent = build_plan("Should I buy HDFC Bank shares?")
    assert intent.label == "advice"


def test_performance_request_is_refused() -> None:
    result = detect_intent("What returns did this give last year?")
    assert result.label == "performance"


def test_out_of_scope_is_reported() -> None:
    plan, intent = build_plan("What is the expense ratio of HDFC Mid Cap Fund?")
    assert intent.label == "out_of_scope"
    assert any(note.startswith("out_of_scope:") for note in plan.notes)


def test_weather_is_an_unresolved_fact_question() -> None:
    plan, intent = build_plan("What is the weather today?")
    assert intent.label == "fact"
    assert plan.scheme_slug is None
