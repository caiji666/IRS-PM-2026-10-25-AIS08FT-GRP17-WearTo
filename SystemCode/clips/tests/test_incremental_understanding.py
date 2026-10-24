"""Session regressions exercise real extraction guards without loading Qwen."""

import json

import pytest

from travel_agent.schemas import ConversationState, TripShoppingRequest
from travel_agent.understanding.extraction import QwenExtractor
from travel_agent.understanding.service import run

MINIMUM = {
    "destination": "Seoul",
    "start_date": "2026-12-20",
    "end_date": "2026-12-27",
    "budget": 500,
    "currency": "SGD",
}


def extractor_for(monkeypatch, candidate=None):
    extractor = QwenExtractor()
    monkeypatch.setattr(
        extractor,
        "generate_raw",
        lambda *args: (json.dumps(candidate or {}), {"provider": "test"}),
    )
    return extractor


def test_preview_facts_accumulate_across_turns_then_confirm_without_repetition(monkeypatch):
    extractor = extractor_for(monkeypatch, {"destination": "Seoul"})
    state = {}
    for message, missing in [
        ("Seoul", {"start_date", "end_date", "budget", "currency"}),
        ("Start date is 2026-12-20", {"end_date", "budget", "currency"}),
        ("2026-12-27", {"budget", "currency"}),
        ("500 SGD", set()),
    ]:
        result = run({"state": state, "message": message}, extractor=extractor)
        assert result.status == "needs_input"
        assert set(result.data["missing_fields"]) == missing
        state = result.data["state"]
        assert state["request_version"] == 0
        assert state["current_request"]["destination"] is None
    assert {k: state["pending_request"][k] for k in MINIMUM} == MINIMUM
    accepted = run({"state": state, "accept_extraction": True})
    assert accepted.status == "ok"
    assert accepted.data["ready_for_downstream"]
    assert accepted.data["state"]["pending_request"] is None
    assert accepted.data["state"]["request_version"] == 1
    assert {k: accepted.data["trip_request"][k] for k in MINIMUM} == MINIMUM


def test_budget_amount_and_currency_can_be_separate_answers(monkeypatch):
    state = run({"request": {k: v for k, v in MINIMUM.items() if k not in {"budget", "currency"}}})
    extractor = extractor_for(monkeypatch)
    amount = run({"state": state.data["state"], "message": "500"}, extractor=extractor)
    assert amount.data["proposed_request"]["budget"] == 500
    currency = run({"state": amount.data["state"], "message": "SGD"}, extractor=extractor)
    assert currency.data["proposed_request"]["budget"] == 500
    assert currency.data["missing_fields"] == []
    # The agent's structured acceptance path applies accumulated facts, too.
    accepted = run({"state": currency.data["state"], "patch": {}})
    assert accepted.status == "ok" and accepted.data["trip_request"]["currency"] == "SGD"


def test_short_activity_reply_and_invalid_edit_preserve_accumulated_facts(monkeypatch):
    first = run({"request": MINIMUM})
    extractor = extractor_for(monkeypatch)
    reply = run({"state": first.data["state"], "message": "Hiking"}, extractor=extractor)
    assert reply.data["proposed_request"]["activities"] == ["hiking"]
    invalid = run({"state": reply.data["state"], "patch": {"set": {"end_date": "2020-01-01"}}})
    assert invalid.status == "needs_input"
    assert invalid.data["state"] == reply.data["state"]
    accepted = run({"state": invalid.data["state"], "patch": {"set": {"budget": 300}}})
    assert accepted.data["trip_request"]["activities"] == ["hiking"]
    assert accepted.data["trip_request"]["destination"] == "Seoul"
    assert accepted.data["trip_request"]["budget"] == 300


def test_minimum_request_allows_empty_results_without_loading_activity_backend():
    result = run({"request": MINIMUM, "kg_backend": "neo4j"})
    assert result.status == "ok" and result.data["ready_for_downstream"]
    assert result.data["questions"] == result.data["optional_questions"] == []
    assert result.data["activity_reasoning"] == {}
    assert result.data["activity_style_requirements"] == {}
    assert result.data["activity_style_options"] == {}
    assert result.data["activity_dressing_requirements"] == {}
    assert result.data["place_context"] is None
    assert not any(result.data["weather_clothing_requirements"].values())


@pytest.mark.parametrize("empty", [None, []])
def test_optional_collection_inputs_accept_empty_values(empty):
    result = run(
        {
            "request": MINIMUM
            | {
                "activities": empty,
                "style": empty,
                "excluded_items": empty,
                "places": empty,
                "owned_item_ids": empty,
                "dress_codes": None,
            }
        }
    )
    assert result.status == "ok"
    assert result.data["trip_request"]["activities"] == []
    assert result.data["trip_request"]["dress_codes"] == {}


def test_missing_dress_code_and_unknown_activity_are_nonblocking():
    result = run(
        {"request": MINIMUM | {"activities": ["dinner", "scuba diving"]}, "allow_draft": True}
    )
    assert result.status == "ok" and result.data["ready_for_downstream"]
    assert result.data["questions"] == []
    assert result.data["optional_questions"][0]["field"] == "dress_codes.dinner"
    assert result.data["activity_reasoning"]["dinner"]["status"] == "needs_clarification"
    assert result.data["activity_style_requirements"]["scuba diving"] == []
    assert result.data["context_status"] == "partial"


@pytest.mark.parametrize("message", ["No preference", "No activities", "Skip optional details"])
def test_optional_answers_can_be_empty(monkeypatch, message):
    state = ConversationState(current_request=TripShoppingRequest(**MINIMUM), request_version=1)
    result = run(
        {"state": state.model_dump(mode="json"), "message": message},
        extractor=extractor_for(monkeypatch),
    )
    assert "validation_errors" not in result.data
    accepted = run({"state": result.data["state"], "patch": {}})
    assert accepted.status == "ok"
    assert accepted.data["trip_request"]["style"] == []
    assert accepted.data["trip_request"]["activities"] == []


def test_missing_date_role_is_clarified_without_dropping_other_facts(monkeypatch):
    state = run({"request": {"destination": "Seoul"}}).data["state"]
    result = run({"state": state, "message": "2026-12-20"}, extractor=extractor_for(monkeypatch))
    assert result.status == "needs_input" and result.data["state"] == state
    assert result.data["state"]["current_request"]["destination"] == "Seoul"


def test_bare_style_reply_is_saved_without_becoming_a_destination(monkeypatch):
    first = run({"request": MINIMUM})
    result = run(
        {"state": first.data["state"], "message": "Minimalist"},
        extractor=extractor_for(monkeypatch, {"set": {"destination": "Minimalist"}}),
    )
    assert result.data["proposed_request"]["style"] == ["minimalist"]
    assert result.data["proposed_request"]["destination"] == "Seoul"


def test_tool_cannot_self_accept_a_saved_preview():
    from travel_agent.understanding import tools

    # Use a structured pending state to isolate the wrapper's approval boundary.
    state = ConversationState(pending_request=TripShoppingRequest(**MINIMUM))
    result = tools.understand_and_reason.invoke(
        {"payload": {"state": state.model_dump(mode="json"), "accept_extraction": True}}
    )
    assert result["status"] == "needs_input"
    assert result["data"]["state"] == state.model_dump(mode="json")
