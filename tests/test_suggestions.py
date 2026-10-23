"""Optional discovery is grounded in scoped KG evidence and explicit user selection."""

import json
from copy import deepcopy

from travel_agent.schemas import ConversationState, TripShoppingRequest
from travel_agent.understanding.conversation import Conversation
from travel_agent.understanding.place_engine import KG
from travel_agent.understanding.service import run
from travel_agent.understanding.suggestions import activity_suggestions, optional_suggestions

MINIMUM = {
    "destination": "Tokyo",
    "start_date": "2026-10-28",
    "end_date": "2026-11-02",
    "budget": 500,
    "currency": "SGD",
}


class NoExtractor:
    def extract(self, *args):
        raise AssertionError("Explicit menu choices must not be guessed by the model")


class ReferenceResponder:
    def respond(self, reference, question):
        return reference, {"provider": "test", "used_fallback": False}


def conversation():
    state = ConversationState(pending_request=TripShoppingRequest(**MINIMUM)).model_dump(
        mode="json"
    )
    return Conversation(
        saved=state,
        settings={"allow_draft": True},
        extractor=NoExtractor(),
        responder=ReferenceResponder(),
    )


def test_tokyo_activity_candidates_have_destination_paths_and_no_automatic_selection():
    request = TripShoppingRequest(**MINIMUM)
    before = request.model_dump()
    results = optional_suggestions(request, allow_draft=True)
    assert request.model_dump() == before
    assert {o["value"] for o in results["activities"]} == {
        "photo spot",
        "sightseeing",
        "museum visit",
        "religious visit",
    }
    for option in results["activities"]:
        assert option["scope"] == "destination" and option["knowledge_status"] == "approved"
        for evidence in option["evidence"]:
            assert evidence["hierarchy"][-1]["record"]["place_id"] == "local:tokyo"
            assert evidence["relationship"]["source_url"]
        assert not any("disney" in e["place_id"] for e in option["examples"])


def test_seoul_four_activity_options_have_scoped_source_evidence():
    request = TripShoppingRequest(**{**MINIMUM, "destination": "Seoul"})
    before = request.model_dump()
    options = optional_suggestions(request, allow_draft=True)["activities"]
    assert request.model_dump() == before
    assert {option["value"] for option in options} == {
        "shopping",
        "cafe visit",
        "sightseeing",
        "photo spot",
    }
    cafe = next(option for option in options if option["value"] == "cafe visit")
    assert cafe["label"] == "Café hopping"
    for option in options:
        assert option["scope"] == "destination"
        assert option["knowledge_status"] == "approved"
        for evidence in option["evidence"]:
            assert evidence["hierarchy"][-1]["record"]["place_id"] == "local:seoul"
            assert evidence["relationship"]["source_url"].startswith("https://")
    assert len(activity_suggestions(request, allow_draft=True, limit=3)) == 3
    assert optional_suggestions(request)["activities"]


def test_seoul_fourth_choice_and_cafe_label_remain_explicit_and_queryable():
    from travel_agent.understanding.knowledge import reason_activities, reason_places
    from travel_agent.web.chat import public_offer

    state = ConversationState(
        pending_request=TripShoppingRequest(**{**MINIMUM, "destination": "Seoul"})
    ).model_dump(mode="json")
    chat = Conversation(
        saved=state,
        settings={"allow_draft": True},
        extractor=NoExtractor(),
        responder=ReferenceResponder(),
    )
    chat.opening()
    offer = public_offer(chat.offered_options)
    assert len(offer["options"]) == 4
    fourth = chat.offered_options["options"][3]["value"]
    chat.turn("Café hopping and sightseeing")
    pending = TripShoppingRequest.model_validate(chat.state["pending_request"])
    assert pending.activities == ["cafe visit", fourth]
    assert chat.state["current_request"]["activities"] == []
    _, results, questions = reason_activities(pending, allow_draft=True)
    assert results["cafe visit"]["status"] == "ok" and not questions
    pending.places = ["local:seongsu_cafe_street"]
    place_result = reason_places(pending, allow_draft=True)
    assert place_result["places"][0]["selected_activities"] == ["cafe_visit"]
    numeric = Conversation(
        saved=state,
        settings={"allow_draft": True},
        extractor=NoExtractor(),
        responder=ReferenceResponder(),
    )
    numeric.opening()
    numeric.turn("1 and 4")
    assert numeric.state["pending_request"]["activities"] == ["cafe visit", fourth]


def test_unknown_destination_gets_clearly_generic_ideas_and_review_policy_is_respected():
    request = TripShoppingRequest(destination="Unmapped City")
    draft = optional_suggestions(request, allow_draft=True)
    assert draft["activities"] and all(o["scope"] == "general" for o in draft["activities"])
    assert all(o["examples"] == [] for o in draft["activities"])
    strict = optional_suggestions(request)
    assert strict["activities"] and strict["style"]
    assert all(o["knowledge_status"] == "approved" for o in strict["activities"])
    assert all(o["scope"] == "general" and o["examples"] == [] for o in strict["activities"])


def test_rejected_destination_is_not_used_as_location_evidence():
    places = KG()
    places.places["local:tokyo"]["review_status"] = "rejected"
    options = activity_suggestions(TripShoppingRequest(**MINIMUM), allow_draft=True, places=places)
    assert all(o["scope"] == "general" for o in options)


def test_styles_follow_selected_activity_and_never_offer_dress_codes_as_preferences():
    selected = TripShoppingRequest(**MINIMUM, activities=["photo spot"])
    result = optional_suggestions(selected, allow_draft=True)
    assert result["activities"] == []
    assert {o["value"] for o in result["style"]} == {"artistic", "maximalist", "retro"}
    assert all(o["scope"] == "selected_activities" and o["evidence"] for o in result["style"])
    general = optional_suggestions(TripShoppingRequest(**MINIMUM), allow_draft=True)
    assert all(o["scope"] == "general_preference" for o in general["style"])


def test_numeric_multi_selection_is_pending_then_queries_kg_after_confirmation():
    chat = conversation()
    before = deepcopy(chat.state)
    opening = chat.opening()
    assert "say skip" in opening["reply"]
    assert chat.awaiting_field == "optional_activities" and chat.state == before
    offered = deepcopy(chat.offered_options["options"])
    output = chat.turn("1 and 3")
    assert chat.state["pending_request"]["activities"] == [offered[0]["value"], offered[2]["value"]]
    assert chat.state["current_request"]["activities"] == []
    assert chat.awaiting_field == "optional_style"
    assert output["result"]["data"]["selection"]["offered_options"]["options"] == offered
    style = chat.offered_options["options"][0]["value"]
    chat.turn("1")
    assert chat.state["pending_request"]["style"] == [style]
    assert chat.awaiting_field is None
    result = chat.turn("confirm")
    assert result["result"]["status"] == "ok"
    assert result["result"]["data"]["ready_for_downstream"]
    assert len(result["result"]["data"]["activity_reasoning"]) == 2
    assert chat.state["current_request"]["style"] == [style]


def test_each_optional_step_can_be_skipped_and_stays_skipped_after_reload():
    chat = conversation()
    chat.opening()
    chat.turn("skip")
    assert chat.awaiting_field == "optional_style"
    chat.turn("no thanks")
    assert chat.awaiting_field is None
    assert (
        chat.state["pending_request"]["activities"] == chat.state["pending_request"]["style"] == []
    )
    reloaded = Conversation(
        saved=chat.checkpoint(),
        settings={"allow_draft": True},
        extractor=NoExtractor(),
        responder=ReferenceResponder(),
    )
    assert "Are these details correct?" in reloaded.opening()["reply"]
    result = reloaded.turn("confirm")
    assert (
        result["result"]["status"] == "ok" and result["result"]["data"]["activity_reasoning"] == {}
    )
    assert reloaded.state["current_request"]["activities"] == []
    assert reloaded.state["current_request"]["style"] == []


def test_selection_by_name_and_invalid_number_do_not_invent_fields():
    chat = conversation()
    chat.opening()
    before = deepcopy(chat.state)
    output = chat.turn("9")
    assert chat.state == before
    assert "displayed number" in output["reply"]
    output = chat.turn("museum visit and sightseeing")
    assert chat.state["pending_request"]["activities"] == ["sightseeing", "museum visit"]
    assert chat.state["pending_request"]["places"] == []
    assert chat.state["pending_request"]["budget"] == 500
    assert chat.awaiting_field == "optional_style"


def test_stale_numbers_cannot_select_options_from_a_previous_destination():
    chat = conversation()
    chat.opening()
    chat.state["pending_request"]["destination"] = "Singapore"
    before = deepcopy(chat.state)
    output = chat.turn("1")
    assert chat.state == before
    assert json.loads(chat.offered_options["scope"])["destination"] == "Singapore"
    assert "Singapore" in output["reply"]


def test_generic_yes_does_not_select_all_candidates_or_confirm_unreviewed_facts():
    chat = conversation()
    chat.opening()
    before = deepcopy(chat.state)
    output = chat.turn("yes")
    assert chat.state == before
    assert chat.awaiting_field == "optional_activities"
    assert "choose options or say skip" in output["reply"]


def test_service_returns_suggestions_separate_from_empty_reasoning():
    result = run({"request": MINIMUM, "allow_draft": True})
    assert result.status == "ok" and result.data["ready_for_downstream"]
    assert result.data["activity_reasoning"] == {}
    assert result.data["trip_request"]["activities"] == []
    assert result.data["optional_suggestions"]["activities"]
    assert result.data["optional_suggestions"]["selection_required"]
