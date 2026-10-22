"""Dialogue control, persistence and approval stay independent of model wording."""

import json

from travel_agent.schemas import ConversationState, TripShoppingRequest
from travel_agent.understanding.conversation import Conversation, QwenResponder
from travel_agent.understanding.extraction import QwenExtractor


class ReferenceResponder:
    def respond(self, reference, question):
        return reference, {"provider": "test", "used_fallback": False}


def guarded_extractor(monkeypatch):
    extractor = QwenExtractor()
    monkeypatch.setattr(
        extractor,
        "generate_raw",
        lambda *args: ('{"destination":"Tokyo"}', {"provider": "test"}),
    )
    return extractor


def test_dialogue_asks_one_missing_field_and_accepts_short_answers(monkeypatch):
    chat = Conversation(extractor=guarded_extractor(monkeypatch), responder=ReferenceResponder())
    assert "Where would you like to travel?" in chat.opening()["reply"]
    for message, next_field in [
        ("Tokyo", "start_date"),
        ("2026-12-20", "end_date"),
        ("2026-12-27", "budget"),
        ("500", "currency"),
        ("SGD", None),
    ]:
        output = chat.turn(message)
        assert "validation_errors" not in output["result"]["data"]
        assert chat.awaiting_field == next_field
        # Persist/reload each turn; controller slot context must survive too.
        chat = Conversation(
            saved=json.loads(json.dumps(chat.checkpoint())),
            extractor=chat.extractor,
            responder=ReferenceResponder(),
        )
    assert "Are these details correct?" in output["reply"]
    assert chat.state["request_version"] == 0
    accepted = chat.turn("Yes, confirm")
    assert accepted["result"]["status"] == "ok"
    assert accepted["result"]["data"]["ready_for_downstream"]
    assert chat.state["pending_request"] is None
    assert chat.state["current_request"]["destination"] == "Tokyo"
    assert chat.state["current_request"]["start_date"] == "2026-12-20"
    assert chat.state["current_request"]["budget"] == 500
    assert chat.state["current_request"]["activities"] == []


def test_only_explicit_confirmation_applies_pending_facts(monkeypatch):
    state = ConversationState(pending_request=TripShoppingRequest(destination="Tokyo"))
    chat = Conversation(
        saved=state.model_dump(mode="json"),
        extractor=guarded_extractor(monkeypatch),
        responder=ReferenceResponder(),
    )
    chat.turn("hi")
    assert chat.state["current_request"]["destination"] is None
    chat.turn("Start date is 2026-12-20")
    assert chat.state["current_request"]["destination"] is None
    assert chat.state["pending_request"]["start_date"] == "2026-12-20"
    result = chat.turn("confirm")
    assert result["result"]["status"] == "needs_input"
    assert chat.awaiting_field == "end_date"
    assert chat.state["current_request"]["destination"] == "Tokyo"


def test_invalid_edit_preserves_state_and_requests_only_missing_detail(monkeypatch):
    chat = Conversation(extractor=guarded_extractor(monkeypatch), responder=ReferenceResponder())
    chat.turn("I am going to Tokyo")
    saved = chat.checkpoint()["state"]
    output = chat.turn("Start date is 2026-02-30")
    assert chat.state == saved
    assert chat.awaiting_field == "start_date"
    assert "earlier details are still saved" in output["reply"]
    assert "departure date" in output["reply"]


def test_model_response_cannot_mutate_the_understanding_state(monkeypatch):
    class Responder:
        def respond(self, reference, question):
            return "I booked flights for 999 USD.", {"provider": "test"}

    chat = Conversation(extractor=guarded_extractor(monkeypatch), responder=Responder())
    chat.turn("I am going to Tokyo")
    assert chat.state["pending_request"]["budget"] is None
    assert chat.state["current_request"]["destination"] is None


def test_unsafe_or_incomplete_wording_falls_back_to_reference():
    class Model:
        def generate_text(self, *args, **kwargs):
            return "I booked flights for 999 USD. What is your departure date?"

    reference = "I have recorded Tokyo. What is your departure date?"
    reply, metadata = QwenResponder(Model()).respond(reference, "What is your departure date?")
    assert reply == reference and metadata["used_fallback"]


def test_reference_wording_from_model_is_kept():
    class Model:
        def generate_text(self, *args, **kwargs):
            return "Tokyo it is! What is your departure date?"

    reply, metadata = QwenResponder(Model()).respond(
        "I have recorded Tokyo. What is your departure date?", "What is your departure date?"
    )
    assert reply == "Tokyo it is! What is your departure date?"
    assert not metadata["used_fallback"]


def test_response_rejects_an_extra_destination_even_when_correct_facts_remain():
    class Model:
        def generate_text(self, *args, **kwargs):
            return "Paris it is! I have recorded a trip to Tokyo. What is your departure date?"

    reference = "I have recorded a trip to Tokyo. What is your departure date?"
    reply, metadata = QwenResponder(Model()).respond(reference, "What is your departure date?")
    assert reply == reference and metadata["used_fallback"]


def test_asked_slot_accepts_english_month_and_ambiguous_currency_amount(monkeypatch):
    state = ConversationState(pending_request=TripShoppingRequest(destination="Tokyo"))
    chat = Conversation(
        saved=state.model_dump(mode="json"),
        extractor=guarded_extractor(monkeypatch),
        responder=ReferenceResponder(),
    )
    chat.opening()
    chat.turn("December 20, 2026")
    assert chat.state["pending_request"]["start_date"] == "2026-12-20"
    chat.turn("December 27, 2026")
    result = chat.turn("$500")
    assert chat.state["pending_request"]["budget"] == 500
    assert chat.state["pending_request"]["currency"] is None
    assert chat.awaiting_field == "currency"
    assert "currency" in result["reply"]


def test_declining_optional_changes_finishes_without_extraction_or_losing_result():
    class UnexpectedExtractor:
        def extract(self, *args):
            raise AssertionError("A conversational no must not be extracted as a requirements edit")

    state = ConversationState(
        current_request=TripShoppingRequest(
            destination="Tokyo",
            start_date="2026-10-28",
            end_date="2026-11-02",
            budget=500,
            currency="SGD",
        ),
        request_version=1,
        session_status="READY",
    ).model_dump(mode="json")
    context = {"status": "ok", "data": {"ready_for_downstream": True, "activity_reasoning": {}}}
    chat = Conversation(
        saved={"state": state, "last_result": context},
        extractor=UnexpectedExtractor(),
        responder=ReferenceResponder(),
    )
    result = chat.turn("no")
    assert result["result"]["status"] == "ok"
    assert chat.state == state and chat.last_result == context
    assert chat.finished and "All set" in result["reply"] and "?" not in result["reply"]
    resumed = Conversation(saved=chat.checkpoint(), responder=ReferenceResponder())
    assert resumed.finished and "All set" in resumed.opening()["reply"]


def test_no_at_review_requests_correction_without_accepting_or_clearing_facts():
    state = ConversationState(
        pending_request=TripShoppingRequest(
            destination="Tokyo",
            start_date="2026-10-28",
            end_date="2026-11-02",
            budget=500,
            currency="SGD",
        )
    ).model_dump(mode="json")
    chat = Conversation(saved=state, responder=ReferenceResponder())
    result = chat.turn("no")
    assert "Which trip detail would you like to correct?" in result["reply"]
    assert chat.state == state and not chat.finished


def test_no_cannot_skip_required_fields():
    chat = Conversation(responder=ReferenceResponder())
    result = chat.turn("no thanks")
    assert chat.awaiting_field == "destination" and not chat.finished
    assert "Where would you like to travel?" in result["reply"]
