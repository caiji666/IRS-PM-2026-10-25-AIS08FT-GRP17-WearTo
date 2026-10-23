"""Web chat shares the real controller; only model generation/geocoding are injected."""

import json

import pytest

from travel_agent.understanding.extraction import QwenExtractor
from travel_agent.web.chat import accept_review, advance_chat, public_offer

LOCATION = {
    "name": "Tokyo",
    "country": "Japan",
    "country_code": "JP",
    "latitude": 35.68,
    "longitude": 139.69,
    "timezone": "Asia/Tokyo",
    "feature_code": "PPLC",
}


class ReferenceResponder:
    def respond(self, reference, question):
        return reference, {"provider": "test", "used_fallback": False}


@pytest.fixture
def dialogue(monkeypatch):
    extractor = QwenExtractor()
    monkeypatch.setattr(
        extractor, "generate_raw", lambda *a: ('{"destination":"Tokyo"}', {"provider": "test"})
    )
    return extractor


def send(conversation, message, extractor, geocode=None):
    conversation = json.loads(json.dumps(conversation))
    conversation.setdefault("messages", []).append({"role": "user", "content": message})
    conversation["turn"] = conversation.get("turn", 0) + 1
    return advance_chat(
        conversation,
        extractor=extractor,
        responder=ReferenceResponder(),
        geocode=geocode or (lambda city: [LOCATION]),
    )


def collect(extractor):
    conversation = {}
    for message in ("Tokyo", "2026-12-20", "2026-12-27", "500 SGD"):
        conversation, audit, review = send(conversation, message, extractor)
        assert review is None
    return conversation


def test_web_keeps_incremental_facts_and_offers_optional_choices(dialogue):
    conversation = collect(dialogue)
    assert conversation["request"]["destination"] == "Tokyo"
    assert conversation["request"]["start_date"] == "2026-12-20"
    assert conversation["request"]["budget"] == 500
    assert conversation["pending_field"] == "optional_activities"
    assert conversation["status"] == "collecting"
    assert conversation["dialogue"]["state"]["current_request"]["destination"] is None
    conversation, _, review = send(conversation, "2 and 3", dialogue)
    assert review is None and conversation["request"]["activities"] == [
        "sightseeing",
        "museum visit",
    ]
    assert conversation["pending_field"] == "optional_style"
    conversation, _, review = send(conversation, "skip", dialogue)
    assert conversation["status"] == "review"
    assert review["request"]["style"] == []
    assert review["locations"][0]["country_code"] == "JP"
    assert conversation["dialogue"]["state"]["pending_request"] is not None
    accepted = accept_review(conversation, review["request"])
    assert accepted["dialogue"]["state"]["pending_request"] is None
    assert accepted["dialogue"]["state"]["current_request"]["budget"] == 500
    assert conversation["dialogue"]["state"]["pending_request"] is not None


def test_web_allows_skip_both_and_keeps_empty_arrays(dialogue):
    conversation = collect(dialogue)
    conversation, _, review = send(conversation, "skip all", dialogue)
    assert review is not None
    assert review["request"]["activities"] == review["request"]["style"] == []
    assert review["dress_code_options"] == {}
    assert conversation["status"] == "review"


def test_invalid_input_preserves_saved_fields_and_asks_targeted_question(dialogue):
    conversation, _, _ = send({}, "Tokyo", dialogue)
    before = json.loads(json.dumps(conversation["request"]))
    conversation, audit, review = send(conversation, "2026-02-30", dialogue)
    assert review is None and conversation["request"] == before
    assert conversation["pending_field"] == "start_date"
    assert audit["understanding"]["data"]["validation_errors"]
    assert "departure date" in conversation["messages"][-1]["content"]


def test_country_only_requests_city_without_losing_other_facts(dialogue):
    conversation, _, review = send({}, "I am going to Japan. Budget 500 SGD.", dialogue)
    assert review is None and conversation["pending_field"] == "destination"
    assert conversation["request"]["budget"] == 500
    assert "Which city in Japan" in conversation["messages"][-1]["content"]
    assert "details are saved" not in conversation["messages"][-1]["content"]


def test_geocoding_failure_preserves_completed_request(dialogue):
    conversation = collect(dialogue)
    conversation, _, review = send(conversation, "skip all", dialogue, geocode=lambda city: [])
    assert review is None and conversation["request"]["destination"] == "Tokyo"
    assert conversation["request"]["budget"] == 500
    assert conversation["status"] == "collecting"


def test_public_offer_excludes_raw_evidence_and_changes_key_with_menu(dialogue):
    conversation = collect(dialogue)
    offered = conversation["dialogue"]["offered_options"]
    offered["options"][0]["evidence"] = [
        {"private_path": "/private/internal", "raw_output": "private"}
    ]
    public = public_offer(offered)
    assert len(public["options"]) == 4
    assert "private" not in json.dumps(public)
    assert "evidence" not in json.dumps(public)
    offered["scope"] = "changed"
    assert public_offer(offered)["key"] != public["key"]


def test_missing_dress_code_is_optional_at_review(dialogue):
    conversation = {}
    message = (
        "Tokyo, 2026-12-20 to 2026-12-27. Dinner. Budget 500 SGD. I prefer minimalist outfits."
    )
    conversation, _, review = send(conversation, message, dialogue)
    assert review is not None and conversation["status"] == "review"
    assert review["request"]["dress_codes"] == {}
    assert "dinner" in review["dress_code_options"]


def test_optional_code_can_be_cleared_in_review_without_changing_other_fields():
    from travel_agent.web.review import confirmed_request

    request = {"activities": ["dinner"], "dress_codes": {"dinner": "smart casual"}}
    assert confirmed_request(request, {"dinner": ""})["dress_codes"] == {}
    assert request["dress_codes"] == {"dinner": "smart casual"}


def test_http_exposes_only_public_choices_and_rejects_stale_selection(
    tmp_path, monkeypatch, dialogue
):
    import http.client
    import threading
    from http.server import ThreadingHTTPServer

    from travel_agent.web import __main__ as web

    monkeypatch.setattr(web, "ROOT", tmp_path)
    monkeypatch.setattr(web, "JOBS", tmp_path / "jobs")
    job = "a" * 32
    folder = web.JOBS / job
    folder.mkdir(parents=True)
    conversation = collect(dialogue)
    web.write(folder / "conversation.json", conversation)
    web.write(folder / "status.json", {"id": job, "status": "collecting"})
    called = []
    finished = threading.Event()

    def fake_execute(job, mode):
        called.append(mode)
        web.ACTIVE.release()
        finished.set()

    monkeypatch.setattr(web, "execute", fake_execute)
    server = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
    try:
        connection.request("GET", "/api/job/" + job)
        response = connection.getresponse()
        output = json.loads(response.read())
        assert response.status == 200
        assert "evidence" not in json.dumps(output)
        assert "raw_output" not in json.dumps(output)
        key = output["optional_offer"]["key"]
        headers = {"Content-Type": "application/json", "X-Trip-UI": "1"}
        connection.request(
            "POST",
            "/api/chat",
            json.dumps({"id": job, "message": "1", "offer_key": "stale"}),
            headers,
        )
        response = connection.getresponse()
        assert response.status == 400
        response.read()
        assert called == [] and web.read(folder / "conversation.json") == conversation
        connection.request(
            "POST",
            "/api/chat",
            json.dumps({"id": job, "message": "2 and 3", "offer_key": key}),
            headers,
        )
        response = connection.getresponse()
        assert response.status == 202
        response.read()
        assert finished.wait(1) and called == ["chat"]
        stored = web.read(folder / "conversation.json")["messages"][-1]
        assert stored == {"role": "user", "content": "2 and 3", "source": "option_selection"}
        connection.request("GET", "/api/job/" + job)
        response = connection.getresponse()
        output = json.loads(response.read())
        assert response.status == 200
        assert stored["content"] not in [item["content"] for item in output["messages"]]
    finally:
        connection.close()
        server.shutdown()
        server.server_close()


def test_choices_and_review_use_ui_without_extra_model_narration(dialogue):
    class UnexpectedResponder:
        def respond(self, *args):
            raise AssertionError("Choice and review UI must not generate duplicate prose")

    conversation = collect(dialogue)
    assert conversation["messages"][-1] == {
        "role": "assistant",
        "content": "What would you like to do in Tokyo?",
    }
    conversation["messages"].append({"role": "user", "content": "photo spot"})
    conversation, _, review = advance_chat(
        conversation,
        extractor=dialogue,
        responder=UnexpectedResponder(),
        geocode=lambda city: [LOCATION],
    )
    assert review is None and conversation["pending_field"] == "optional_style"
    assert conversation["messages"][-1] == {
        "role": "assistant",
        "content": "Do you have a preferred style?",
    }
    conversation["messages"].append({"role": "user", "content": "retro"})
    conversation, _, review = advance_chat(
        conversation,
        extractor=dialogue,
        responder=UnexpectedResponder(),
        geocode=lambda city: [LOCATION],
    )
    assert review is not None and conversation["status"] == "review"
    assert conversation["messages"][-1] == {"role": "user", "content": "retro"}
    assert conversation["request"]["activities"] == ["photo spot"]
    assert conversation["request"]["style"] == ["retro"]


def test_legacy_redundant_prose_is_hidden_without_removing_user_messages():
    from travel_agent.web.chat import public_messages

    source = [
        {
            "role": "assistant",
            "content": "For Tokyo, here are optional activity ideas: choose one.",
        },
        {"role": "assistant", "content": "Here are your accumulated trip details: Tokyo."},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "Here are your accumulated trip details: Tokyo."},
        {"role": "assistant", "content": "What is your shopping budget?"},
    ]
    assert public_messages(source) == source[3:]
    assert len(source) == 5


def test_setup_failure_is_reported_as_setup_not_weather():
    from travel_agent.web.__main__ import pipeline_failure

    stage, message = pipeline_failure(
        {"stage": "setup", "reason_code": "missing_unsplash_config"}, "run"
    )
    assert stage == "setup" and "Unsplash" in message and "Weather" not in message
    assert pipeline_failure({"stage": "weather"}, "run")[0] == "weather"


def test_option_replies_are_hidden_but_numeric_trip_answers_are_preserved():
    from travel_agent.web.chat import public_messages

    source = [
        {"role": "assistant", "content": "What is your shopping budget?"},
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "For Tokyo, here are optional activity ideas:"},
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "Here are optional aesthetic directions; choose."},
        {"role": "user", "content": "3"},
        {"role": "user", "content": "retro", "source": "option_selection"},
        {"role": "user", "content": "skip", "source": "option_selection"},
        {"role": "user", "content": "3"},
    ]
    assert public_messages(source) == source[:2] + source[-1:]
    assert len(source) == 9 and source[3]["content"] == "1"


@pytest.mark.parametrize(
    "dates",
    [
        "2026/10/28 to 2026/11/02",
        "28 Oct 2026 to 2 Nov 2026",
        "October 28, 2026 to November 2, 2026",
    ],
)
def test_flexible_dates_with_other_conditions_are_recorded_together(dialogue, dates):
    conversation, _, _ = send({}, f"Tokyo, {dates}, budget 500 SGD", dialogue)
    assert conversation["request"]["start_date"] == "2026-10-28"
    assert conversation["request"]["end_date"] == "2026-11-02"
    assert conversation["request"]["budget"] == 500
    assert conversation["pending_field"] == "optional_activities"


def test_ambiguous_dates_save_budget_and_do_not_open_review(dialogue):
    conversation, _, review = send({}, "Tokyo, 03/04/2026, budget 500 SGD", dialogue)
    assert review is None and conversation["request"]["budget"] == 500
    assert conversation["request"]["start_date"] is None
    assert "2026-03-04 or 2026-04-03" in conversation["messages"][-1]["content"]
    conversation, _, review = send(conversation, "confirm", dialogue)
    assert review is None and conversation["request"]["start_date"] is None


def test_year_only_followup_completes_saved_dates_without_replaying_budget(dialogue):
    conversation, _, _ = send({}, "Tokyo, Oct 28 to Nov 2, budget 500 SGD", dialogue)
    assert conversation["request"]["budget"] == 500
    assert "Which year" in conversation["messages"][-1]["content"]
    conversation, _, _ = send(conversation, "Change budget to 600 SGD", dialogue)
    assert conversation["request"]["budget"] == 600
    conversation, _, _ = send(conversation, "2026", dialogue)
    assert conversation["request"]["start_date"] == "2026-10-28"
    assert conversation["request"]["end_date"] == "2026-11-02"
    assert conversation["request"]["budget"] == 600
    assert conversation["pending_field"] == "optional_activities"


def test_single_date_followup_can_also_supply_budget(dialogue):
    conversation, _, _ = send({}, "Tokyo", dialogue)
    conversation, _, _ = send(conversation, "28 Oct 2026, budget 500 SGD", dialogue)
    assert conversation["request"]["start_date"] == "2026-10-28"
    assert conversation["request"]["budget"] == 500
    assert conversation["pending_field"] == "end_date"
    conversation, _, _ = send(conversation, "2 Nov 2026", dialogue)
    assert conversation["request"]["end_date"] == "2026-11-02"


def test_duration_question_routes_next_date_to_return_slot(dialogue):
    conversation, _, _ = send({}, "Tokyo", dialogue)
    conversation, _, _ = send(
        conversation, "depart 2026/10/28 for five days, budget 500 SGD", dialogue
    )
    assert conversation["request"]["start_date"] == "2026-10-28"
    assert conversation["pending_field"] == "end_date"
    conversation, _, _ = send(conversation, "2 Nov 2026", dialogue)
    assert conversation["request"]["start_date"] == "2026-10-28"
    assert conversation["request"]["end_date"] == "2026-11-02"


def test_year_reply_keeps_already_explicit_date(dialogue):
    conversation, _, _ = send({}, "Tokyo, October 28, 2026 to Nov 2, budget 500 SGD", dialogue)
    assert conversation["request"]["start_date"] == "2026-10-28"
    conversation, _, _ = send(conversation, "2026", dialogue)
    assert conversation["request"]["start_date"] == "2026-10-28"
    assert conversation["request"]["end_date"] == "2026-11-02"


def test_activity_question_uses_actual_destination(dialogue):
    conversation, _, _ = send({}, "Beijing, 2026-10-23 to 2026-11-01, budget 200 SGD", dialogue)
    assert conversation["messages"][-1] == {
        "role": "assistant",
        "content": "What would you like to do in Beijing?",
    }
    assert conversation["dialogue"]["offered_options"]["field"] == "activities"


def test_legacy_saved_details_filler_is_removed_only_from_assistant_display():
    from travel_agent.web.chat import public_messages

    source = [
        {
            "role": "assistant",
            "content": "Which city in china will you be visiting? Your other details are saved.",
        },
        {"role": "user", "content": "Your other details are saved."},
    ]
    assert public_messages(source) == [
        {"role": "assistant", "content": "Which city in china will you be visiting?"},
        source[1],
    ]
    assert source[0]["content"].endswith("Your other details are saved.")
