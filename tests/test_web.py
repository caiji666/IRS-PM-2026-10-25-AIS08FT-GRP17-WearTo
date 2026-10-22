"""Local UI input boundaries and private-artifact isolation; no model/API calls."""

import http.client
import json
import threading
from datetime import datetime, timedelta
from http.server import ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from travel_agent.web import __main__ as web


def form():
    today = datetime.now(ZoneInfo("Asia/Singapore")).date()
    return {
        "country": "KR",
        "city": "Seoul",
        "start_date": str(today + timedelta(days=1)),
        "end_date": str(today + timedelta(days=4)),
        "activities": ["sightseeing", "dinner"],
        "budget": 500,
        "currency": "SGD",
        "style": "minimalist",
        "dress_code": "smart casual",
    }


def test_form_requires_trip_fields_but_allows_optional_fields_and_longer_dates():
    assert web.validate_form(form())["country_name"] == "South Korea"
    assert web.validate_form(
        {**form(), "country": "US", "city": "Los Angeles", "activities": ["beach walk"]}
    )["activities"] == ["beach walk"]
    assert web.validate_form({**form(), "country": "JP", "city": "Tokyo"})["city"] == "Tokyo"
    assert web.validate_form({**form(), "activities": [], "style": ""})["activities"] == []
    for updates in (
        {"country": "XX"},
        {"budget": -1},
        {"currency": "XXX"},
        {"activities": ["x"] * 9},
        {"start_date": "2026-02-30"},
        {"notes": "x" * 1501},
    ):
        with pytest.raises(ValueError):
            web.validate_form({**form(), **updates})


def test_result_view_excludes_embeddings_paths_and_raw_data(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "ROOT", tmp_path)
    stamp = "20261021_153732_278241+0800"
    folder = tmp_path / "runs/product_retrieval" / stamp
    folder.mkdir(parents=True)
    report = {
        "bundle": {
            "top_k_products": {
                "TOP": [
                    {
                        "product_id": "example",
                        "title": "Example",
                        "image_path": "product_images/image.jpg",
                        "embedding": {"secret-internal-vector": [1]},
                        "provenance": {"original": {"private": "/private/data"}},
                        "evidence": {"availability_verified": True},
                        "availability": "IN_STOCK",
                    }
                ]
            },
            "product_ranking": {"ranked_products": {"TOP": [{"score": 0.5}]}},
            "owned_items": [],
        }
    }
    (folder / "result.json").write_text(json.dumps(report))
    output = web.public_result(stamp)
    assert output["count"] == 1
    assert output["products"]["TOP"][0]["image"] == f"/media/{stamp}/product_images/image.jpg"
    assert "secret-internal-vector" not in json.dumps(output)
    assert "/private/data" not in json.dumps(output)
    with pytest.raises(ValueError):
        web.public_result("../../.secrets")


def test_local_http_rejects_foreign_origin_host_and_private_files(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "ROOT", tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port)
        connection.request("GET", "/api/history", headers={"Host": "attacker.example"})
        response = connection.getresponse()
        assert response.status == 403
        response.read()
        connection.request(
            "POST",
            "/api/prepare",
            body="{}",
            headers={
                "Host": f"127.0.0.1:{port}",
                "Origin": "https://attacker.example",
                "X-Trip-UI": "1",
            },
        )
        response = connection.getresponse()
        assert response.status == 403
        response.read()
        connection.request("GET", "/../../.secrets/ebay-production.json")
        response = connection.getresponse()
        assert response.status == 404
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()


def test_legacy_preview_url_serves_existing_generated_png(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "ROOT", tmp_path)
    name = "a" * 32 + ".png"
    preview = tmp_path / "artifacts/previews" / name
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"\x89PNG\r\n\x1a\nlegacy-preview")
    server = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port)
        connection.request(
            "GET",
            f"/artifacts/previews/{name}",
            headers={"Host": f"127.0.0.1:{port}"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "image/png"
        assert response.read().startswith(b"\x89PNG\r\n\x1a\n")
        connection.close()
    finally:
        server.shutdown()
        server.server_close()


def test_review_allows_missing_activity_codes_but_checks_explicit_choices():
    from travel_agent.web.review import confirmed_request, dress_code_options

    request = {"activities": ["dinner", "business meeting"], "dress_codes": {}}
    options = dress_code_options(request)
    assert "smart casual" in options["dinner"]
    assert options["business meeting"] == ["business casual", "business formal"]
    assert confirmed_request(request, {})["dress_codes"] == {}
    assert confirmed_request(request, {"dinner": "smart casual"})["dress_codes"] == {
        "dinner": "smart casual"
    }
    with pytest.raises(ValueError, match="business"):
        confirmed_request(request, {"dinner": "smart casual", "business meeting": "casual"})
    fixed = confirmed_request(
        request, {"dinner": "smart casual", "business meeting": "business casual"}
    )
    assert fixed["dress_codes"]["business meeting"] == "business casual"
    assert request["dress_codes"] == {}


def test_no_occasion_code_is_invented_for_sightseeing():
    from travel_agent.web.review import confirmed_request, dress_code_options

    request = {"activities": ["sightseeing"], "dress_codes": {}}
    assert dress_code_options(request) == {}
    assert confirmed_request(request, {})["dress_codes"] == {}


def test_student5_outfit_tabs_reuse_product_rows_and_details():
    static = Path(web.STATIC)
    html = (static / "index.html").read_text(encoding="utf-8")
    script = (static / "app.js").read_text(encoding="utf-8")

    assert 'id="outfitTabs"' in html
    assert 'role="tablist" aria-label="Recommended outfits"' in html
    assert "`OUTFIT ${index+1}`" in script
    assert "adaptStudent5Product" in script
    assert "renderOutfitProducts" in script
    assert "showProduct(item)" in script
    assert "final_outfit_score" not in script


def test_student5_ui_has_real_confirm_refine_and_preview_actions():
    static = Path(web.STATIC)
    html = (static / "index.html").read_text(encoding="utf-8")
    script = (static / "app.js").read_text(encoding="utf-8")

    assert 'id="historySelect"' not in html
    assert "Previous search" not in html
    assert 'id="confirmOutfit"' in html
    assert 'id="generatePreview"' in html
    assert 'id="previewImage"' in html
    assert "state.outfits.length" in script
    assert "/api/student5/confirm" in script
    assert "/api/student5/preview" in script
    assert "/api/student5/refine" in script


def test_student5_ui_shows_loading_adjustment_and_preview_dialog():
    static = Path(web.STATIC)
    html = (static / "index.html").read_text(encoding="utf-8")
    script = (static / "app.js").read_text(encoding="utf-8")

    assert 'id="adjustRequirements"' in html
    assert 'id="previewDialog"' in html
    assert 'id="previewLoading"' in html
    assert 'id="previewImage"' in html
    assert 'id="downloadPreview"' in html
    assert "scrollIntoView" in script
    assert "showModal()" in script
    assert "aria-busy" in script


def test_student5_ui_can_confirm_and_track_an_upstream_refinement():
    static = Path(web.STATIC)
    html = (static / "index.html").read_text(encoding="utf-8")
    script = (static / "app.js").read_text(encoding="utf-8")

    assert 'id="refinementDialog"' in html
    assert 'id="refinementFields"' in html
    assert 'id="confirmRefinement"' in html
    assert "/api/student5/refine/confirm" in script
    assert "confirmation_required" in script
    assert "pollStudent5Refinement" in script
    assert "job.student5" in script


def test_fixed_student5_template_cache_reuses_calculation_and_returns_copies(tmp_path):
    source = tmp_path / "ranked_outfits.json"
    source.write_text("first", encoding="utf-8")
    cache = {}
    calls = []

    def calculate():
        calls.append(True)
        return {
            "student4": {"large_read_only_payload": [1, 2, 3]},
            "state": {"recommendation_round": 1},
        }

    first = web.cached_file_template(source, cache, calculate)
    first["state"]["recommendation_round"] = 99
    second = web.cached_file_template(source, cache, calculate)
    assert len(calls) == 1
    assert second["state"]["recommendation_round"] == 1
    assert first["student4"] is second["student4"]

    source.write_text("second version", encoding="utf-8")
    third = web.cached_file_template(source, cache, calculate)
    assert len(calls) == 2
    assert third["state"]["recommendation_round"] == 1


def test_student5_session_rejects_stale_selection_and_clears_confirmation(tmp_path):
    from travel_agent.web.student5_session import SessionError, confirm, refine

    current = {
        "state": {"recommendation_round": 2},
        "recommendation_set": {"recommendations": [{"outfit_id": "NEW"}]},
        "confirmed_outfit": {"selected_outfit_id": "OLD"},
        "preview": {"status": "SUCCESS"},
    }
    with pytest.raises(SessionError, match="current Top-3"):
        confirm(current, "OLD", run=lambda payload: None)

    class Result:
        status = "ok"
        reason = None

        def __init__(self, data):
            self.data = data

        def model_dump(self, mode="json"):
            return {"data": self.data}

    calls = []

    def fake_run(payload):
        calls.append(payload)
        if payload["action"] == "modify_requirements":
            return Result(
                {
                    "state": {"recommendation_round": 2},
                    "updated_request": {"budget": "200", "currency": "USD"},
                    "changed": True,
                    "changed_fields": ["budget"],
                }
            )
        return Result(
            {
                "state": {"recommendation_round": 3},
                "recommendation_set": {"recommendations": [{"outfit_id": "NEXT"}]},
                "filter_summary": {},
            }
        )

    current["student4"] = {"status": "ok", "outfits": []}
    updated = refine(current, {"intent": "MODIFY_REQUIREMENTS", "set": {"budget": "200"}}, fake_run)
    assert [call["action"] for call in calls] == ["modify_requirements", "recommend"]
    assert updated["confirmed_outfit"] is None
    assert updated["preview"] is None
    assert updated["recommendation_set"]["recommendations"][0]["outfit_id"] == "NEXT"


def test_student5_refinement_chat_parses_budget_size_and_exclusion():
    from travel_agent.web.student5_session import patch_from_message

    assert patch_from_message("Change my budget to 200 USD")["set"]["budget"] == "200"
    assert patch_from_message("My size is M")["set"]["user_size"] == "M"
    assert patch_from_message("Exclude product ebay:123")["add"]["excluded_items"] == ["ebay:123"]
    assert patch_from_message("Change my destination to Tokyo") == {
        "kind": "upstream",
        "message": "Change my destination to Tokyo",
    }


def _student5_web_session_request():
    return {
        "destination": "Seoul",
        "start_date": "2026-11-01",
        "end_date": "2026-11-05",
        "activities": ["sightseeing"],
        "budget": 500,
        "currency": "USD",
        "style": ["minimalist"],
        "excluded_items": [],
        "user_size": "M",
    }


def _student5_web_session():
    request = _student5_web_session_request()
    return {
        "request": request,
        "state": {
            "session_id": "web-test",
            "request_version": 4,
            "current_request": request,
            "recommendation_round": 2,
            "last_recommended_outfit_ids": ["OLD"],
            "selected_outfit_id": None,
            "session_status": "AWAITING_CONFIRMATION",
        },
    }


class _UnderstandingResult:
    status = "needs_input"
    reason = "Human extraction review required"

    def __init__(self, data):
        self.data = data

    def model_dump(self, mode="json"):
        return {"status": self.status, "reason": self.reason, "data": self.data}


@pytest.mark.parametrize(
    ("message", "updates", "changed_fields", "rerun_from_stage"),
    [
        (
            "Change my travel dates to 10 November 2026 to 15 November 2026",
            {"start_date": "2026-11-10", "end_date": "2026-11-15"},
            ["start_date", "end_date"],
            2,
        ),
        ("Change my destination to Tokyo", {"destination": "Tokyo"}, ["destination"], 2),
        (
            "Add hiking to my activities",
            {"activities": ["sightseeing", "hiking"]},
            ["activities"],
            3,
        ),
        ("Change my style to sporty", {"style": ["sporty"]}, ["style"], 4),
    ],
)
def test_propose_refinement_validates_upstream_changes(
    message, updates, changed_fields, rerun_from_stage
):
    from travel_agent.web.student5_session import propose_refinement

    request = _student5_web_session_request()
    calls = []

    def understand(payload):
        calls.append(payload)
        proposed = {**request, **updates}
        proposed.pop("user_size")
        return _UnderstandingResult(
            {
                "state": {**payload["state"], "pending_request": proposed},
                "proposed_request": proposed,
                "questions": [{"field": "extraction_review"}],
            }
        )

    proposal = propose_refinement(_student5_web_session(), message, understand)

    if set(updates) == {"start_date", "end_date"}:
        assert calls == []
    else:
        assert calls[0]["message"] == message
        assert calls[0]["state"]["request_version"] == 4
        assert calls[0]["state"]["current_request"]["destination"] == "Seoul"
    assert proposal["patch"]["set"] == updates
    assert proposal["changed_fields"] == changed_fields
    assert proposal["rerun_plan"]["rerun_from_stage"] == rerun_from_stage
    assert proposal["updated_request"]["user_size"] == "M"
    assert proposal["student1_state"]["pending_request"] == proposal["student1_request"]


def test_propose_refinement_rejects_reversed_dates_before_student5():
    from travel_agent.web.student5_session import SessionError, propose_refinement

    request = _student5_web_session_request()

    def understand(payload):
        proposed = {
            **request,
            "start_date": "2026-11-20",
            "end_date": "2026-11-10",
        }
        proposed.pop("user_size")
        return _UnderstandingResult(
            {"state": payload["state"], "proposed_request": proposed, "questions": []}
        )

    with pytest.raises(SessionError, match="End date cannot precede start date"):
        propose_refinement(_student5_web_session(), "dates", understand)


def test_propose_refinement_requires_a_reviewable_extraction():
    from travel_agent.web.student5_session import SessionError, propose_refinement

    result = _UnderstandingResult(
        {
            "state": {},
            "validation_errors": [{"field": "input", "message": "ambiguous date"}],
            "questions": [{"field": "input", "question": "Please clarify"}],
        }
    )
    with pytest.raises(SessionError, match="Please clarify"):
        propose_refinement(
            _student5_web_session(),
            "change it",
            lambda payload: result,
        )


def test_propose_refinement_rejects_an_unchanged_request():
    from travel_agent.web.student5_session import SessionError, propose_refinement

    request = _student5_web_session_request()
    proposed = dict(request)
    proposed.pop("user_size")
    result = _UnderstandingResult(
        {"state": {}, "proposed_request": proposed, "questions": [{"field": "extraction_review"}]}
    )
    with pytest.raises(SessionError, match="did not change"):
        propose_refinement(
            _student5_web_session(),
            "keep it",
            lambda payload: result,
        )
