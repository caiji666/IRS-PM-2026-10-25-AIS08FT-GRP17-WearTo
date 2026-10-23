"""Displayed item references must exclude exact products from every new outfit."""

import copy
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from travel_agent.recommendation.service import run
from travel_agent.web import __main__ as web
from travel_agent.web.student5_session import SessionError, patch_from_message, refine


@pytest.fixture
def session():
    request = {
        "destination": "Seoul",
        "start_date": "2026-11-11",
        "end_date": "2026-11-26",
        "budget": "200",
        "currency": "SGD",
        "activities": [],
        "style": [],
        "excluded_items": [],
    }
    outfits = []
    for index in range(7):
        items = []
        for category in ("TOP", "BOTTOM", "FOOTWEAR"):
            # The top two candidates share shoes: excluding one displayed product
            # must remove all combinations containing it, not just outfit 1.
            product_id = f"{category}-{0 if category == 'FOOTWEAR' and index == 1 else index}"
            items.append(
                {
                    "product_id": product_id,
                    "category": category,
                    "title": product_id,
                    "source": "mock_contract_fixture",
                    "price_minor": 1000,
                    "currency": "SGD",
                    "availability": True,
                    "is_user_owned": False,
                    "product_score": 0.8,
                    "product_url": f"https://shop.example.com/{product_id}",
                }
            )
        outfits.append(
            {
                "outfit_id": f"outfit-{index}",
                "compatibility_score": 0.99 - index * 0.05,
                "items": items,
            }
        )
    student4 = {"status": "ok", "source": "mock_contract_fixture", "outfits": outfits}
    result = run({"action": "recommend", "request": request, "student4": student4})
    assert result.status == "ok", result.reason
    data = result.model_dump(mode="json")["data"]
    return {
        "request": request,
        "student4": student4,
        "state": data["state"],
        "recommendation_set": data["recommendation_set"],
        "confirmed_outfit": None,
        "preview": None,
    }


def shoes(session, position=0):
    outfit = session["recommendation_set"]["recommendations"][position]
    return next(
        item["product_id"] for item in outfit["purchase_items"] if item["category"] == "FOOTWEAR"
    )


def returned_ids(session):
    return {
        item["product_id"]
        for outfit in session["recommendation_set"]["recommendations"]
        for item in outfit["purchase_items"] + outfit["existing_items"]
    }


@pytest.mark.parametrize(
    "message",
    [
        "I don't like the shoes from outfit 1",
        "I don’t like the shoes from outfit 1",
        "Replace the boots in the first outfit",
        "Remove shoes from outfit #1",
    ],
)
def test_exact_displayed_shoes_are_resolved_and_excluded_across_combinations(session, message):
    before = copy.deepcopy(session)
    unwanted = shoes(session)
    patch = patch_from_message(message, session)
    assert patch["add"]["excluded_items"] == [unwanted]
    updated = refine(session, patch, run)
    assert len(updated["recommendation_set"]["recommendations"]) == 3
    assert unwanted not in returned_ids(updated)
    assert updated["student4"] == before["student4"]
    assert session == before


def test_exclusions_accumulate_and_survive_later_budget_changes(session):
    first = shoes(session)
    updated = refine(session, patch_from_message("Replace shoes from outfit 1", session), run)
    second = shoes(updated)
    updated = refine(updated, patch_from_message("Replace shoes from outfit 1", updated), run)
    updated = refine(updated, patch_from_message("Change my budget to 100", updated), run)
    assert set(updated["request"]["excluded_items"]) == {first, second}
    assert not ({first, second} & returned_ids(updated))


@pytest.mark.parametrize(
    "message",
    [
        "I don't like the shoes",
        "Remove shoes from outfit 9",
        "Remove the coat from outfit 1",
        "Replace outfit 1",
        "Remove shoes from outfits 1 and 2",
        "Don't remove shoes from outfit 1",
    ],
)
def test_ambiguous_or_missing_references_leave_recommendations_unchanged(session, message):
    before = copy.deepcopy(session)
    with pytest.raises(SessionError):
        patch_from_message(message, session)
    assert session == before


def test_http_item_dislike_stays_in_student5_and_returns_new_top3(session, monkeypatch):
    session_id = "b" * 32
    monkeypatch.setattr(web, "STUDENT5_SESSIONS", {session_id: session})

    def no_upstream(*args, **kwargs):
        raise AssertionError("An exact displayed item must not rerun trip extraction")

    monkeypatch.setattr("travel_agent.understanding.service.run", no_upstream)
    server = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
    unwanted = shoes(session)
    try:
        connection.request(
            "POST",
            "/api/student5/refine",
            json.dumps(
                {
                    "id": session_id,
                    "message": "I don't like the shoes from outfit 1",
                }
            ),
            {"X-Trip-UI": "1", "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        data = json.loads(response.read())
        assert response.status == 200, data
        assert len(data["top3"]) == 3
        assert unwanted not in {
            item["product_id"] for outfit in data["top3"] for item in outfit["purchase_items"]
        }
        assert web.STUDENT5_SESSIONS[session_id]["request"]["excluded_items"] == [unwanted]
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
