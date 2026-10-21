"""Local UI input boundaries and private-artifact isolation; no model/API calls."""

import http.client
import json
import threading
from datetime import datetime, timedelta
from http.server import ThreadingHTTPServer
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
    assert (
        web.validate_form(
            {
                **form(),
                "activities": [],
                "style": "",
                "start_date": "2026-12-20",
                "end_date": "2026-12-27",
            }
        )["activities"]
        == []
    )
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
