import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from travel_agent.schemas import (
    ConversationState,
    TripShoppingRequest,
    TripShoppingRequestPatch,
    WeatherContext,
)
from travel_agent.understanding.extraction import (
    check_money_grounding,
    ground_extraction,
    parse_json,
)
from travel_agent.understanding.knowledge import reason_activities, reason_places
from travel_agent.understanding.rules import reason_weather
from travel_agent.understanding.service import run
from travel_agent.understanding.state import apply_patch, confirm_outfit
from travel_agent.understanding.weather import FixtureWeather, OpenMeteoWeather

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def payload():
    return json.loads((ROOT / "examples/understanding/seoul_winter.json").read_text())


def test_proposal_appendix_request_and_reasoning_shape(payload):
    result = run(payload)
    assert result.status == "ok"
    data = result.data
    assert data["trip_request"]["activities"] == ["sightseeing", "dinner"]
    assert data["trip_request"]["budget"] == 500
    assert data["hard_constraints"]["budget_minor"] == 50000
    assert data["activity_style_requirements"]["dinner"] == ["smart casual", "polished"]
    assert "warm" in data["weather_clothing_requirements"]["properties"]
    assert "weather-resistant" in data["weather_clothing_requirements"]["properties"]
    assert isinstance(data["expanded_fashion_context"], str)
    assert data["enriched_trip_context"]["weather_context"]["data_type"] == "synthetic_fixture"
    assert data["context_status"] == "partial"
    assert data["feasibility_status"] == "unverified"
    assert any(r["review_status"] == "pending" for r in data["weather_rule_trace"])


def test_missing_information_is_preserved_then_patch_completes(payload):
    partial = {"destination": "Seoul", "activities": ["sightseeing"]}
    first = run({"request": partial})
    assert first.status == "needs_input"
    assert first.data["trip_request"]["currency"] is None
    assert {q["field"] for q in first.data["questions"]} == {
        "start_date",
        "end_date",
        "budget",
        "currency",
    }
    second = run(
        {
            "state": first.data["state"],
            "patch": {
                "set": {
                    "start_date": "2026-12-20",
                    "end_date": "2026-12-27",
                    "budget": 500,
                    "currency": "SGD",
                }
            },
            "weather_fixture": payload["weather_fixture"],
            "allow_draft": True,
        }
    )
    assert second.status == "ok"
    assert second.data["trip_request"]["destination"] == "Seoul"
    assert second.data["trip_request"]["activities"] == ["sightseeing"]
    assert second.data["state"]["request_version"] == 2


@pytest.mark.parametrize("budget", [True, 0, -1, "500", float("nan"), float("inf")])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValidationError):
        TripShoppingRequest(budget=budget)


def test_currency_precision_and_no_default_currency():
    assert TripShoppingRequest(budget=500).currency is None
    assert TripShoppingRequest(budget=12.34, currency="SGD").budget_minor_units() == 1234
    with pytest.raises(ValidationError):
        TripShoppingRequest(budget=12.345, currency="SGD")
    with pytest.raises(ValidationError):
        TripShoppingRequest(budget=12.5, currency="JPY")
    with pytest.raises(ValidationError):
        TripShoppingRequest(currency="XYZ")


@pytest.mark.parametrize(
    "dates",
    [
        {"start_date": "2026-02-30"},
        {"start_date": 123456},
        {"start_date": "2026-12-27", "end_date": "2026-12-20"},
    ],
)
def test_dates_need_correction(dates):
    assert run({"request": dates}).status == "needs_input"


def test_patch_preserves_facts_cheaper_is_not_new_budget(payload):
    request = TripShoppingRequest.model_validate(payload["request"])
    patch = TripShoppingRequestPatch(
        set={"price_preference": "lower"}, add={"excluded_items": ["long coat"]}
    )
    updated = apply_patch(request, patch)
    assert updated.budget == request.budget
    assert updated.destination == request.destination
    assert updated.activities == request.activities
    assert updated.excluded_items == ["long coat"]
    assert request.excluded_items == []
    removed = apply_patch(updated, TripShoppingRequestPatch(remove={"style": []}))
    assert removed.style == []
    assert removed.budget == 500


def test_invalid_patch_is_atomic_and_does_not_discard_state(payload):
    first = run(payload)
    state = first.data["state"]
    edited = run({"state": state, "patch": {"set": {"end_date": "2020-01-01"}}})
    assert edited.status == "needs_input"
    assert edited.data["state"] == state
    with pytest.raises(ValidationError):
        TripShoppingRequestPatch(set={"destination": "Tokyo"}, remove={"destination": None})
    with pytest.raises(ValidationError):
        TripShoppingRequestPatch(add={"budget": ["500"]})
    with pytest.raises(ValidationError):
        TripShoppingRequestPatch(set={"invented_field": True})


def test_confirmation_uses_latest_round_and_changes_invalidate_old_outfits(payload):
    state = ConversationState(
        current_request=TripShoppingRequest.model_validate(payload["request"]),
        request_version=1,
        recommendation_round=2,
        last_recommended_outfit_ids=["a", "b", "c"],
        session_status="AWAITING_CONFIRMATION",
    )
    assert confirm_outfit(state, display_position=2).selected_outfit_id == "b"
    with pytest.raises(ValueError):
        confirm_outfit(state, outfit_id="old_outfit")
    with pytest.raises(ValueError):
        confirm_outfit(state, outfit_id="a", display_position=2)
    changed = run({"state": state.model_dump(mode="json"), "patch": {"set": {"budget": 400}}})
    assert changed.data["state"]["last_recommended_outfit_ids"] == []
    assert changed.data["state"]["selected_outfit_id"] is None
    assert changed.data["state"]["recommendation_round"] == 2


def test_review_policy_one_hop_multi_hop_and_unknown_activity(payload):
    from travel_agent.understanding.activity_engine import KnowledgeGraph

    graph = KnowledgeGraph.load()
    for edge in graph.data["edges"]:
        edge.update(review_status="pending", reviewed_by=None)
    request = TripShoppingRequest(activities=["hiking", "scuba diving"])
    strict, results, _ = reason_activities(request, graph=graph)
    assert strict["hiking"] == []
    one, r1, _ = reason_activities(request, allow_draft=True, max_depth=1, graph=graph)
    multi, r2, _ = reason_activities(request, allow_draft=True, graph=graph)
    assert set(one["hiking"]) <= set(multi["hiking"])
    assert all(len(p["edge_ids"]) == 1 for c in r1["hiking"]["conclusions"] for p in c["paths"])
    assert any(len(p["edge_ids"]) > 1 for c in r2["hiking"]["conclusions"] for p in c["paths"])
    assert r1["hiking"]["knowledge_status"] == "draft"
    assert r2["scuba diving"]["status"] == "unknown_activity"
    assert request.activities == ["hiking", "scuba diving"]
    assert multi["scuba diving"] == []
    assert any(
        path["evidence"]
        for conclusion in r2["hiking"]["conclusions"]
        for path in conclusion["paths"]
    )
    request = TripShoppingRequest(activities=["dinner"])
    assert reason_activities(request, allow_draft=True)[2][0]["field"] == "dress_codes.dinner"


def test_unavailable_weather_does_not_suppress_activity_reasoning(payload):
    payload.pop("weather_fixture")
    result = run(payload)
    assert result.status == "ok"
    assert result.data["enriched_trip_context"]["weather_context"]["status"] == "unknown"
    assert not any(result.data["weather_clothing_requirements"].values())
    assert result.data["activity_style_requirements"]["sightseeing"]
    assert result.data["context_status"] == "partial"


@pytest.mark.parametrize("temp,expected", [(9.9, True), (10, False), (10.1, False)])
def test_weather_rule_boundary(payload, temp, expected):
    raw = payload["weather_fixture"] | {
        "temp_min_celsius": temp,
        "temp_max_celsius": 20,
        "snow": None,
    }
    _, output, trace = reason_weather(WeatherContext.model_validate(raw))
    assert ("warm" in output.properties) == expected
    assert any(r["rule_id"] == "COLD" for r in trace) == expected
    assert not any(r["rule_id"] == "SNOW" for r in trace)


def test_fixture_requires_exact_scope_and_truthful_type(payload):
    provider = FixtureWeather(payload["weather_fixture"])
    request = TripShoppingRequest.model_validate(payload["request"])
    request.destination = "Tokyo"
    assert provider.fetch(request).status == "unknown"
    with pytest.raises(ValueError):
        FixtureWeather(payload["weather_fixture"] | {"data_type": "forecast"})


def test_live_provider_out_of_range_and_ambiguous_location():
    request = TripShoppingRequest(
        destination="Tokyo", start_date="2026-12-20", end_date="2026-12-27"
    )

    def no_network(url):
        raise AssertionError("Out-of-range request must not call the API")

    provider = OpenMeteoWeather(transport=no_network, today=date(2026, 10, 11))
    assert provider.fetch(request).status == "out_of_range"
    request.start_date, request.end_date = date(2026, 10, 12), date(2026, 10, 13)
    provider.transport = lambda url: {"results": [{"name": "Tokyo"}, {"name": "Other Tokyo"}]}
    assert provider.fetch(request).status == "needs_location"


def test_api_units_date_coverage_probability_and_missing_values():
    request = TripShoppingRequest(
        destination="Tokyo", start_date="2026-10-12", end_date="2026-10-13"
    )
    response = {
        "daily_units": {
            "temperature_2m_min": "°C",
            "temperature_2m_max": "°C",
            "precipitation_probability_max": "%",
            "snowfall_sum": "cm",
        },
        "daily": {
            "time": ["2026-10-12", "2026-10-13"],
            "temperature_2m_min": [12, 11],
            "temperature_2m_max": [20, 21],
            "precipitation_probability_max": [35, 60],
            "snowfall_sum": [0, 0],
        },
    }
    provider = OpenMeteoWeather(
        location={"destination": "Tokyo", "latitude": 35.68, "longitude": 139.69},
        transport=lambda url: response,
        today=date(2026, 10, 11),
    )
    weather = provider.fetch(request)
    assert weather.status == "ok"
    assert weather.precipitation_probability == 0.6
    assert weather.temp_min_celsius == 11
    assert weather.data_type == "forecast"
    response["daily"]["temperature_2m_min"][0] = None
    assert provider.fetch(request).status == "unknown"
    response["daily"]["temperature_2m_min"][0] = 12
    response["daily_units"]["temperature_2m_min"] = "°F"
    assert provider.fetch(request).status == "unknown"
    response["daily_units"]["temperature_2m_min"] = "°C"
    response["daily"]["time"] = ["2026-10-12"]
    assert provider.fetch(request).status == "unknown"


def test_explicit_poi_rules_do_not_mix_in_static_climate():
    request = TripShoppingRequest(places=["上野公园"], activities=["sightseeing"])
    result = reason_places(request, allow_draft=True)
    assert all(not p["climate_profiles"] for p in result["places"])
    assert "comfy_walking_shoes" in [r["req_id"] for r in result["requirements"]]
    assert "warm_layer" not in [r["req_id"] for r in result["requirements"]]


def test_json_and_currency_grounding():
    assert parse_json('```json\n{"budget":500}\n```') == {"budget": 500}
    with pytest.raises(ValueError):
        parse_json("[1,2]")
    with pytest.raises(ValueError):
        check_money_grounding({"budget": 400}, "Make it cheaper")
    with pytest.raises(ValueError):
        check_money_grounding({"currency": "SGD"}, "Budget 500, visiting Singapore")
    check_money_grounding({"budget": 500, "currency": "SGD"}, "预算500新币")


def test_model_adapter_orchestration_without_model_dependency(payload):
    class FakeExtractor:
        def extract(self, message, current_request):
            assert current_request is None
            return payload["request"], {"provider": "test_double"}

    result = run({"message": "test", "allow_draft": True}, extractor=FakeExtractor())
    assert result.status == "needs_input"
    assert result.data["state"]["request_version"] == 0
    assert result.data["proposed_request"]["destination"] == "Seoul"
    accepted = run(
        {"message": "test", "allow_draft": True, "accept_extraction": True},
        extractor=FakeExtractor(),
    )
    assert accepted.status == "ok"
    assert accepted.data["extraction"]["provider"] == "test_double"


def test_confirmed_coordinates_do_not_survive_destination_change():
    request = TripShoppingRequest(
        destination="Seoul", start_date="2026-10-12", end_date="2026-10-13"
    )
    provider = OpenMeteoWeather(
        location={"destination": "Tokyo", "latitude": 35.68, "longitude": 139.69},
        today=date(2026, 10, 11),
    )
    assert provider.fetch(request).status == "needs_location"


def test_agent_tool_cannot_approve_its_own_natural_language_extraction(monkeypatch):
    from travel_agent.understanding import tools

    captured = {}

    def fake_run(payload):
        captured.update(payload)
        from travel_agent.schemas import ModuleResult

        return ModuleResult(status="needs_input")

    monkeypatch.setattr(tools, "run", fake_run)
    tools.understand_and_reason.invoke({"payload": {"message": "test", "accept_extraction": True}})
    assert captured["accept_extraction"] is False


@pytest.mark.parametrize(
    "candidate",
    [
        {"activities": [1]},
        {"style": [None]},
        {"destination": []},
        {"dress_codes": {"dinner": []}},
    ],
)
def test_malformed_model_value_types_fail_as_validation(candidate):
    with pytest.raises(ValueError):
        ground_extraction(candidate, "test")


def test_grounding_drops_hallucinations_and_normalizes_explicit_chinese():
    parsed, warnings = ground_extraction(
        {
            "destination": "首尔",
            "activities": ["观光", "dinner"],
            "style": ["极简风", "sporty"],
            "currency": "USD",
            "budget": 400,
            "start_date": "2026-12-20",
            "price_preference": "lower",
            "dress_codes": {"dinner": "smart casual"},
        },
        "去首尔观光，喜欢极简风。",
    )
    assert parsed["destination"] == "Seoul"
    assert parsed["activities"] == ["sightseeing"]
    assert parsed["style"] == ["minimalist"]
    assert "currency" not in parsed and "budget" not in parsed
    assert "start_date" not in parsed and "price_preference" not in parsed
    assert parsed["dress_codes"] == {}
    assert warnings


def test_neo4j_adapter_decodes_seed_properties_and_checks_retained_paths(monkeypatch):
    import sys
    from types import SimpleNamespace

    from travel_agent.understanding.activity_engine import KnowledgeGraph
    from travel_agent.understanding.neo4j_backend import Neo4jKnowledgeGraph

    original = KnowledgeGraph.load()
    expected = original.query("sightseeing", allow_draft=True)
    paths = [{"edge_ids": p["edge_ids"]} for c in expected["conclusions"] for p in c["paths"]]
    nodes = [
        {"props": {k: v for k, v in n.items() if k != "type"}, "labels": ["Student1KG", n["type"]]}
        for n in original.data["nodes"]
    ]
    edges = [
        {
            "source": e["source"],
            "target": e["target"],
            "relation": e["relation"],
            "props": {
                **{
                    k: v
                    for k, v in e.items()
                    if k not in {"source", "target", "relation", "conditions"}
                },
                "conditions_json": json.dumps(e["conditions"]),
                "source_record_json": json.dumps(original.sources[e["source_id"]]),
            },
        }
        for e in original.data["edges"]
    ]

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def run(self, query, **params):
            assert params.get("version") == original.data["version"]
            data = paths if "MATCH path=" in query else (nodes if "labels(n)" in query else edges)
            return SimpleNamespace(data=lambda: data)

    class FakeDriver:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def session(self, **kwargs):
            assert kwargs["default_access_mode"] == "READ"
            return FakeSession()

    monkeypatch.setenv("NEO4J_URI", "bolt://test.invalid")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-only")
    monkeypatch.setitem(
        sys.modules,
        "neo4j",
        SimpleNamespace(GraphDatabase=SimpleNamespace(driver=lambda *args, **kwargs: FakeDriver())),
    )
    graph = Neo4jKnowledgeGraph.from_environment()
    result = graph.query("sightseeing", allow_draft=True)
    for key in ("styles", "attributes", "conclusions", "knowledge_status"):
        assert result[key] == expected[key]
    assert result["path_query_verified"] is True
    paths.clear()
    with pytest.raises(ValueError, match="changed or traversal is incomplete"):
        graph.query("sightseeing", allow_draft=True)
