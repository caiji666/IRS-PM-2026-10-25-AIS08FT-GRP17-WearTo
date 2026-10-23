"""Alternative aesthetics, independent needs, conditions and graph alignment."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from travel_agent.schemas import TripShoppingRequest
from travel_agent.understanding.activity_engine import KnowledgeGraph
from travel_agent.understanding.english_evidence import grounded_english
from travel_agent.understanding.knowledge import reason_places
from travel_agent.understanding.service import run


def test_opposite_aesthetics_are_alternatives_not_shared_obligations():
    result = KnowledgeGraph.load().query("shopping", allow_draft=True)
    candidates = {c["style"]: c for c in result["style_candidates"]}
    assert {"minimalist", "maximalist"} <= set(candidates)
    assert "clean_lines" in candidates["minimalist"]["attributes"]
    assert "expressive_patterns" in candidates["maximalist"]["attributes"]
    assert "expressive_patterns" not in candidates["minimalist"]["attributes"]
    assert "clean_lines" not in result["attributes"]
    assert "expressive_patterns" not in {r["value"] for r in result["dressing_requirements"]}


def test_preference_keeps_function_and_does_not_invent_styles():
    result = KnowledgeGraph.load().query(
        "outdoor_trek", allow_draft=True, preferred_styles=["极简", "unregistered"]
    )
    assert result["preference_match"] == {"matched": ["极简"], "unmatched": ["unregistered"]}
    assert {"traction", "freedom_of_movement", "moisture_wicking"} <= {
        r["value"] for r in result["dressing_requirements"]
    }
    assert "unregistered" not in result["styles"]
    assert all(r["strength"] != "must" for r in result["dressing_requirements"])


def test_explicit_formality_and_missing_context():
    graph = KnowledgeGraph.load()
    missing = graph.query("wedding", allow_draft=True)
    assert missing["status"] == "needs_clarification" and missing["style_candidates"] == []
    formal = graph.query("wedding", {"dress_code": "Black-Tie"}, allow_draft=True)
    assert {"black_tie", "minimalist", "romantic"} <= set(formal["styles"])
    assert not {"streetwear", "sporty", "casual"} & set(formal["styles"])
    creative = graph.query("wedding", {"dress_code": "creative black tie"}, allow_draft=True)
    assert "artistic" in creative["styles"] and "creative_accent" in creative["attributes"]
    smart = graph.query("dinner", {"dress_code": "smart casual"}, allow_draft=True)
    dressy = graph.query("dinner", {"dress_code": "dressy casual"}, allow_draft=True)
    assert "smart_casual" in smart["styles"] and "dressy_casual" not in smart["styles"]
    assert "dressy_casual" in dressy["styles"] and "smart_casual" not in dressy["styles"]


@pytest.mark.parametrize(
    "activity",
    [
        "city_walk",
        "outdoor_trek",
        "museum_indoor",
        "theme_park",
        "religious_visit",
        "beach",
        "photo_spot",
        "food_market",
        "shopping",
        "aquarium_visit",
        "nightlife_fine_dining",
    ],
)
def test_every_place_activity_resolves(activity):
    context = {"dress_code": "casual"} if activity == "nightlife_fine_dining" else {}
    result = KnowledgeGraph.load().query(activity, context, allow_draft=True)
    assert result["status"] == "ok" and result["place_activity_id"] == activity
    assert result["style_candidates"]
    assert all(p["evidence"] for c in result["conclusions"] for p in c["paths"])


def test_place_bridge_and_no_universal_religious_or_swimming_rules():
    result = reason_places(
        TripShoppingRequest(places=["东京国立博物馆"], activities=["museum_visit"]),
        allow_draft=True,
    )
    assert result["places"][0]["selected_activities"] == ["museum_indoor"]
    graph = KnowledgeGraph.load()
    religious = graph.query("religious_visit", allow_draft=True)
    assert "verify_venue_dress_rules" in religious["attributes"]
    assert not {"cover_shoulders", "no_shoes_indoor"} & set(religious["attributes"])
    beach = graph.query("beach", allow_draft=True)
    assert "verify_actual_water_activity" in beach["attributes"]
    assert not any("swim" in x for x in beach["attributes"])


def test_review_filter_and_visual_descriptor_validation():
    graph = KnowledgeGraph.load()
    pending = copy.deepcopy(graph.data)
    for edge in pending["edges"]:
        edge.update(review_status="pending", reviewed_by=None)
    assert KnowledgeGraph(pending).query("shopping")["style_candidates"] == []
    data = copy.deepcopy(graph.data)
    for edge in data["edges"]:
        if edge["source"] == "activity:shopping" and edge["target"] == "style:maximalist":
            edge["review_status"] = "rejected"
    assert "maximalist" not in KnowledgeGraph(data).query("shopping", allow_draft=True)["styles"]
    data = copy.deepcopy(graph.data)
    next(e for e in data["edges"] if e["relation"] == "NEEDS_ATTRIBUTE")["target"] = (
        "attribute:clean_lines"
    )
    with pytest.raises(ValueError, match="Visual descriptor"):
        KnowledgeGraph(data)
    for preferred in ["minimalist", [None], [""]]:
        with pytest.raises(ValueError, match="preferred_styles"):
            graph.query("hiking", preferred_styles=preferred)


def test_service_options_are_separate_from_needs_and_items():
    result = run(
        {
            "request": {
                "destination": "Singapore",
                "start_date": "2026-10-20",
                "end_date": "2026-10-21",
                "activities": ["shopping"],
                "budget": 500,
                "currency": "SGD",
                "style": ["极简"],
            },
            "allow_draft": True,
        }
    )
    assert result.status == "ok"
    options = result.data["activity_style_options"]["shopping"]
    assert next(c for c in options if c["style"] == "minimalist")["matched_preferences"] == ["极简"]
    assert "minimalist" not in result.data["activity_style_requirements"]["shopping"]
    assert result.data["activity_dressing_requirements"]["shopping"]
    assert all("items" not in c for c in options)


def test_snapshot_fingerprints():
    base = Path(__file__).resolve().parents[1] / "travel_agent/understanding/kg_data"
    manifest = json.loads((base / "manifest.json").read_text())
    assert manifest["activity_graph_version"] == "0.2.0"
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == expected


def test_new_style_words_are_explicit_preferences_not_activity_inferences():
    result, _ = grounded_english({}, "I prefer preppy and romantic style.")
    assert set(result["style"]) == {"preppy", "romantic"}
    result, _ = grounded_english({}, "I will visit a museum.")
    assert "artistic" not in result.get("style", [])
