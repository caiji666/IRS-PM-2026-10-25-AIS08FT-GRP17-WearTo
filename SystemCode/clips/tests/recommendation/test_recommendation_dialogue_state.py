"""Tests for the Conversational Recommendation State (Module 6, dialogue_state)."""

import pytest

from travel_agent.recommendation.student5.dialogue_state import (
    InvalidOutfitSelectionError,
    StaleRecommendationSetError,
    create_initial_state,
    modify_requirements,
    record_recommendation_set,
    resolve_current_recommendation,
)
from travel_agent.recommendation.student5.models import (
    RecommendationSet,
    SessionStatus,
    TripShoppingRequest,
    TripShoppingRequestPatch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_request(**overrides):
    data = {
        "destination": "Seoul",
        "start_date": "2026-12-20",
        "end_date": "2026-12-27",
        "activities": ["sightseeing", "dinner"],
        "budget": 500,
        "currency": "SGD",
    }
    data.update(overrides)
    return TripShoppingRequest.model_validate(data)


def product(category, price):
    return {
        "product_id": f"p_{category.lower()}",
        "source": "mock",
        "title": f"Mock {category}",
        "product_url": f"https://example.com/{category.lower()}",
        "price": price,
        "currency": "SGD",
        "availability": True,
        "category": category,
        "product_score": 0.5,
    }


def make_set(recommendation_round, outfit_ids):
    recs = [
        {
            "outfit_id": oid,
            "display_position": pos,
            "existing_items": [],
            "purchase_items": [
                product("TOP", "40.00"),
                product("BOTTOM", "80.00"),
                product("FOOTWEAR", "60.00"),
            ],
            "total_purchase_cost": "180.00",
            "currency": "SGD",
            "compatibility_score": 0.75,
            "outfit_product_relevance": 0.5,
            "final_outfit_score": 0.9 - 0.1 * pos,
        }
        for pos, oid in enumerate(outfit_ids, start=1)
    ]
    return RecommendationSet.model_validate(
        {"recommendation_round": recommendation_round, "recommendations": recs}
    )


def modify(**operations):
    return TripShoppingRequestPatch.model_validate({"intent": "MODIFY_REQUIREMENTS", **operations})


ROUND1 = ["outfit_101", "outfit_102", "outfit_103"]
ROUND2 = ["outfit_201", "outfit_202", "outfit_203"]


@pytest.fixture
def initial():
    return create_initial_state("session_001", make_request())


@pytest.fixture
def round1(initial):
    return record_recommendation_set(initial, make_set(1, ROUND1))


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_initial_values(self, initial):
        assert initial.session_id == "session_001"
        assert initial.request_version == 1
        assert initial.recommendation_round == 0
        assert initial.last_recommended_outfit_ids == []
        assert initial.selected_outfit_id is None
        assert initial.session_status is SessionStatus.AWAITING_CONFIRMATION


# ---------------------------------------------------------------------------
# MODIFY_REQUIREMENTS
# ---------------------------------------------------------------------------


class TestModifyRequirements:
    @pytest.mark.parametrize(
        "operations",
        [
            {"set": {"budget": 400}},
            {"add": {"excluded_items": ["long coat"]}},
        ],
        ids=["budget", "excluded_items"],
    )
    def test_real_change_invalidates_current_recommendations(self, round1, operations):
        state, result = modify_requirements(round1, modify(**operations))
        assert result.changed is True
        assert state.last_recommended_outfit_ids == []
        assert state.recommendation_round == 1  # only a new recorded set advances the round

    def test_request_version_increments_only_on_real_change(self, round1):
        same, _ = modify_requirements(round1, modify(set={"budget": 500}))
        assert same.request_version == 1
        changed, _ = modify_requirements(same, modify(set={"budget": 450}))
        assert changed.request_version == 2
        changed_again, _ = modify_requirements(changed, modify(set={"destination": "Busan"}))
        assert changed_again.request_version == 3

    def test_old_set_cannot_be_resolved_after_change(self, round1):
        state, _ = modify_requirements(round1, modify(set={"budget": 400}))
        with pytest.raises(StaleRecommendationSetError):
            resolve_current_recommendation(state, make_set(1, ROUND1), "outfit_101")


# ---------------------------------------------------------------------------
# Recording a new RecommendationSet
# ---------------------------------------------------------------------------


class TestRecordRecommendationSet:
    def test_round_incremented_and_ids_saved_in_display_order(self, initial):
        state = record_recommendation_set(initial, make_set(1, ["z_last", "a_first", "m_mid"]))
        assert state.recommendation_round == 1
        assert state.last_recommended_outfit_ids == ["z_last", "a_first", "m_mid"]
        assert state.session_status is SessionStatus.AWAITING_CONFIRMATION
        assert state.selected_outfit_id is None

    @pytest.mark.parametrize("wrong_round", [1, 3])
    def test_set_round_must_be_next_round(self, round1, wrong_round):
        with pytest.raises(ValueError, match="recommendation_round"):
            record_recommendation_set(round1, make_set(wrong_round, ROUND2))


# ---------------------------------------------------------------------------
# Resolving an outfit against the current recommendations
# ---------------------------------------------------------------------------


class TestResolveCurrentRecommendation:
    def test_unknown_id_rejected(self, round1):
        with pytest.raises(InvalidOutfitSelectionError, match="outfit_999"):
            resolve_current_recommendation(round1, make_set(1, ROUND1), "outfit_999")

    @pytest.mark.parametrize("bad", [2, ""])
    def test_display_position_or_blank_not_accepted_as_id(self, round1, bad):
        with pytest.raises(InvalidOutfitSelectionError):
            resolve_current_recommendation(round1, make_set(1, ROUND1), bad)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Full multi-round flow
# ---------------------------------------------------------------------------
