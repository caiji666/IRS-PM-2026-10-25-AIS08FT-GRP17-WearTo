"""Tests for CONFIRM_OUTFIT handling (Module 6, confirmation_handler)."""

import pytest

from travel_agent.recommendation.student5.confirmation_handler import (
    OutfitAlreadyConfirmedError,
    confirm_outfit,
)
from travel_agent.recommendation.student5.dialogue_state import (
    InvalidOutfitSelectionError,
    StaleRecommendationSetError,
    create_initial_state,
    record_recommendation_set,
)
from travel_agent.recommendation.student5.models import (
    RecommendationSet,
    SessionStatus,
    TripShoppingRequest,
    TripShoppingRequestPatch,
)


def make_request():
    return TripShoppingRequest.model_validate(
        {
            "destination": "Seoul",
            "start_date": "2026-12-20",
            "end_date": "2026-12-27",
            "budget": 500,
            "currency": "SGD",
        }
    )


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


ROUND1 = ["outfit_101", "outfit_102", "outfit_103"]
ROUND2 = ["outfit_201", "outfit_202", "outfit_203"]


@pytest.fixture
def set1():
    return make_set(1, ROUND1)


@pytest.fixture
def state(set1):
    return record_recommendation_set(create_initial_state("session_001", make_request()), set1)


class TestConfirmOutfit:
    def test_valid_id_confirmed(self, state, set1):
        confirmed = confirm_outfit(state, set1, "outfit_102")
        assert confirmed.selected_outfit_id == "outfit_102"
        assert confirmed.session_status is SessionStatus.CONFIRMED

    def test_unknown_id_rejected(self, state, set1):
        with pytest.raises(InvalidOutfitSelectionError, match="outfit_404"):
            confirm_outfit(state, set1, "outfit_404")

    def test_previous_round_id_rejected(self, state, set1):
        state2 = record_recommendation_set(state, make_set(2, ROUND2))
        with pytest.raises(InvalidOutfitSelectionError):
            confirm_outfit(state2, make_set(2, ROUND2), "outfit_101")
        with pytest.raises(StaleRecommendationSetError):
            confirm_outfit(state2, set1, "outfit_101")

    @pytest.mark.parametrize("position", [2])
    def test_display_position_not_treated_as_id(self, state, set1, position):
        with pytest.raises(InvalidOutfitSelectionError):
            confirm_outfit(state, set1, position)

    def test_same_id_confirmed_again_is_idempotent(self, state, set1):
        once = confirm_outfit(state, set1, "outfit_102")
        twice = confirm_outfit(once, set1, "outfit_102")
        assert twice == once
        assert twice is not once

    def test_switching_after_confirmation_rejected(self, state, set1):
        confirmed = confirm_outfit(state, set1, "outfit_102")
        with pytest.raises(OutfitAlreadyConfirmedError, match="outfit_102"):
            confirm_outfit(confirmed, set1, "outfit_101")
        assert issubclass(OutfitAlreadyConfirmedError, ValueError)


def budget_patch(budget):
    return TripShoppingRequestPatch.model_validate(
        {"intent": "MODIFY_REQUIREMENTS", "set": {"budget": budget}}
    )
