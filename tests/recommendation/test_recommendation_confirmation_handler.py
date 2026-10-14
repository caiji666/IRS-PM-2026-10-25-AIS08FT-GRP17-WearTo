"""Tests for CONFIRM_OUTFIT handling (Module 6, confirmation_handler)."""

import pytest

from travel_agent.recommendation.student5.confirmation_handler import (
    OutfitAlreadyConfirmedError,
    build_confirmed_outfit,
    confirm_outfit,
)
from travel_agent.recommendation.student5.dialogue_state import (
    InvalidOutfitSelectionError,
    StaleRecommendationSetError,
    create_initial_state,
    modify_requirements,
    record_recommendation_set,
)
from travel_agent.recommendation.student5.models import (
    ConfirmedOutfit,
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

    def test_other_fields_unchanged(self, state, set1):
        confirmed = confirm_outfit(state, set1, "outfit_103")
        for field in (
            "session_id",
            "request_version",
            "recommendation_round",
            "current_request",
            "last_recommended_outfit_ids",
        ):
            assert getattr(confirmed, field) == getattr(state, field), field

    def test_unknown_id_rejected(self, state, set1):
        with pytest.raises(InvalidOutfitSelectionError, match="outfit_404"):
            confirm_outfit(state, set1, "outfit_404")

    def test_previous_round_id_rejected(self, state, set1):
        state2 = record_recommendation_set(state, make_set(2, ROUND2))
        with pytest.raises(InvalidOutfitSelectionError):
            confirm_outfit(state2, make_set(2, ROUND2), "outfit_101")
        with pytest.raises(StaleRecommendationSetError):
            confirm_outfit(state2, set1, "outfit_101")

    @pytest.mark.parametrize("position", [2, "2", "#2"])
    def test_display_position_not_treated_as_id(self, state, set1, position):
        with pytest.raises(InvalidOutfitSelectionError):
            confirm_outfit(state, set1, position)

    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_blank_id_rejected(self, state, set1, bad):
        with pytest.raises(InvalidOutfitSelectionError):
            confirm_outfit(state, set1, bad)

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

    def test_inputs_not_modified(self, state, set1):
        state_snapshot, set_snapshot = state.model_dump(), set1.model_dump()
        confirmed = confirm_outfit(state, set1, "outfit_101")
        assert state.model_dump() == state_snapshot
        assert set1.model_dump() == set_snapshot
        assert state.selected_outfit_id is None
        assert confirmed is not state


def budget_patch(budget):
    return TripShoppingRequestPatch.model_validate(
        {"intent": "MODIFY_REQUIREMENTS", "set": {"budget": budget}}
    )


class TestAfterRequirementChange:
    def test_old_outfit_cannot_be_confirmed(self, state, set1):
        modified, _ = modify_requirements(state, budget_patch(90))
        with pytest.raises(StaleRecommendationSetError):
            confirm_outfit(modified, set1, "outfit_101")

    def test_previous_confirmation_cannot_be_rebuilt(self, state, set1):
        confirmed = confirm_outfit(state, set1, "outfit_102")
        modified, _ = modify_requirements(confirmed, budget_patch(90))
        with pytest.raises(StaleRecommendationSetError):
            build_confirmed_outfit(modified, set1)

    def test_new_round_can_be_confirmed(self, state, set1):
        modified, _ = modify_requirements(state, budget_patch(400))
        set2 = make_set(2, ROUND2)
        state2 = record_recommendation_set(modified, set2)
        confirmed = confirm_outfit(state2, set2, "outfit_201")
        assert confirmed.selected_outfit_id == "outfit_201"
        assert confirmed.session_status is SessionStatus.CONFIRMED
        assert build_confirmed_outfit(confirmed, set2).recommendation.outfit_id == "outfit_201"

    def test_input_state_not_modified(self, state, set1):
        snapshot = state.model_dump()
        modify_requirements(state, budget_patch(90))
        assert state.model_dump() == snapshot
        assert state.last_recommended_outfit_ids == ROUND1


class TestBuildConfirmedOutfit:
    def test_returns_confirmed_recommendation(self, state, set1):
        confirmed_state = confirm_outfit(state, set1, "outfit_102")
        confirmed = build_confirmed_outfit(confirmed_state, set1)
        assert isinstance(confirmed, ConfirmedOutfit)
        assert confirmed.session_id == "session_001"
        assert confirmed.selected_outfit_id == "outfit_102"
        assert confirmed.recommendation == set1.recommendations[1]

    def test_requires_confirmed_state(self, state, set1):
        with pytest.raises(ValueError, match="CONFIRMED"):
            build_confirmed_outfit(state, set1)

    def test_requires_current_set(self, state, set1):
        confirmed_state = confirm_outfit(state, set1, "outfit_102")
        with pytest.raises(StaleRecommendationSetError):
            build_confirmed_outfit(confirmed_state, make_set(1, ["a", "b", "c"]))

    def test_json_round_trip(self, state, set1):
        confirmed = build_confirmed_outfit(confirm_outfit(state, set1, "outfit_101"), set1)
        assert ConfirmedOutfit.model_validate_json(confirmed.model_dump_json()) == confirmed
