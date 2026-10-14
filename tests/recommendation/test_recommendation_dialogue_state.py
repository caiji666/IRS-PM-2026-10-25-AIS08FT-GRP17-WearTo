"""Tests for the Conversational Recommendation State (Module 6, dialogue_state)."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from travel_agent.recommendation.student5.confirmation_handler import confirm_outfit
from travel_agent.recommendation.student5.dialogue_state import (
    InvalidOutfitSelectionError,
    StaleRecommendationSetError,
    create_initial_state,
    modify_requirements,
    record_recommendation_set,
    resolve_current_recommendation,
)
from travel_agent.recommendation.student5.explanation_handler import explain_outfit
from travel_agent.recommendation.student5.models import (
    ConversationalRecommendationState,
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

    def test_request_copied(self):
        request = make_request()
        state = create_initial_state("s", request)
        assert state.current_request == request
        assert state.current_request is not request

    def test_blank_session_id_rejected(self):
        with pytest.raises(ValidationError):
            create_initial_state("  ", make_request())


# ---------------------------------------------------------------------------
# MODIFY_REQUIREMENTS
# ---------------------------------------------------------------------------


class TestModifyRequirements:
    def test_request_version_incremented(self, round1):
        state, result = modify_requirements(round1, modify(set={"budget": 400}))
        assert state.request_version == 2
        assert state.current_request.budget == Decimal("400")
        assert result.changed_fields == ["budget"]
        assert result.previous_request == round1.current_request
        assert result.updated_request == state.current_request

    def test_recommendation_round_not_incremented(self, round1):
        state, _ = modify_requirements(round1, modify(set={"budget": 400}))
        assert state.recommendation_round == 1

    def test_selection_cleared_and_awaiting_confirmation(self, round1):
        confirmed = confirm_outfit(round1, make_set(1, ROUND1), "outfit_102")
        state, _ = modify_requirements(confirmed, modify(add={"style": ["casual"]}))
        assert state.selected_outfit_id is None
        assert state.session_status is SessionStatus.AWAITING_CONFIRMATION

    def test_session_id_kept(self, round1):
        state, _ = modify_requirements(round1, modify(set={"budget": 400}))
        assert state.session_id == "session_001"

    @pytest.mark.parametrize(
        "operations",
        [
            {"set": {"budget": 400}},
            {"add": {"excluded_items": ["long coat"]}},
            {"set": {"user_size": "M"}},
            {"add": {"style": ["minimalist"]}},
        ],
        ids=["budget", "excluded_items", "user_size", "style"],
    )
    def test_real_change_invalidates_current_recommendations(self, round1, operations):
        state, result = modify_requirements(round1, modify(**operations))
        assert result.changed is True
        assert state.last_recommended_outfit_ids == []
        assert state.recommendation_round == 1  # only a new recorded set advances the round

    def test_no_change_keeps_state_version_and_ids(self, round1):
        state, result = modify_requirements(round1, modify(add={"activities": ["dinner"]}))
        assert result.changed is False
        assert state == round1
        assert state.request_version == 1
        assert state.last_recommended_outfit_ids == ROUND1
        # the current set is still valid
        assert resolve_current_recommendation(state, make_set(1, ROUND1), "outfit_101")

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

    def test_next_set_valid_after_change(self, round1):
        state, _ = modify_requirements(round1, modify(set={"budget": 400}))
        state = record_recommendation_set(state, make_set(2, ROUND2))
        assert state.last_recommended_outfit_ids == ROUND2
        item = resolve_current_recommendation(state, make_set(2, ROUND2), "outfit_203")
        assert item.display_position == 3

    def test_invalid_patch_leaves_state_unchanged(self, round1):
        snapshot = round1.model_dump()
        with pytest.raises(ValidationError):
            modify_requirements(round1, modify(set={"budget": -1}))
        assert round1.model_dump() == snapshot

    def test_input_state_not_modified(self, round1):
        snapshot = round1.model_dump()
        state, _ = modify_requirements(round1, modify(add={"activities": ["hiking"]}))
        assert round1.model_dump() == snapshot
        assert state is not round1


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

    @pytest.mark.parametrize("n", [1, 2])
    def test_fewer_than_three_ids(self, initial, n):
        state = record_recommendation_set(initial, make_set(1, ROUND1[:n]))
        assert state.last_recommended_outfit_ids == ROUND1[:n]

    def test_new_set_replaces_previous_ids(self, round1):
        state = record_recommendation_set(round1, make_set(2, ROUND2))
        assert state.recommendation_round == 2
        assert state.last_recommended_outfit_ids == ROUND2

    def test_request_and_version_unchanged(self, round1):
        modified, _ = modify_requirements(round1, modify(set={"budget": 400}))
        state = record_recommendation_set(modified, make_set(2, ROUND2))
        assert state.request_version == 2
        assert state.current_request == modified.current_request

    def test_clears_previous_confirmation(self, round1):
        confirmed = confirm_outfit(round1, make_set(1, ROUND1), "outfit_101")
        state = record_recommendation_set(confirmed, make_set(2, ROUND2))
        assert state.selected_outfit_id is None
        assert state.session_status is SessionStatus.AWAITING_CONFIRMATION

    @pytest.mark.parametrize("wrong_round", [1, 3])
    def test_set_round_must_be_next_round(self, round1, wrong_round):
        with pytest.raises(ValueError, match="recommendation_round"):
            record_recommendation_set(round1, make_set(wrong_round, ROUND2))

    def test_inputs_not_modified(self, round1):
        state_snapshot = round1.model_dump()
        rs = make_set(2, ROUND2)
        set_snapshot = rs.model_dump()
        record_recommendation_set(round1, rs)
        assert round1.model_dump() == state_snapshot
        assert rs.model_dump() == set_snapshot


# ---------------------------------------------------------------------------
# Resolving an outfit against the current recommendations
# ---------------------------------------------------------------------------


class TestResolveCurrentRecommendation:
    def test_returns_matching_item(self, round1):
        item = resolve_current_recommendation(round1, make_set(1, ROUND1), "outfit_102")
        assert item.outfit_id == "outfit_102"
        assert item.display_position == 2

    def test_unknown_id_rejected(self, round1):
        with pytest.raises(InvalidOutfitSelectionError, match="outfit_999"):
            resolve_current_recommendation(round1, make_set(1, ROUND1), "outfit_999")

    @pytest.mark.parametrize("bad", [2, "2", "", "   ", None])
    def test_display_position_or_blank_not_accepted_as_id(self, round1, bad):
        with pytest.raises(InvalidOutfitSelectionError):
            resolve_current_recommendation(round1, make_set(1, ROUND1), bad)

    def test_no_close_match_guessing(self, round1):
        with pytest.raises(InvalidOutfitSelectionError):
            resolve_current_recommendation(round1, make_set(1, ROUND1), "outfit_10")

    def test_previous_round_set_is_stale(self, round1):
        state = record_recommendation_set(round1, make_set(2, ROUND2))
        with pytest.raises(StaleRecommendationSetError):
            resolve_current_recommendation(state, make_set(1, ROUND1), "outfit_101")

    def test_set_with_different_ids_is_stale(self, round1):
        with pytest.raises(StaleRecommendationSetError):
            resolve_current_recommendation(round1, make_set(1, ["x", "y", "z"]), "x")

    def test_no_recommendations_yet_is_stale(self, initial):
        with pytest.raises(StaleRecommendationSetError):
            resolve_current_recommendation(initial, make_set(1, ROUND1), "outfit_101")

    def test_errors_are_value_errors(self):
        assert issubclass(InvalidOutfitSelectionError, ValueError)
        assert issubclass(StaleRecommendationSetError, ValueError)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestStateSerialisation:
    def test_json_round_trip(self, round1):
        state, _ = modify_requirements(round1, modify(set={"budget": 400}))
        assert (
            ConversationalRecommendationState.model_validate_json(state.model_dump_json()) == state
        )
        assert ConversationalRecommendationState.model_validate(state.model_dump()) == state

    def test_unknown_field_rejected(self, round1):
        data = round1.model_dump()
        data["recommendation_set"] = None
        with pytest.raises(ValidationError, match="extra"):
            ConversationalRecommendationState.model_validate(data)


# ---------------------------------------------------------------------------
# Full multi-round flow
# ---------------------------------------------------------------------------


class TestMultiRoundFlow:
    def test_full_flow(self):
        # 1. initial state
        state = create_initial_state("session_001", make_request())
        assert (state.request_version, state.recommendation_round) == (1, 0)

        # 2. first RecommendationSet recorded
        set1 = make_set(1, ROUND1)
        state = record_recommendation_set(state, set1)
        assert state.recommendation_round == 1
        assert state.last_recommended_outfit_ids == ROUND1
        assert state.session_status is SessionStatus.AWAITING_CONFIRMATION

        # 3. MODIFY_REQUIREMENTS
        state, update = modify_requirements(
            state, modify(set={"budget": 400}, add={"excluded_items": ["long coat"]})
        )
        assert update.changed_fields == ["budget", "excluded_items"]
        assert state.request_version == 2
        assert state.recommendation_round == 1
        assert state.selected_outfit_id is None
        assert state.last_recommended_outfit_ids == []

        # 3b. the first-round set is immediately stale
        with pytest.raises(StaleRecommendationSetError):
            confirm_outfit(state, set1, "outfit_101")
        with pytest.raises(StaleRecommendationSetError):
            explain_outfit(state, set1, "outfit_101")

        # 4. second RecommendationSet recorded
        set2 = make_set(2, ROUND2)
        state = record_recommendation_set(state, set2)
        assert state.recommendation_round == 2
        assert state.last_recommended_outfit_ids == ROUND2

        # 5. previous-round outfits can no longer be confirmed or explained
        with pytest.raises(InvalidOutfitSelectionError):
            confirm_outfit(state, set2, "outfit_101")
        with pytest.raises(StaleRecommendationSetError):
            confirm_outfit(state, set1, "outfit_101")
        with pytest.raises(InvalidOutfitSelectionError):
            explain_outfit(state, set2, "outfit_102")
        with pytest.raises(StaleRecommendationSetError):
            explain_outfit(state, set1, "outfit_102")

        # 6. confirm a valid second-round outfit
        explanation = explain_outfit(state, set2, "outfit_202")
        assert explanation.display_position == 2
        state = confirm_outfit(state, set2, "outfit_202")
        assert state.selected_outfit_id == "outfit_202"
        assert state.session_status is SessionStatus.CONFIRMED
        assert (state.request_version, state.recommendation_round) == (2, 2)
        assert state.current_request.budget == Decimal("400")
        assert state.current_request.excluded_items == ["long coat"]
