"""Stage 13 — Conversational Recommendation State transitions.

Uses the proposal's ConversationalRecommendationState (Appendix A) unchanged. Every function
returns a new, re-validated state; inputs are never modified.

Transition rules (PROVISIONAL where the proposal is silent):

- `create_initial_state`: request_version 1, recommendation_round 0, no recommended IDs,
  no selection, session_status AWAITING_CONFIRMATION (the only status named by the proposal).
- `modify_requirements` (MODIFY_REQUIREMENTS): merges the patch; if anything changed,
  request_version += 1, last_recommended_outfit_ids is cleared to [] (PROVISIONAL: "there is no
  valid recommendation left to confirm or explain" — the previous RecommendationSet is
  immediately stale; history, if needed, belongs in a separate store), selected_outfit_id is
  cleared and session_status returns to AWAITING_CONFIRMATION. recommendation_round is not
  touched (it only advances when a new RecommendationSet is recorded). A patch that changes
  nothing returns an equal state, so the current RecommendationSet stays valid.
- `record_recommendation_set`: recommendation_round += 1 (the set must carry that round),
  last_recommended_outfit_ids is replaced by the set's IDs in display order, selection is
  cleared, session_status is AWAITING_CONFIRMATION; the request and its version are kept.
- `resolve_current_recommendation`: an outfit can only be confirmed or explained when the
  RecommendationSet passed in is the one recorded in the state (same round, same IDs in the
  same order) and the outfit_id is an exact member of it. No fuzzy matching; display positions
  are not accepted as IDs. Earlier rounds are rejected as stale.

No pipeline stage is run here; deciding what to re-run belongs to the later rerun planner.
Function names and the two exception classes are PROVISIONAL.
"""

from __future__ import annotations

from typing import Any

from travel_agent.recommendation.student5.models import (
    ConversationalRecommendationState,
    RecommendationItem,
    RecommendationSet,
    SessionStatus,
    TripShoppingRequest,
    TripShoppingRequestPatch,
)
from travel_agent.recommendation.student5.patch_processor import (
    RequirementUpdateResult,
    apply_requirement_patch,
)


class StaleRecommendationSetError(ValueError):
    """PROVISIONAL. The RecommendationSet is not the one currently recorded in the state."""


class InvalidOutfitSelectionError(ValueError):
    """PROVISIONAL. The outfit_id is not part of the current recommendations."""


def _updated(
    state: ConversationalRecommendationState, **changes: Any
) -> ConversationalRecommendationState:
    return ConversationalRecommendationState.model_validate({**state.model_dump(), **changes})


def create_initial_state(
    session_id: str, request: TripShoppingRequest
) -> ConversationalRecommendationState:
    """State before any recommendation round."""
    return ConversationalRecommendationState.model_validate(
        {
            "session_id": session_id,
            "request_version": 1,
            "current_request": request.model_dump(),
            "recommendation_round": 0,
            "last_recommended_outfit_ids": [],
            "selected_outfit_id": None,
            "session_status": SessionStatus.AWAITING_CONFIRMATION,
        }
    )


def modify_requirements(
    state: ConversationalRecommendationState, patch: TripShoppingRequestPatch
) -> tuple[ConversationalRecommendationState, RequirementUpdateResult]:
    """Apply a MODIFY_REQUIREMENTS patch. Returns the new state and the update result."""
    result = apply_requirement_patch(state.current_request, patch)
    if not result.changed:
        return _updated(state), result
    new_state = _updated(
        state,
        current_request=result.updated_request.model_dump(),
        request_version=state.request_version + 1,
        last_recommended_outfit_ids=[],
        selected_outfit_id=None,
        session_status=SessionStatus.AWAITING_CONFIRMATION,
    )
    return new_state, result


def record_recommendation_set(
    state: ConversationalRecommendationState, recommendation_set: RecommendationSet
) -> ConversationalRecommendationState:
    """Store a newly generated RecommendationSet as the current recommendation round."""
    expected_round = state.recommendation_round + 1
    if recommendation_set.recommendation_round != expected_round:
        raise ValueError(
            f"recommendation_round of the new set must be {expected_round}, "
            f"got {recommendation_set.recommendation_round}"
        )
    return _updated(
        state,
        recommendation_round=expected_round,
        last_recommended_outfit_ids=[r.outfit_id for r in recommendation_set.recommendations],
        selected_outfit_id=None,
        session_status=SessionStatus.AWAITING_CONFIRMATION,
    )


def ensure_current_recommendation_set(
    state: ConversationalRecommendationState, recommendation_set: RecommendationSet
) -> None:
    """Raise StaleRecommendationSetError unless the set is the one recorded in the state.

    After a real requirement change last_recommended_outfit_ids is [], so every earlier set is
    stale until a new one is recorded.
    """
    set_ids = [r.outfit_id for r in recommendation_set.recommendations]
    if (
        recommendation_set.recommendation_round != state.recommendation_round
        or set_ids != state.last_recommended_outfit_ids
    ):
        raise StaleRecommendationSetError(
            f"RecommendationSet round {recommendation_set.recommendation_round} {set_ids} is not "
            f"the current recommendation round {state.recommendation_round} "
            f"{state.last_recommended_outfit_ids}"
        )


def resolve_current_recommendation(
    state: ConversationalRecommendationState,
    recommendation_set: RecommendationSet,
    outfit_id: str,
) -> RecommendationItem:
    """Return the current RecommendationItem for `outfit_id`, or raise."""
    if not isinstance(outfit_id, str) or not outfit_id.strip():
        raise InvalidOutfitSelectionError(
            f"outfit_id must be a non-empty string, got {outfit_id!r}"
        )
    ensure_current_recommendation_set(state, recommendation_set)
    set_ids = [r.outfit_id for r in recommendation_set.recommendations]
    for item in recommendation_set.recommendations:
        if item.outfit_id == outfit_id:
            return item
    raise InvalidOutfitSelectionError(
        f"outfit_id {outfit_id!r} is not in the current recommendations {set_ids}"
    )
