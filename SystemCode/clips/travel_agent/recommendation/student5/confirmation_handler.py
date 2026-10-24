"""Stage 13 — CONFIRM_OUTFIT.

Proposal: the user's selection is resolved against the Top-3 outfits of the latest round, saved
as the final selection, and the recommendation loop ends.

Rules (PROVISIONAL where the proposal is silent):

- `selected_outfit_id` must be an exact outfit_id of the current RecommendationSet, which must
  match the state's recorded round and IDs (see dialogue_state.resolve_current_recommendation).
- On success: selected_outfit_id is set and session_status becomes CONFIRMED (a PROVISIONAL
  status). request_version, recommendation_round, current_request and
  last_recommended_outfit_ids are unchanged.
- Confirming the same outfit again is idempotent (an equal new state is returned).
- Confirming a different outfit after confirmation raises OutfitAlreadyConfirmedError.

Function names and OutfitAlreadyConfirmedError are PROVISIONAL.
"""

from __future__ import annotations

from travel_agent.recommendation.student5.dialogue_state import (
    ensure_current_recommendation_set,
    resolve_current_recommendation,
)
from travel_agent.recommendation.student5.models import (
    ConfirmedOutfit,
    ConversationalRecommendationState,
    RecommendationSet,
    SessionStatus,
)


class OutfitAlreadyConfirmedError(ValueError):
    """PROVISIONAL. A different outfit was already confirmed in this session."""


def confirm_outfit(
    state: ConversationalRecommendationState,
    recommendation_set: RecommendationSet,
    selected_outfit_id: str,
) -> ConversationalRecommendationState:
    """Confirm one outfit of the current RecommendationSet; returns a new state."""
    if (
        state.session_status is SessionStatus.CONFIRMED
        and selected_outfit_id != state.selected_outfit_id
    ):
        raise OutfitAlreadyConfirmedError(
            f"outfit {state.selected_outfit_id!r} is already confirmed; "
            f"cannot switch to {selected_outfit_id!r}"
        )
    item = resolve_current_recommendation(state, recommendation_set, selected_outfit_id)
    return ConversationalRecommendationState.model_validate(
        {
            **state.model_dump(),
            "selected_outfit_id": item.outfit_id,
            "session_status": SessionStatus.CONFIRMED,
        }
    )


def build_confirmed_outfit(
    state: ConversationalRecommendationState, recommendation_set: RecommendationSet
) -> ConfirmedOutfit:
    """Return the saved final selection (ConfirmedOutfit) of a CONFIRMED state.

    Staleness is checked first, so a set invalidated by a requirement change is always
    reported as StaleRecommendationSetError.
    """
    ensure_current_recommendation_set(state, recommendation_set)
    if state.session_status is not SessionStatus.CONFIRMED:
        raise ValueError(f"session must be CONFIRMED, got {state.session_status}")
    item = resolve_current_recommendation(state, recommendation_set, state.selected_outfit_id)
    return ConfirmedOutfit.model_validate(
        {
            "session_id": state.session_id,
            "selected_outfit_id": item.outfit_id,
            "recommendation": item.model_dump(),
        }
    )
