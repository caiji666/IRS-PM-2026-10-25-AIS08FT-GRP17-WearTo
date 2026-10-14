"""Module 9 — Student5Orchestrator: thin end-to-end flow over the existing Stage 10–14 functions.

    CandidateOutfits (with CompatibilityScore)
    → Stage 10 filter_candidate_outfits
    → Stage 12 rank_outfits → select_top_outfits → build_recommendation_set
    → record_recommendation_set (Conversational Recommendation State)
    → Stage 13 modify_requirements (+ plan_from_update_result) / confirm / explain
    → Stage 14 build_preview_request → generate_preview_with_fallback

The orchestrator does not know or care whether candidate outfits come from mock fixtures or from
Student 4 (via student4_adapter). It never computes CompatibilityScore, never re-runs upstream
stages itself (the caller does that using the returned RerunPlan), and holds no hidden state:
every call takes the current state and returns a new one. A preview never changes the confirmed
state.

Class and result names are PROVISIONAL.
"""

from __future__ import annotations

from dataclasses import dataclass

from travel_agent.recommendation.student5.confirmation_handler import (
    build_confirmed_outfit,
    confirm_outfit,
)
from travel_agent.recommendation.student5.constraint_engine import (
    ConstraintConflict,
    ConstraintFilterResult,
    filter_candidate_outfits,
)
from travel_agent.recommendation.student5.dialogue_state import (
    create_initial_state,
    modify_requirements,
    record_recommendation_set,
)
from travel_agent.recommendation.student5.explanation_handler import (
    RecommendationExplanation,
    explain_outfit,
)
from travel_agent.recommendation.student5.models import (
    CandidateOutfit,
    ConfirmedOutfit,
    ConversationalRecommendationState,
    PreviewMode,
    PreviewResult,
    RecommendationSet,
    ScoredOutfit,
    TripShoppingRequest,
    TripShoppingRequestPatch,
)
from travel_agent.recommendation.student5.patch_processor import RequirementUpdateResult
from travel_agent.recommendation.student5.preview_adapter import (
    PreviewProvider,
    build_preview_request,
    generate_preview_with_fallback,
)
from travel_agent.recommendation.student5.ranking_engine import (
    rank_outfits,
    select_top_outfits,
    validate_beta,
)
from travel_agent.recommendation.student5.recommendation_formatter import build_recommendation_set
from travel_agent.recommendation.student5.rerun_planner import RerunPlan, plan_from_update_result


@dataclass(frozen=True)
class RecommendationRun:
    """PROVISIONAL. Result of one Stage 10 → 12 recommendation round."""

    state: ConversationalRecommendationState
    filter_result: ConstraintFilterResult
    ranked_outfits: list[ScoredOutfit]
    recommendation_set: RecommendationSet | None

    @property
    def conflict(self) -> ConstraintConflict | None:
        """Structured explanation when candidates existed but none was feasible."""
        return self.filter_result.conflict


@dataclass(frozen=True)
class RequirementChange:
    """PROVISIONAL. Result of MODIFY_REQUIREMENTS; `rerun_plan` is None when nothing changed."""

    state: ConversationalRecommendationState
    update: RequirementUpdateResult
    rerun_plan: RerunPlan | None


@dataclass(frozen=True)
class ConfirmationOutcome:
    """PROVISIONAL. Result of CONFIRM_OUTFIT."""

    state: ConversationalRecommendationState
    confirmed_outfit: ConfirmedOutfit


class Student5Orchestrator:
    """PROVISIONAL. Stateless coordinator for Stages 10, 12, 13 and 14."""

    def __init__(
        self,
        *,
        beta: float,
        preview_provider: PreviewProvider | None = None,
        session_id: str = "session",
    ) -> None:
        self.beta = validate_beta(beta)
        self.preview_provider = preview_provider
        self.session_id = session_id

    def run(
        self,
        *,
        request: TripShoppingRequest,
        candidate_outfits: list[CandidateOutfit],
        state: ConversationalRecommendationState | None = None,
    ) -> RecommendationRun:
        """Filter, rank and record one recommendation round.

        Without `state` a new session is started from `request`; with `state` its
        current_request must equal `request` (call after modify_requirements with the
        updated request and freshly produced candidates).
        """
        if state is None:
            state = create_initial_state(self.session_id, request)
        elif state.current_request != request:
            raise ValueError("request must equal state.current_request")
        filter_result = filter_candidate_outfits(request, candidate_outfits)
        ranked = rank_outfits(filter_result.feasible_outfits, self.beta)
        top = select_top_outfits(ranked)
        recommendation_set = None
        if top:
            recommendation_set = build_recommendation_set(
                top, state.recommendation_round + 1, currency=request.currency
            )
            state = record_recommendation_set(state, recommendation_set)
        return RecommendationRun(
            state=state,
            filter_result=filter_result,
            ranked_outfits=ranked,
            recommendation_set=recommendation_set,
        )

    def modify_requirements(
        self, state: ConversationalRecommendationState, patch: TripShoppingRequestPatch
    ) -> RequirementChange:
        """MODIFY_REQUIREMENTS: merge the patch and plan which stages the caller must re-run."""
        new_state, update = modify_requirements(state, patch)
        plan = plan_from_update_result(update) if update.changed else None
        return RequirementChange(state=new_state, update=update, rerun_plan=plan)

    def confirm(
        self,
        state: ConversationalRecommendationState,
        recommendation_set: RecommendationSet,
        outfit_id: str,
    ) -> ConfirmationOutcome:
        """CONFIRM_OUTFIT: confirm a current outfit and return the saved final selection."""
        new_state = confirm_outfit(state, recommendation_set, outfit_id)
        return ConfirmationOutcome(
            state=new_state, confirmed_outfit=build_confirmed_outfit(new_state, recommendation_set)
        )

    def explain(
        self,
        state: ConversationalRecommendationState,
        recommendation_set: RecommendationSet,
        outfit_id: str,
    ) -> RecommendationExplanation:
        """REQUEST_EXPLANATION for a current outfit (read-only)."""
        return explain_outfit(state, recommendation_set, outfit_id)

    def generate_preview(
        self,
        confirmed_outfit: ConfirmedOutfit,
        preview_mode: PreviewMode | str,
        user_portrait: str | None = None,
        portrait_consent: bool = False,
    ) -> PreviewResult:
        """Stage 14 for a confirmed outfit; falls back to product cards, never undoes the
        confirmation."""
        request = build_preview_request(
            confirmed_outfit, preview_mode, user_portrait, portrait_consent
        )
        return generate_preview_with_fallback(request, self.preview_provider)
