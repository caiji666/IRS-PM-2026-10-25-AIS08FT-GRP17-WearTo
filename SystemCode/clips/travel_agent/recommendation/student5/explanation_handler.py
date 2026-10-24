"""Stage 13 — REQUEST_EXPLANATION.

Explains one outfit of the current RecommendationSet using only data already present in it and
the current budget: display position, the three stored scores, purchase total, budget status
and item counts. Nothing is re-scored, re-ranked or re-filtered, and the state is not changed.

The message is a fixed English template (no LLM, no randomness). It never claims weather,
activity, style, size, quality or visual-matching suitability, because no such reason is
recorded in the recommendation data.

RecommendationExplanation and explain_outfit are PROVISIONAL.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import Field, model_validator

from travel_agent.recommendation.student5.constraint_engine import CurrencyMismatchError
from travel_agent.recommendation.student5.dialogue_state import resolve_current_recommendation
from travel_agent.recommendation.student5.models import (
    CompatibilityScoreValue,
    ContractModel,
    ConversationalRecommendationState,
    CosineScoreValue,
    Money,
    NonEmptyStr,
    RecommendationSet,
)


class RecommendationExplanation(ContractModel):
    """PROVISIONAL. Deterministic, data-only explanation of one recommended outfit."""

    outfit_id: NonEmptyStr
    display_position: int = Field(ge=1)
    recommendation_round: int = Field(ge=1)
    compatibility_score: CompatibilityScoreValue
    outfit_product_relevance: CosineScoreValue
    final_outfit_score: CosineScoreValue
    total_purchase_cost: Money
    currency: NonEmptyStr
    budget: Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]
    within_budget: bool
    existing_item_count: int = Field(ge=0)
    purchase_item_count: int = Field(ge=0)
    message: NonEmptyStr

    @model_validator(mode="after")
    def _budget_flag_consistent(self) -> RecommendationExplanation:
        if self.within_budget != (self.total_purchase_cost <= self.budget):
            raise ValueError("within_budget does not match total_purchase_cost and budget")
        return self


def explain_outfit(
    state: ConversationalRecommendationState,
    recommendation_set: RecommendationSet,
    outfit_id: str,
) -> RecommendationExplanation:
    """Explain a current recommendation; the state and set are not modified."""
    item = resolve_current_recommendation(state, recommendation_set, outfit_id)
    request = state.current_request
    if item.currency != request.currency:
        raise CurrencyMismatchError(
            f"outfit {item.outfit_id!r} is priced in {item.currency} but the budget is in "
            f"{request.currency}"
        )
    within_budget = item.total_purchase_cost <= request.budget
    budget_phrase = "is within" if within_budget else "exceeds"
    message = (
        f"Outfit {item.outfit_id} is shown at position {item.display_position} of "
        f"recommendation round {recommendation_set.recommendation_round}, ordered by "
        f"FinalOutfitScore. Scores: CompatibilityScore {item.compatibility_score:.4f}, "
        f"OutfitProductRelevance {item.outfit_product_relevance:.4f}, "
        f"FinalOutfitScore {item.final_outfit_score:.4f}. "
        f"It has {len(item.purchase_items)} item(s) to purchase costing "
        f"{item.total_purchase_cost} {item.currency}, which {budget_phrase} the current budget "
        f"of {request.budget} {request.currency}, and {len(item.existing_items)} item(s) you "
        f"already own."
    )
    return RecommendationExplanation(
        outfit_id=item.outfit_id,
        display_position=item.display_position,
        recommendation_round=recommendation_set.recommendation_round,
        compatibility_score=item.compatibility_score,
        outfit_product_relevance=item.outfit_product_relevance,
        final_outfit_score=item.final_outfit_score,
        total_purchase_cost=item.total_purchase_cost,
        currency=item.currency,
        budget=request.budget,
        within_budget=within_budget,
        existing_item_count=len(item.existing_items),
        purchase_item_count=len(item.purchase_items),
        message=message,
    )
