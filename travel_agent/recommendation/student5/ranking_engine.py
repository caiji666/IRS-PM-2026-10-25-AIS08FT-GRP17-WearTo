"""Stage 12 — Final Outfit Ranking (scoring, sorting, Top-3 selection).

Implements the proposal formulas exactly:

    OutfitProductRelevance(o) = mean(ProductScore(item) for item in o)
    FinalOutfitScore(o) = beta * CompatibilityScore(o) + (1 - beta) * OutfitProductRelevance(o)

Input is Stage 10's `feasible_outfits` only, each already carrying a CompatibilityScore written
by Stage 11 (Student 4). This module never calls or imitates the Stage 11 model. `beta` is a
configurable hyperparameter in [0, 1]; no value is assumed optimal.

Ranking order (deterministic): FinalOutfitScore desc, then CompatibilityScore desc, then
OutfitProductRelevance desc, then the original order of `feasible_outfits`. outfit_id is never
used as a tie-breaker because the proposal gives it no format or ordering meaning.

Function names, MissingCompatibilityScoreError and the tie-breaking rule are PROVISIONAL.
"""

from __future__ import annotations

import math
from collections import Counter
from numbers import Real

from travel_agent.recommendation.student5.models import (
    CandidateOutfit,
    FeasibleCandidateOutfit,
    ScoredOutfit,
)

DEFAULT_TOP_K = 3


class MissingCompatibilityScoreError(ValueError):
    """PROVISIONAL. A feasible outfit reached Stage 12 without a CompatibilityScore."""


def validate_beta(beta: float) -> float:
    """Return beta as float if it is a finite real number in [0, 1]; otherwise raise ValueError."""
    if isinstance(beta, bool) or not isinstance(beta, Real):
        raise ValueError(f"beta must be a real number in [0, 1], got {beta!r}")
    if not 0.0 <= beta <= 1.0:  # also rejects NaN
        raise ValueError(f"beta must be in [0, 1], got {beta!r}")
    return float(beta)


def _check_range(name: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:  # also rejects NaN
        raise ValueError(f"{name} must be in [{low}, {high}], got {value!r}")


def calculate_outfit_product_relevance(outfit: CandidateOutfit) -> float:
    """OutfitProductRelevance(o): mean ProductScore over every item, user-owned items included."""
    return math.fsum(item.product_score for item in outfit.items) / len(outfit.items)


def calculate_final_outfit_score(
    compatibility_score: float, outfit_product_relevance: float, beta: float
) -> float:
    """FinalOutfitScore = beta * CompatibilityScore + (1 - beta) * OutfitProductRelevance."""
    beta = validate_beta(beta)
    _check_range("CompatibilityScore", compatibility_score, 0.0, 1.0)
    _check_range("OutfitProductRelevance", outfit_product_relevance, -1.0, 1.0)
    score = beta * compatibility_score + (1 - beta) * outfit_product_relevance
    # A convex combination of values in [-1, 1] stays in [-1, 1]; clamp only guards against
    # last-bit floating-point rounding so ScoredOutfit validation cannot fail spuriously.
    return min(1.0, max(-1.0, score))


def score_outfit(outfit: FeasibleCandidateOutfit, beta: float) -> ScoredOutfit:
    """Return a new ScoredOutfit; the input outfit is not modified."""
    if outfit.compatibility_score is None:
        raise MissingCompatibilityScoreError(
            f"outfit {outfit.outfit_id!r} has no compatibility_score; Stage 11 must score every "
            f"feasible outfit before Stage 12"
        )
    relevance = calculate_outfit_product_relevance(outfit)
    final = calculate_final_outfit_score(outfit.compatibility_score, relevance, beta)
    return ScoredOutfit.model_validate(
        {
            **outfit.model_dump(),
            "outfit_product_relevance": relevance,
            "final_outfit_score": final,
        }
    )


def rank_outfits(
    feasible_outfits: list[FeasibleCandidateOutfit], beta: float
) -> list[ScoredOutfit]:
    """Score and sort Stage 10 feasible outfits. Empty input returns an empty list."""
    beta = validate_beta(beta)
    for outfit in feasible_outfits:
        if not isinstance(outfit, FeasibleCandidateOutfit):
            raise TypeError(
                f"rank_outfits only accepts FeasibleCandidateOutfit from Stage 10, "
                f"got {type(outfit).__name__}"
            )
    repeated = sorted(i for i, n in Counter(o.outfit_id for o in feasible_outfits).items() if n > 1)
    if repeated:
        raise ValueError(f"feasible outfit_id values must be unique, duplicated: {repeated}")

    scored = [score_outfit(outfit, beta) for outfit in feasible_outfits]
    # sorted() is stable, so full ties keep the original feasible_outfits order.
    return sorted(
        scored,
        key=lambda o: (-o.final_outfit_score, -o.compatibility_score, -o.outfit_product_relevance),
    )


def select_top_outfits(
    ranked_outfits: list[ScoredOutfit], limit: int = DEFAULT_TOP_K
) -> list[ScoredOutfit]:
    """Return the first `limit` ranked outfits (Top-3 by default) as a new list."""
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    return list(ranked_outfits[:limit])
