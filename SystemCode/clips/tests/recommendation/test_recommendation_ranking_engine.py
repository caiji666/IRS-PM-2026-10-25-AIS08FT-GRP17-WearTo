"""Tests for Stage 12 scoring and ranking (Module 4, ranking_engine)."""

import math

import pytest

from travel_agent.recommendation.student5.models import (
    CandidateOutfit,
    FeasibleCandidateOutfit,
)
from travel_agent.recommendation.student5.ranking_engine import (
    MissingCompatibilityScoreError,
    calculate_final_outfit_score,
    calculate_outfit_product_relevance,
    rank_outfits,
    score_outfit,
    select_top_outfits,
)

# Product scores below are dyadic fractions (0.25, 0.5, ...) so float arithmetic is exact
# and tie-breaking tests compare genuinely equal scores.


def product(category, score, **overrides):
    data = {
        "product_id": f"p_{category.lower()}",
        "source": "mock_catalogue",
        "title": f"Mock {category.lower()}",
        "product_url": f"https://example.com/{category.lower()}",
        "price": "50.00",
        "currency": "SGD",
        "availability": True,
        "category": category,
        "product_score": score,
    }
    if overrides.get("is_user_owned"):
        data["product_url"] = None
    data.update(overrides)
    return data


def feasible(outfit_id="o1", compat=0.5, scores=(0.5, 0.5, 0.5), outerwear=None, **top):
    items = [
        product("TOP", scores[0], product_id=f"{outfit_id}_top", **top),
        product("BOTTOM", scores[1], product_id=f"{outfit_id}_bottom"),
        product("FOOTWEAR", scores[2], product_id=f"{outfit_id}_footwear"),
    ]
    if outerwear is not None:
        items.append(product("OUTERWEAR", outerwear, product_id=f"{outfit_id}_outerwear"))
    return FeasibleCandidateOutfit.model_validate(
        {"outfit_id": outfit_id, "items": items, "compatibility_score": compat}
    )


def uniform(outfit_id, compat, relevance):
    """Outfit whose OutfitProductRelevance equals `relevance` exactly."""
    return feasible(outfit_id, compat, (relevance, relevance, relevance))


# ---------------------------------------------------------------------------
# OutfitProductRelevance
# ---------------------------------------------------------------------------


class TestOutfitProductRelevance:
    def test_mean_includes_outerwear(self):
        outfit = feasible(scores=(0.25, 0.5, 0.75), outerwear=1.0)
        assert calculate_outfit_product_relevance(outfit) == 0.625


# ---------------------------------------------------------------------------
# beta and FinalOutfitScore
# ---------------------------------------------------------------------------


class TestFinalOutfitScore:
    def test_beta_zero_uses_relevance_only(self):
        assert calculate_final_outfit_score(0.75, -0.25, beta=0.0) == -0.25

    def test_beta_one_uses_compatibility_only(self):
        assert calculate_final_outfit_score(0.75, -0.25, beta=1.0) == 0.75

    def test_beta_half(self):
        assert calculate_final_outfit_score(0.8, 0.5, beta=0.5) == pytest.approx(0.65)

    def test_formula(self):
        beta, c, r = 0.3, 0.9, 0.2
        assert calculate_final_outfit_score(c, r, beta) == pytest.approx(beta * c + (1 - beta) * r)

    @pytest.mark.parametrize("beta", [-0.01, 1.01])
    def test_beta_out_of_range_rejected(self, beta):
        with pytest.raises(ValueError, match="beta"):
            calculate_final_outfit_score(0.5, 0.5, beta)

    @pytest.mark.parametrize("relevance", [-1.1, 1.1, math.nan])
    def test_relevance_out_of_range_rejected(self, relevance):
        with pytest.raises(ValueError, match="OutfitProductRelevance"):
            calculate_final_outfit_score(0.5, relevance, 0.5)


# ---------------------------------------------------------------------------
# score_outfit
# ---------------------------------------------------------------------------


class TestScoreOutfit:
    def test_missing_compatibility_score_rejected(self):
        outfit = feasible("needs_stage11", compat=None)
        with pytest.raises(MissingCompatibilityScoreError, match="needs_stage11"):
            score_outfit(outfit, beta=0.5)


# ---------------------------------------------------------------------------
# rank_outfits
# ---------------------------------------------------------------------------


def ids(outfits):
    return [o.outfit_id for o in outfits]


class TestRankOutfits:
    def test_sorted_by_final_score_descending(self):
        outfits = [uniform("low", 0.25, 0.25), uniform("high", 1.0, 0.75), uniform("mid", 0.5, 0.5)]
        ranked = rank_outfits(outfits, beta=0.5)
        assert ids(ranked) == ["high", "mid", "low"]
        scores = [o.final_outfit_score for o in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_tie_on_final_score_broken_by_compatibility(self):
        # beta=0.5: 0.5*0.5 + 0.5*0.5 == 0.5*0.75 + 0.5*0.25 == 0.5
        outfits = [uniform("lower_compat", 0.5, 0.5), uniform("higher_compat", 0.75, 0.25)]
        ranked = rank_outfits(outfits, beta=0.5)
        assert ranked[0].final_outfit_score == ranked[1].final_outfit_score == 0.5
        assert ids(ranked) == ["higher_compat", "lower_compat"]

    def test_full_tie_keeps_input_order_not_id_order(self):
        outfits = [uniform(i, 0.5, 0.5) for i in ["z", "a", "m", "b"]]
        assert ids(rank_outfits(outfits, beta=0.5)) == ["z", "a", "m", "b"]

    def test_duplicate_outfit_id_rejected(self):
        with pytest.raises(ValueError, match="dup"):
            rank_outfits([uniform("dup", 0.5, 0.5), uniform("dup", 0.25, 0.25)], beta=0.5)

    def test_only_stage10_feasible_outfits_accepted(self):
        candidate = CandidateOutfit.model_validate(feasible().model_dump())
        with pytest.raises(TypeError, match="FeasibleCandidateOutfit"):
            rank_outfits([candidate], beta=0.5)


# ---------------------------------------------------------------------------
# select_top_outfits
# ---------------------------------------------------------------------------


class TestSelectTopOutfits:
    def ranked(self, n):
        return rank_outfits([uniform(f"o{i}", 1.0 - i / 8, 0.5) for i in range(n)], beta=0.5)

    def test_more_than_three_keeps_top_three(self):
        ranked = self.ranked(5)
        assert ids(select_top_outfits(ranked)) == ["o0", "o1", "o2"]

    def test_skips_higher_scored_outfits_that_repeat_more_than_one_selected_product(self):
        ranked = self.ranked(5)
        ranked[1].items[0].product_id = ranked[0].items[0].product_id
        ranked[1].items[1].product_id = ranked[0].items[1].product_id
        ranked[2].items[0].product_id = ranked[0].items[0].product_id

        assert ids(select_top_outfits(ranked)) == ["o0", "o2", "o3"]

    @pytest.mark.parametrize("limit", [0, -1])
    def test_invalid_limit_rejected(self, limit):
        with pytest.raises(ValueError, match="limit"):
            select_top_outfits(self.ranked(3), limit=limit)
