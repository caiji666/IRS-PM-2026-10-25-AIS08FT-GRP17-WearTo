"""Tests for Stage 12 scoring and ranking (Module 4, ranking_engine)."""

import math

import pytest

from travel_agent.recommendation.student5.models import (
    CandidateOutfit,
    FeasibleCandidateOutfit,
    ScoredOutfit,
)
from travel_agent.recommendation.student5.ranking_engine import (
    MissingCompatibilityScoreError,
    calculate_final_outfit_score,
    calculate_outfit_product_relevance,
    rank_outfits,
    score_outfit,
    select_top_outfits,
    validate_beta,
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
        product("TOP", scores[0], **top),
        product("BOTTOM", scores[1]),
        product("FOOTWEAR", scores[2]),
    ]
    if outerwear is not None:
        items.append(product("OUTERWEAR", outerwear))
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
    def test_mean_of_three_items(self):
        outfit = feasible(scores=(0.25, 0.5, 0.75))
        assert calculate_outfit_product_relevance(outfit) == 0.5

    def test_mean_includes_outerwear(self):
        outfit = feasible(scores=(0.25, 0.5, 0.75), outerwear=1.0)
        assert calculate_outfit_product_relevance(outfit) == 0.625

    def test_user_owned_item_still_counted(self):
        outfit = feasible(scores=(1.0, 0.0, 0.5), is_user_owned=True)
        assert outfit.items[0].is_user_owned
        assert calculate_outfit_product_relevance(outfit) == 0.5

    def test_negative_product_scores(self):
        outfit = feasible(scores=(-0.5, -0.25, -0.75))
        assert calculate_outfit_product_relevance(outfit) == -0.5

    def test_works_before_compatibility_score_exists(self):
        outfit = CandidateOutfit.model_validate(
            {
                "outfit_id": "c",
                "items": [product("TOP", 0.25), product("BOTTOM", 0.25), product("FOOTWEAR", 0.25)],
            }
        )
        assert calculate_outfit_product_relevance(outfit) == 0.25


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

    def test_final_score_can_be_negative(self):
        assert calculate_final_outfit_score(0.0, -0.5, beta=0.5) == -0.25

    @pytest.mark.parametrize("beta", [-0.01, 1.01, -1, 2, math.nan, math.inf, -math.inf])
    def test_beta_out_of_range_rejected(self, beta):
        with pytest.raises(ValueError, match="beta"):
            calculate_final_outfit_score(0.5, 0.5, beta)

    @pytest.mark.parametrize("beta", ["0.5", None, True])
    def test_beta_must_be_a_number(self, beta):
        with pytest.raises(ValueError, match="beta"):
            validate_beta(beta)

    @pytest.mark.parametrize("beta", [0, 0.0, 0.25, 1, 1.0])
    def test_valid_beta(self, beta):
        assert validate_beta(beta) == float(beta)

    @pytest.mark.parametrize("compat", [-0.1, 1.1, math.nan])
    def test_compatibility_score_out_of_range_rejected(self, compat):
        with pytest.raises(ValueError, match="CompatibilityScore"):
            calculate_final_outfit_score(compat, 0.5, 0.5)

    @pytest.mark.parametrize("relevance", [-1.1, 1.1, math.nan])
    def test_relevance_out_of_range_rejected(self, relevance):
        with pytest.raises(ValueError, match="OutfitProductRelevance"):
            calculate_final_outfit_score(0.5, relevance, 0.5)

    def test_extremes_stay_within_bounds(self):
        for beta in (0.0, 0.1, 0.3, 0.7, 0.9, 1.0):
            assert -1.0 <= calculate_final_outfit_score(1.0, 1.0, beta) <= 1.0
            assert -1.0 <= calculate_final_outfit_score(0.0, -1.0, beta) <= 1.0

    def test_deterministic(self):
        values = {calculate_final_outfit_score(0.37, 0.11, 0.42) for _ in range(5)}
        assert len(values) == 1


# ---------------------------------------------------------------------------
# score_outfit
# ---------------------------------------------------------------------------


class TestScoreOutfit:
    def test_returns_scored_outfit(self):
        outfit = feasible("o1", compat=0.75, scores=(0.25, 0.5, 0.75))
        scored = score_outfit(outfit, beta=0.5)
        assert isinstance(scored, ScoredOutfit)
        assert scored.outfit_id == "o1"
        assert scored.compatibility_score == 0.75
        assert scored.outfit_product_relevance == 0.5
        assert scored.final_outfit_score == 0.625

    def test_missing_compatibility_score_rejected(self):
        outfit = feasible("needs_stage11", compat=None)
        with pytest.raises(MissingCompatibilityScoreError, match="needs_stage11"):
            score_outfit(outfit, beta=0.5)

    def test_missing_compatibility_score_is_value_error(self):
        assert issubclass(MissingCompatibilityScoreError, ValueError)

    def test_input_not_modified_and_items_preserved(self):
        outfit = feasible(outerwear=0.25)
        before = outfit.model_dump()
        scored = score_outfit(outfit, beta=0.5)
        assert outfit.model_dump() == before
        assert scored is not outfit
        assert scored.items[0] is not outfit.items[0]
        assert [i.model_dump() for i in scored.items] == before["items"]

    def test_invalid_beta_rejected(self):
        with pytest.raises(ValueError, match="beta"):
            score_outfit(feasible(), beta=1.5)


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

    def test_tie_on_final_and_compatibility_broken_by_relevance(self):
        # beta=1: FinalOutfitScore == CompatibilityScore, relevance differs.
        outfits = [uniform("lower_rel", 0.5, 0.25), uniform("higher_rel", 0.5, 0.75)]
        ranked = rank_outfits(outfits, beta=1.0)
        assert ranked[0].final_outfit_score == ranked[1].final_outfit_score
        assert ids(ranked) == ["higher_rel", "lower_rel"]

    def test_full_tie_keeps_input_order_not_id_order(self):
        outfits = [uniform(i, 0.5, 0.5) for i in ["z", "a", "m", "b"]]
        assert ids(rank_outfits(outfits, beta=0.5)) == ["z", "a", "m", "b"]

    def test_negative_scores_ranked_below_positive(self):
        outfits = [uniform("neg", 0.0, -0.5), uniform("pos", 0.25, 0.25)]
        ranked = rank_outfits(outfits, beta=0.5)
        assert ids(ranked) == ["pos", "neg"]
        assert ranked[1].final_outfit_score == -0.25

    def test_beta_changes_ranking(self):
        outfits = [uniform("compatible", 1.0, 0.0), uniform("relevant", 0.0, 1.0)]
        assert ids(rank_outfits(outfits, beta=1.0)) == ["compatible", "relevant"]
        assert ids(rank_outfits(outfits, beta=0.0)) == ["relevant", "compatible"]

    def test_empty_input_returns_empty_list(self):
        assert rank_outfits([], beta=0.5) == []

    def test_invalid_beta_rejected_even_for_empty_input(self):
        with pytest.raises(ValueError, match="beta"):
            rank_outfits([], beta=-0.5)

    def test_missing_compatibility_score_not_skipped(self):
        outfits = [uniform("ok", 0.5, 0.5), feasible("missing", compat=None)]
        with pytest.raises(MissingCompatibilityScoreError, match="missing"):
            rank_outfits(outfits, beta=0.5)

    def test_only_stage10_feasible_outfits_accepted(self):
        candidate = CandidateOutfit.model_validate(feasible().model_dump())
        with pytest.raises(TypeError, match="FeasibleCandidateOutfit"):
            rank_outfits([candidate], beta=0.5)

    def test_duplicate_outfit_id_rejected(self):
        with pytest.raises(ValueError, match="dup"):
            rank_outfits([uniform("dup", 0.5, 0.5), uniform("dup", 0.25, 0.25)], beta=0.5)

    def test_input_list_and_objects_not_modified(self):
        outfits = [uniform("b", 0.25, 0.25), uniform("a", 1.0, 1.0)]
        snapshot = [o.model_dump() for o in outfits]
        original_list = list(outfits)
        ranked = rank_outfits(outfits, beta=0.5)
        assert outfits == original_list and ids(outfits) == ["b", "a"]
        assert [o.model_dump() for o in outfits] == snapshot
        assert all(r is not o for r in ranked for o in outfits)

    def test_returns_scored_outfits(self):
        ranked = rank_outfits([uniform("a", 0.5, 0.5)], beta=0.5)
        assert all(isinstance(o, ScoredOutfit) for o in ranked)


# ---------------------------------------------------------------------------
# select_top_outfits
# ---------------------------------------------------------------------------


class TestSelectTopOutfits:
    def ranked(self, n):
        return rank_outfits([uniform(f"o{i}", 1.0 - i / 8, 0.5) for i in range(n)], beta=0.5)

    def test_more_than_three_keeps_top_three(self):
        ranked = self.ranked(5)
        assert ids(select_top_outfits(ranked)) == ["o0", "o1", "o2"]

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_fewer_or_equal_returns_all(self, n):
        assert len(select_top_outfits(self.ranked(n))) == n

    def test_empty_input(self):
        assert select_top_outfits([]) == []

    def test_custom_limit(self):
        assert ids(select_top_outfits(self.ranked(5), limit=2)) == ["o0", "o1"]

    @pytest.mark.parametrize("limit", [0, -1])
    def test_invalid_limit_rejected(self, limit):
        with pytest.raises(ValueError, match="limit"):
            select_top_outfits(self.ranked(3), limit=limit)

    def test_returns_new_list(self):
        ranked = self.ranked(2)
        top = select_top_outfits(ranked)
        top.pop()
        assert len(ranked) == 2
