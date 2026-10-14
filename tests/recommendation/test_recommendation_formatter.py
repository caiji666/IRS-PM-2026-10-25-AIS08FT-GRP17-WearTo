"""Tests for Stage 12 RecommendationSet formatting (Module 4, recommendation_formatter)."""

import json
from decimal import Decimal

import pytest

from travel_agent.recommendation.student5.constraint_engine import CurrencyMismatchError
from travel_agent.recommendation.student5.models import (
    FeasibleCandidateOutfit,
    RecommendationItem,
    RecommendationSet,
    ScoredOutfit,
)
from travel_agent.recommendation.student5.ranking_engine import rank_outfits
from travel_agent.recommendation.student5.recommendation_formatter import (
    EmptyRecommendationError,
    build_recommendation_set,
    format_recommendation_item,
    format_recommendation_set_text,
)


def product(category, price="50.00", **overrides):
    data = {
        "product_id": f"p_{category.lower()}",
        "source": "mock_catalogue",
        "title": f"Mock {category.lower()}",
        "description": f"A {category.lower()} for travel",
        "product_url": f"https://example.com/{category.lower()}",
        "image_url": f"https://example.com/{category.lower()}.jpg",
        "price": price,
        "currency": "SGD",
        "availability": True,
        "category": category,
        "product_score": 0.5,
        "brand": "MockBrand",
    }
    if overrides.get("is_user_owned"):
        data["product_url"] = None
    data.update(overrides)
    return data


def scored(outfit_id="o1", final=0.5, compat=0.5, relevance=0.5, items=None):
    items = items or [product("TOP"), product("BOTTOM"), product("FOOTWEAR")]
    return ScoredOutfit.model_validate(
        {
            "outfit_id": outfit_id,
            "items": items,
            "compatibility_score": compat,
            "outfit_product_relevance": relevance,
            "final_outfit_score": final,
        }
    )


def ranked(n):
    return [scored(f"o{i}", final=0.9 - 0.1 * i) for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# format_recommendation_item
# ---------------------------------------------------------------------------


class TestFormatRecommendationItem:
    def test_scores_and_ids_copied(self):
        item = format_recommendation_item(
            scored("keep_me", final=0.625, compat=0.75, relevance=0.5), display_position=1
        )
        assert isinstance(item, RecommendationItem)
        assert item.outfit_id == "keep_me"
        assert item.display_position == 1
        assert item.compatibility_score == 0.75
        assert item.outfit_product_relevance == 0.5
        assert item.final_outfit_score == 0.625

    def test_existing_and_purchase_items_separated(self):
        outfit = scored(
            items=[
                product("TOP"),
                product("BOTTOM", is_user_owned=True),
                product("FOOTWEAR"),
                product("OUTERWEAR", is_user_owned=True),
            ]
        )
        item = format_recommendation_item(outfit, 1)
        assert [p.product_id for p in item.existing_items] == ["p_bottom", "p_outerwear"]
        assert [p.product_id for p in item.purchase_items] == ["p_top", "p_footwear"]
        assert all(p.is_user_owned for p in item.existing_items)
        assert not any(p.is_user_owned for p in item.purchase_items)

    def test_user_owned_price_not_in_total(self):
        outfit = scored(
            items=[
                product("TOP", price="40.00"),
                product("BOTTOM", price="999.00", is_user_owned=True),
                product("FOOTWEAR", price="60.00"),
            ]
        )
        item = format_recommendation_item(outfit, 1)
        assert item.total_purchase_cost == Decimal("100.00")
        assert item.existing_items[0].price == Decimal("999.00")

    def test_decimal_total_is_exact(self):
        outfit = scored(
            items=[
                product("TOP", price="0.10"),
                product("BOTTOM", price="0.20"),
                product("FOOTWEAR", price="0.00"),
            ]
        )
        total = format_recommendation_item(outfit, 1).total_purchase_cost
        assert isinstance(total, Decimal)
        assert total == Decimal("0.30")

    def test_all_four_items_fully_preserved(self):
        items = [
            product("TOP", available_sizes=["M"], size_data_reliable=True, embedding_ref="emb/top"),
            product("BOTTOM", colour="navy", material="cotton"),
            product("FOOTWEAR", is_user_owned=True, price="120.00"),
            product("OUTERWEAR", price="149.99", collected_at="2026-10-01T08:00:00Z"),
        ]
        outfit = scored(items=items)
        item = format_recommendation_item(outfit, 1)
        out = {p.product_id: p.model_dump() for p in item.existing_items + item.purchase_items}
        assert out == {p.product_id: p.model_dump() for p in outfit.items}
        assert len(out) == 4

    def test_currency_taken_from_purchase_items(self):
        assert format_recommendation_item(scored(), 1).currency == "SGD"

    def test_purchase_currency_mismatch_fails(self):
        outfit = scored(
            "mixed", items=[product("TOP"), product("BOTTOM", currency="USD"), product("FOOTWEAR")]
        )
        with pytest.raises(CurrencyMismatchError) as exc:
            format_recommendation_item(outfit, 1)
        for expected in ("mixed", "p_bottom", "USD", "SGD"):
            assert expected in str(exc.value)

    def test_explicit_currency_must_match_purchases(self):
        with pytest.raises(CurrencyMismatchError, match="EUR"):
            format_recommendation_item(scored(), 1, currency="EUR")

    def test_user_owned_item_currency_ignored(self):
        outfit = scored(
            items=[
                product("TOP"),
                product("BOTTOM"),
                product("FOOTWEAR", is_user_owned=True, currency="USD"),
            ]
        )
        assert format_recommendation_item(outfit, 1).currency == "SGD"

    def test_all_user_owned_outfit_needs_explicit_currency(self):
        owned = [product(c, is_user_owned=True) for c in ("TOP", "BOTTOM", "FOOTWEAR")]
        with pytest.raises(ValueError, match="currency"):
            format_recommendation_item(scored("owned", items=owned), 1)
        item = format_recommendation_item(scored("owned", items=owned), 1, currency="SGD")
        assert item.purchase_items == [] and item.total_purchase_cost == Decimal("0")
        assert item.currency == "SGD"

    def test_scored_outfit_not_modified(self):
        outfit = scored()
        before = outfit.model_dump()
        item = format_recommendation_item(outfit, 1)
        assert outfit.model_dump() == before
        assert item.purchase_items[0] is not outfit.items[0]

    def test_rejects_unscored_outfit(self):
        unscored = FeasibleCandidateOutfit.model_validate(
            {
                "outfit_id": "x",
                "items": [product("TOP"), product("BOTTOM"), product("FOOTWEAR")],
                "compatibility_score": 0.5,
            }
        )
        with pytest.raises(TypeError, match="ScoredOutfit"):
            format_recommendation_item(unscored, 1)


# ---------------------------------------------------------------------------
# build_recommendation_set
# ---------------------------------------------------------------------------


class TestBuildRecommendationSet:
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_one_to_three_recommendations(self, n):
        rs = build_recommendation_set(ranked(n), recommendation_round=1)
        assert isinstance(rs, RecommendationSet)
        assert len(rs.recommendations) == n
        assert [r.display_position for r in rs.recommendations] == list(range(1, n + 1))
        assert [r.outfit_id for r in rs.recommendations] == [f"o{i}" for i in range(1, n + 1)]

    def test_positions_start_at_one_in_ranked_order(self):
        rs = build_recommendation_set(ranked(3), recommendation_round=1)
        assert rs.recommendations[0].display_position == 1
        assert rs.recommendations[0].outfit_id == "o1"

    def test_recommendation_round_preserved(self):
        assert build_recommendation_set(ranked(2), recommendation_round=4).recommendation_round == 4

    def test_empty_list_raises_documented_error(self):
        with pytest.raises(EmptyRecommendationError):
            build_recommendation_set([], recommendation_round=1)
        assert issubclass(EmptyRecommendationError, ValueError)

    def test_more_than_three_rejected(self):
        with pytest.raises(ValueError, match="select_top_outfits"):
            build_recommendation_set(ranked(4), recommendation_round=1)

    def test_duplicate_outfit_id_rejected(self):
        outfits = [scored("same", final=0.9), scored("same", final=0.8)]
        with pytest.raises(ValueError, match="same"):
            build_recommendation_set(outfits, recommendation_round=1)

    def test_invalid_round_rejected(self):
        with pytest.raises(ValueError):
            build_recommendation_set(ranked(1), recommendation_round=0)

    def test_unsorted_input_rejected_by_set_validation(self):
        outfits = [scored("a", final=0.1), scored("b", final=0.9)]
        with pytest.raises(ValueError, match="final_outfit_score"):
            build_recommendation_set(outfits, recommendation_round=1)

    def test_currency_passed_to_items(self):
        owned = [product(c, is_user_owned=True) for c in ("TOP", "BOTTOM", "FOOTWEAR")]
        rs = build_recommendation_set(
            [scored("owned", items=owned)], recommendation_round=1, currency="SGD"
        )
        assert rs.recommendations[0].currency == "SGD"

    def test_inputs_not_modified(self):
        outfits = ranked(3)
        snapshot = [o.model_dump() for o in outfits]
        build_recommendation_set(outfits, recommendation_round=1)
        assert [o.model_dump() for o in outfits] == snapshot

    def test_round_trip(self):
        rs = build_recommendation_set(ranked(3), recommendation_round=2)
        assert RecommendationSet.model_validate(rs.model_dump()) == rs
        assert RecommendationSet.model_validate_json(rs.model_dump_json()) == rs

    def test_decimal_json_serialisation(self):
        outfit = scored(
            items=[
                product("TOP", price="19.90"),
                product("BOTTOM", price="0.10"),
                product("FOOTWEAR", price="80.00", is_user_owned=True),
            ]
        )
        rs = build_recommendation_set([outfit], recommendation_round=1)
        dumped = json.loads(rs.model_dump_json())["recommendations"][0]
        assert dumped["total_purchase_cost"] == "20.00"
        assert dumped["purchase_items"][0]["price"] == "19.90"
        assert dumped["existing_items"][0]["price"] == "80.00"

    def test_end_to_end_with_ranking_engine(self):
        def outfit(outfit_id, compat):
            return FeasibleCandidateOutfit.model_validate(
                {
                    "outfit_id": outfit_id,
                    "compatibility_score": compat,
                    "items": [product("TOP"), product("BOTTOM"), product("FOOTWEAR")],
                }
            )

        ranked_outfits = rank_outfits(
            [outfit("c", 0.25), outfit("a", 1.0), outfit("b", 0.5)], beta=0.5
        )
        rs = build_recommendation_set(ranked_outfits, recommendation_round=1)
        assert [r.outfit_id for r in rs.recommendations] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Module 5 — detailed allocation: product information and existing vs purchasable items
# ---------------------------------------------------------------------------

REQUIRED_PRODUCT_FIELDS = (
    "product_id",
    "title",
    "image_url",
    "product_url",
    "price",
    "currency",
    "availability",
    "category",
    "product_score",
    "is_user_owned",
)


def mixed_outfit(outfit_id="mix"):
    """TOP + FOOTWEAR to purchase, BOTTOM + OUTERWEAR already owned."""
    return scored(
        outfit_id,
        final=0.625,
        compat=0.75,
        relevance=0.5,
        items=[
            product(
                "TOP",
                price="39.90",
                product_score=0.25,
                available_sizes=["M"],
                size_data_reliable=True,
            ),
            product(
                "BOTTOM",
                price="80.00",
                is_user_owned=True,
                image_url=None,
                product_score=0.75,
                colour="navy",
            ),
            product("FOOTWEAR", price="60.10", product_score=-0.25, material="leather"),
            product("OUTERWEAR", price="200.00", is_user_owned=True, product_score=0.5),
        ],
    )


def all_items(item):
    return item.existing_items + item.purchase_items


class TestProductInformationPreserved:
    def test_complete_product_information_kept(self):
        outfit = mixed_outfit()
        item = format_recommendation_item(outfit, 1)
        by_id = {p.product_id: p.model_dump() for p in all_items(item)}
        for original in outfit.items:
            assert by_id[original.product_id] == original.model_dump()

    @pytest.mark.parametrize("field", REQUIRED_PRODUCT_FIELDS)
    def test_required_field_present_in_output(self, field):
        item = format_recommendation_item(mixed_outfit(), 1)
        for p in all_items(item):
            assert field in p.model_dump()

    def test_title_image_url_product_url_kept(self):
        item = format_recommendation_item(mixed_outfit(), 1)
        top = item.purchase_items[0]
        assert top.title == "Mock top"
        assert top.image_url == "https://example.com/top.jpg"
        assert top.product_url == "https://example.com/top"
        bottom = item.existing_items[0]
        assert bottom.image_url is None and bottom.product_url is None

    def test_price_currency_and_product_score_kept(self):
        item = format_recommendation_item(mixed_outfit(), 1)
        prices = {p.product_id: (p.price, p.currency, p.product_score) for p in all_items(item)}
        assert prices == {
            "p_top": (Decimal("39.90"), "SGD", 0.25),
            "p_bottom": (Decimal("80.00"), "SGD", 0.75),
            "p_footwear": (Decimal("60.10"), "SGD", -0.25),
            "p_outerwear": (Decimal("200.00"), "SGD", 0.5),
        }

    def test_optional_fields_kept(self):
        item = format_recommendation_item(mixed_outfit(), 1)
        by_id = {p.product_id: p for p in all_items(item)}
        assert by_id["p_top"].available_sizes == ["M"]
        assert by_id["p_bottom"].colour == "navy"
        assert by_id["p_footwear"].material == "leather"
        assert by_id["p_top"].description == "A top for travel"
        assert by_id["p_top"].brand == "MockBrand"


class TestExistingVersusPurchasable:
    def test_owned_items_go_to_existing_items(self):
        item = format_recommendation_item(mixed_outfit(), 1)
        assert [p.product_id for p in item.existing_items] == ["p_bottom", "p_outerwear"]
        assert all(p.is_user_owned for p in item.existing_items)

    def test_purchasable_items_go_to_purchase_items(self):
        item = format_recommendation_item(mixed_outfit(), 1)
        assert [p.product_id for p in item.purchase_items] == ["p_top", "p_footwear"]
        assert not any(p.is_user_owned for p in item.purchase_items)

    def test_every_item_appears_exactly_once(self):
        outfit = mixed_outfit()
        item = format_recommendation_item(outfit, 1)
        out_ids = [p.product_id for p in all_items(item)]
        assert sorted(out_ids) == sorted(p.product_id for p in outfit.items)
        assert len(out_ids) == len(set(out_ids)) == len(outfit.items)

    def test_no_existing_items(self):
        item = format_recommendation_item(scored(), 1)
        assert item.existing_items == []
        assert len(item.purchase_items) == 3

    def test_all_existing_items(self):
        owned = [
            product(c, is_user_owned=True, price="70.00") for c in ("TOP", "BOTTOM", "FOOTWEAR")
        ]
        item = format_recommendation_item(scored("owned", items=owned), 1, currency="SGD")
        assert len(item.existing_items) == 3 and item.purchase_items == []
        assert item.total_purchase_cost == Decimal("0")
        assert [p.price for p in item.existing_items] == [Decimal("70.00")] * 3

    def test_total_counts_purchase_items_only(self):
        item = format_recommendation_item(mixed_outfit(), 1)
        assert item.total_purchase_cost == Decimal("100.00")  # 39.90 + 60.10
        assert isinstance(item.total_purchase_cost, Decimal)

    def test_multi_currency_purchases_rejected(self):
        outfit = scored(
            items=[
                product("TOP"),
                product("BOTTOM", currency="JPY"),
                product("FOOTWEAR", currency="USD"),
            ]
        )
        with pytest.raises(CurrencyMismatchError):
            format_recommendation_item(outfit, 1)


class TestIdentifiersAndScoresKept:
    def make_set(self):
        outfits = [mixed_outfit("first"), scored("second", final=0.5, compat=0.5, relevance=0.5)]
        return build_recommendation_set(outfits, recommendation_round=3)

    def test_outfit_id_position_and_round_unchanged(self):
        rs = self.make_set()
        assert rs.recommendation_round == 3
        assert [(r.outfit_id, r.display_position) for r in rs.recommendations] == [
            ("first", 1),
            ("second", 2),
        ]

    def test_three_scores_kept_without_recomputation(self):
        first = self.make_set().recommendations[0]
        # Stored scores are copied as-is; they are not recomputed from product scores.
        assert (
            first.compatibility_score,
            first.outfit_product_relevance,
            first.final_outfit_score,
        ) == (0.75, 0.5, 0.625)

    def test_json_contains_product_information(self):
        data = json.loads(self.make_set().model_dump_json())
        first = data["recommendations"][0]
        top = first["purchase_items"][0]
        for field in REQUIRED_PRODUCT_FIELDS:
            assert field in top
        assert top["title"] == "Mock top"
        assert top["image_url"] == "https://example.com/top.jpg"
        assert first["existing_items"][0]["product_id"] == "p_bottom"

    def test_json_decimal_serialisation(self):
        first = json.loads(self.make_set().model_dump_json())["recommendations"][0]
        assert first["total_purchase_cost"] == "100.00"
        assert first["purchase_items"][0]["price"] == "39.90"
        assert first["existing_items"][1]["price"] == "200.00"

    def test_json_round_trip(self):
        rs = self.make_set()
        assert RecommendationSet.model_validate_json(rs.model_dump_json()) == rs
        assert RecommendationSet.model_validate(json.loads(rs.model_dump_json())) == rs

    def test_no_beta_or_explanation_fields(self):
        dumped = self.make_set().model_dump_json().lower()
        for forbidden in ("beta", "explanation", "recommendation_reason"):
            assert forbidden not in dumped


# ---------------------------------------------------------------------------
# Lightweight text output
# ---------------------------------------------------------------------------


class TestRecommendationSetText:
    def make_set(self):
        outfits = [mixed_outfit("first"), scored("second", final=0.5, compat=0.5, relevance=0.5)]
        return build_recommendation_set(outfits, recommendation_round=2)

    def test_existing_and_purchase_sections(self):
        text = format_recommendation_set_text(self.make_set())
        first, second = text.split("#2")
        assert "Existing items:" in first and "Items to purchase:" in first
        assert first.index("Mock bottom") < first.index("Items to purchase:")
        assert first.index("Items to purchase:") < first.index("Mock top")
        assert "Existing items:" not in second
        assert "Items to purchase:" in second

    def test_positions_ids_and_round_shown(self):
        text = format_recommendation_set_text(self.make_set())
        assert "Recommendation round 2" in text
        assert "#1 outfit_id: first" in text
        assert "#2 outfit_id: second" in text
        assert text.index("#1") < text.index("#2")

    def test_product_name_category_and_price_shown(self):
        text = format_recommendation_set_text(self.make_set())
        assert "[TOP] Mock top - 39.90 SGD" in text
        assert "[OUTERWEAR] Mock outerwear - 200.00 SGD" in text

    def test_missing_product_url_shows_not_available(self):
        text = format_recommendation_set_text(self.make_set())
        bottom_line = next(line for line in text.splitlines() if "Mock bottom" in line)
        assert "Link: Not available" in bottom_line
        assert "Image: Not available" in bottom_line
        top_line = next(line for line in text.splitlines() if "Mock top" in line)
        assert "Link: https://example.com/top" in top_line
        assert "Image: https://example.com/top.jpg" in top_line

    def test_total_purchase_cost_shown(self):
        text = format_recommendation_set_text(self.make_set())
        assert "Total purchase cost: 100.00 SGD" in text
        assert "Total purchase cost: 150.00 SGD" in text

    def test_three_scores_shown(self):
        first = format_recommendation_set_text(self.make_set()).split("#2")[0]
        assert "CompatibilityScore: 0.7500" in first
        assert "OutfitProductRelevance: 0.5000" in first
        assert "FinalOutfitScore: 0.6250" in first

    def test_all_existing_outfit_shows_no_purchases(self):
        owned = [product(c, is_user_owned=True) for c in ("TOP", "BOTTOM", "FOOTWEAR")]
        rs = build_recommendation_set([scored("owned", items=owned)], 1, currency="SGD")
        text = format_recommendation_set_text(rs)
        assert "Items to purchase: none" in text
        assert "Total purchase cost: 0 SGD" in text

    def test_deterministic(self):
        assert format_recommendation_set_text(self.make_set()) == format_recommendation_set_text(
            self.make_set()
        )

    def test_input_not_modified(self):
        rs = self.make_set()
        before = rs.model_dump()
        format_recommendation_set_text(rs)
        assert rs.model_dump() == before

    def test_no_ansi_beta_or_explanation(self):
        text = format_recommendation_set_text(self.make_set())
        assert "\x1b" not in text
        for forbidden in ("beta", "explanation", "reason"):
            assert forbidden not in text.lower()

    def test_rejects_non_recommendation_set(self):
        with pytest.raises(TypeError, match="RecommendationSet"):
            format_recommendation_set_text(self.make_set().model_dump())
