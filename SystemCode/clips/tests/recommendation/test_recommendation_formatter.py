"""Tests for Stage 12 RecommendationSet formatting (Module 4, recommendation_formatter)."""

import json
from decimal import Decimal

import pytest

from travel_agent.recommendation.student5.constraint_engine import CurrencyMismatchError
from travel_agent.recommendation.student5.models import (
    FeasibleCandidateOutfit,
    RecommendationItem,
    ScoredOutfit,
)
from travel_agent.recommendation.student5.recommendation_formatter import (
    EmptyRecommendationError,
    build_recommendation_set,
    format_recommendation_item,
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

    def test_all_user_owned_outfit_needs_explicit_currency(self):
        owned = [product(c, is_user_owned=True) for c in ("TOP", "BOTTOM", "FOOTWEAR")]
        with pytest.raises(ValueError, match="currency"):
            format_recommendation_item(scored("owned", items=owned), 1)
        item = format_recommendation_item(scored("owned", items=owned), 1, currency="SGD")
        assert item.purchase_items == [] and item.total_purchase_cost == Decimal("0")
        assert item.currency == "SGD"

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
    def test_positions_start_at_one_in_ranked_order(self):
        rs = build_recommendation_set(ranked(3), recommendation_round=1)
        assert rs.recommendations[0].display_position == 1
        assert rs.recommendations[0].outfit_id == "o1"

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

    def test_duplicate_outfit_id_rejected(self):
        outfits = [scored("same", final=0.9), scored("same", final=0.8)]
        with pytest.raises(ValueError, match="same"):
            build_recommendation_set(outfits, recommendation_round=1)

    def test_empty_list_raises_documented_error(self):
        with pytest.raises(EmptyRecommendationError):
            build_recommendation_set([], recommendation_round=1)
        assert issubclass(EmptyRecommendationError, ValueError)

    def test_more_than_three_rejected(self):
        with pytest.raises(ValueError, match="select_top_outfits"):
            build_recommendation_set(ranked(4), recommendation_round=1)


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


class TestExistingVersusPurchasable:
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


# ---------------------------------------------------------------------------
# Lightweight text output
# ---------------------------------------------------------------------------
