"""Tests for REQUEST_EXPLANATION handling (Module 6, explanation_handler)."""

import re
from decimal import Decimal

import pytest

from travel_agent.recommendation.student5.constraint_engine import CurrencyMismatchError
from travel_agent.recommendation.student5.dialogue_state import (
    InvalidOutfitSelectionError,
    StaleRecommendationSetError,
    create_initial_state,
    record_recommendation_set,
)
from travel_agent.recommendation.student5.explanation_handler import (
    RecommendationExplanation,
    explain_outfit,
)
from travel_agent.recommendation.student5.models import (
    RecommendationSet,
    TripShoppingRequest,
)


def make_request(**overrides):
    data = {
        "destination": "Seoul",
        "start_date": "2026-12-20",
        "end_date": "2026-12-27",
        "activities": ["hiking"],
        "style": ["minimalist"],
        "user_size": "M",
        "budget": 500,
        "currency": "SGD",
    }
    data.update(overrides)
    return TripShoppingRequest.model_validate(data)


def product(category, price, owned=False):
    return {
        "product_id": f"p_{category.lower()}",
        "source": "mock",
        "title": f"Mock {category}",
        "product_url": None if owned else f"https://example.com/{category.lower()}",
        "price": price,
        "currency": "SGD",
        "availability": True,
        "category": category,
        "product_score": 0.5,
        "is_user_owned": owned,
    }


def make_set(recommendation_round, outfit_ids, currency="SGD"):
    recs = []
    for pos, oid in enumerate(outfit_ids, start=1):
        purchases = [product("TOP", "39.90"), product("FOOTWEAR", "60.10")]
        for p in purchases:
            p["currency"] = currency
        recs.append(
            {
                "outfit_id": oid,
                "display_position": pos,
                "existing_items": [product("BOTTOM", "80.00", owned=True)],
                "purchase_items": purchases,
                "total_purchase_cost": "100.00",
                "currency": currency,
                "compatibility_score": 0.8125,
                "outfit_product_relevance": -0.125,
                "final_outfit_score": 0.375 - 0.125 * pos,
            }
        )
    return RecommendationSet.model_validate(
        {"recommendation_round": recommendation_round, "recommendations": recs}
    )


ROUND1 = ["outfit_101", "outfit_102", "outfit_103"]


@pytest.fixture
def set1():
    return make_set(1, ROUND1)


@pytest.fixture
def state(set1):
    return record_recommendation_set(create_initial_state("session_001", make_request()), set1)


class TestExplainOutfit:
    def test_structured_fields_copied_from_recommendation(self, state, set1):
        e = explain_outfit(state, set1, "outfit_102")
        assert isinstance(e, RecommendationExplanation)
        assert (e.outfit_id, e.display_position, e.recommendation_round) == ("outfit_102", 2, 1)
        assert e.compatibility_score == 0.8125
        assert e.outfit_product_relevance == -0.125
        assert e.final_outfit_score == 0.125
        assert e.total_purchase_cost == Decimal("100.00")
        assert e.currency == "SGD"
        assert e.budget == Decimal("500")
        assert e.within_budget is True
        assert (e.existing_item_count, e.purchase_item_count) == (1, 2)

    def test_no_invented_reasons(self, state, set1):
        words = set(
            re.findall(r"[a-z]+", explain_outfit(state, set1, "outfit_101").message.lower())
        )
        for invented in (
            "weather",
            "activity",
            "activities",
            "hiking",
            "style",
            "minimalist",
            "size",
            "fit",
            "fits",
            "quality",
            "like",
            "love",
            "colour",
            "color",
            "look",
            "looks",
            "suitable",
            "match",
            "matches",
        ):
            assert invented not in words, invented

    def test_stale_id_rejected(self, state, set1):
        set2 = make_set(2, ["outfit_201", "outfit_202"])
        state2 = record_recommendation_set(state, set2)
        with pytest.raises(InvalidOutfitSelectionError):
            explain_outfit(state2, set2, "outfit_101")
        with pytest.raises(StaleRecommendationSetError):
            explain_outfit(state2, set1, "outfit_101")

    def test_budget_currency_mismatch_rejected(self):
        usd_set = make_set(1, ROUND1, currency="USD")
        state = record_recommendation_set(
            create_initial_state("s", make_request(currency="SGD")), usd_set
        )
        with pytest.raises(CurrencyMismatchError, match="USD"):
            explain_outfit(state, usd_set, "outfit_101")
