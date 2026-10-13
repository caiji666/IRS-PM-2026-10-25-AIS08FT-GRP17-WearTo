"""Tests for the proposal data contract models (Module 1)."""

import json
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from travel_agent.recommendation.student5.models import (
    CandidateOutfit,
    ConfirmedOutfit,
    ConversationalRecommendationState,
    FeasibleCandidateOutfit,
    PreviewItemReference,
    PreviewMode,
    PreviewRequest,
    PreviewResult,
    PreviewStatus,
    Product,
    ProductCategory,
    RecommendationIntent,
    RecommendationItem,
    RecommendationSet,
    ScoredOutfit,
    SessionStatus,
    TripShoppingRequest,
    TripShoppingRequestPatch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def request_data(**overrides):
    data = {
        "destination": "Seoul",
        "start_date": "2026-12-20",
        "end_date": "2026-12-27",
        "budget": 500,
        "currency": "SGD",
    }
    data.update(overrides)
    return data


def product_data(category="TOP", **overrides):
    data = {
        "product_id": f"p_{category.lower()}",
        "source": "mock_catalogue",
        "title": f"Mock {category.lower()}",
        "product_url": f"https://example.com/{category.lower()}",
        "image_url": f"https://example.com/{category.lower()}.jpg",
        "price": "50.00",
        "currency": "SGD",
        "availability": True,
        "category": category,
        "product_score": 0.5,
    }
    data.update(overrides)
    return data


def outfit_items(include_outerwear=False):
    categories = ["TOP", "BOTTOM", "FOOTWEAR"]
    if include_outerwear:
        categories.append("OUTERWEAR")
    return [product_data(c) for c in categories]


def recommendation_item_data(outfit_id="o1", position=1, final_score=0.7):
    return {
        "outfit_id": outfit_id,
        "display_position": position,
        "existing_items": [
            product_data("FOOTWEAR", is_user_owned=True, product_url=None, price="80.00")
        ],
        "purchase_items": [product_data("TOP"), product_data("BOTTOM", price="30.00")],
        "total_purchase_cost": "80.00",
        "currency": "SGD",
        "compatibility_score": 0.8,
        "outfit_product_relevance": 0.6,
        "final_outfit_score": final_score,
    }


def state_data(**overrides):
    data = {
        "session_id": "session_001",
        "request_version": 2,
        "current_request": request_data(budget=400, excluded_items=["long coat"]),
        "recommendation_round": 2,
        "last_recommended_outfit_ids": ["outfit_201", "outfit_202", "outfit_203"],
        "selected_outfit_id": None,
        "session_status": "AWAITING_CONFIRMATION",
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# TripShoppingRequest
# ---------------------------------------------------------------------------


class TestTripShoppingRequest:
    def test_minimal_valid_request(self):
        req = TripShoppingRequest.model_validate(request_data())
        assert req.destination == "Seoul"
        assert req.start_date == date(2026, 12, 20)
        assert req.budget == Decimal("500")
        assert req.activities == []
        assert req.style == []
        assert req.excluded_items == []
        assert req.user_size is None

    def test_full_request(self):
        req = TripShoppingRequest.model_validate(
            request_data(
                activities=["sightseeing", "dinner"],
                style=["minimalist"],
                excluded_items=["long coat"],
                user_size="M",
            )
        )
        assert req.activities == ["sightseeing", "dinner"]
        assert req.style == ["minimalist"]
        assert req.excluded_items == ["long coat"]
        assert req.user_size == "M"

    def test_appendix_a_example_validates(self):
        appendix_a = {
            "destination": "Seoul",
            "start_date": "2026-12-20",
            "end_date": "2026-12-27",
            "activities": ["sightseeing", "dinner"],
            "budget": 500,
            "currency": "SGD",
            "style": [],
        }
        assert TripShoppingRequest.model_validate(appendix_a).budget == Decimal("500")

    def test_same_day_trip_allowed(self):
        req = TripShoppingRequest.model_validate(
            request_data(start_date="2026-12-20", end_date="2026-12-20")
        )
        assert req.start_date == req.end_date

    def test_end_date_before_start_date_rejected(self):
        with pytest.raises(ValidationError, match="end_date"):
            TripShoppingRequest.model_validate(
                request_data(start_date="2026-12-27", end_date="2026-12-20")
            )

    @pytest.mark.parametrize("budget", [0, -1, "-0.01"])
    def test_non_positive_budget_rejected(self, budget):
        with pytest.raises(ValidationError):
            TripShoppingRequest.model_validate(request_data(budget=budget))

    @pytest.mark.parametrize("field", ["destination", "currency"])
    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_strings_rejected(self, field, value):
        with pytest.raises(ValidationError):
            TripShoppingRequest.model_validate(request_data(**{field: value}))

    def test_missing_required_field_rejected(self):
        data = request_data()
        del data["budget"]
        with pytest.raises(ValidationError):
            TripShoppingRequest.model_validate(data)

    def test_non_iso_date_rejected(self):
        with pytest.raises(ValidationError):
            TripShoppingRequest.model_validate(request_data(start_date="20/12/2026"))

    def test_does_not_invent_weather_or_style(self):
        req = TripShoppingRequest.model_validate(request_data())
        dumped = req.model_dump()
        assert "weather" not in json.dumps(dumped, default=str)
        assert dumped["style"] == [] and dumped["activities"] == []

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="extra"):
            TripShoppingRequest.model_validate(request_data(weather_class="COLD"))

    def test_default_lists_not_shared(self):
        a = TripShoppingRequest.model_validate(request_data())
        b = TripShoppingRequest.model_validate(request_data())
        a.activities.append("hiking")
        a.style.append("casual")
        a.excluded_items.append("boots")
        assert b.activities == [] and b.style == [] and b.excluded_items == []

    def test_json_serialisation_iso_dates_and_decimal_string(self):
        req = TripShoppingRequest.model_validate(request_data(budget="500.00"))
        dumped = req.model_dump(mode="json")
        assert dumped["start_date"] == "2026-12-20"
        assert dumped["end_date"] == "2026-12-27"
        assert dumped["budget"] == "500.00"
        assert isinstance(dumped["budget"], str)

    def test_float_budget_kept_exact_as_decimal(self):
        req = TripShoppingRequest.model_validate_json(
            '{"destination": "Seoul", "start_date": "2026-12-20", '
            '"end_date": "2026-12-27", "budget": 499.99, "currency": "SGD"}'
        )
        assert req.budget == Decimal("499.99")

    def test_round_trip(self):
        req = TripShoppingRequest.model_validate(request_data(activities=["dinner"]))
        assert TripShoppingRequest.model_validate(req.model_dump()) == req
        assert TripShoppingRequest.model_validate_json(req.model_dump_json()) == req


# ---------------------------------------------------------------------------
# TripShoppingRequestPatch
# ---------------------------------------------------------------------------


class TestTripShoppingRequestPatch:
    def test_appendix_a_patch(self):
        patch = TripShoppingRequestPatch.model_validate(
            {
                "intent": "MODIFY_REQUIREMENTS",
                "set": {"budget": 400},
                "add": {"excluded_items": ["long coat"]},
                "remove": {},
            }
        )
        assert patch.intent is RecommendationIntent.MODIFY_REQUIREMENTS
        assert patch.set == {"budget": 400}
        assert patch.add == {"excluded_items": ["long coat"]}
        assert patch.remove == {}

    def test_external_json_keys_preserved(self):
        patch = TripShoppingRequestPatch.model_validate(
            {"intent": "MODIFY_REQUIREMENTS", "set": {"budget": 400}}
        )
        assert set(patch.model_dump(mode="json")) == {"intent", "set", "add", "remove"}

    @pytest.mark.parametrize(
        "intent", ["MODIFY_REQUIREMENTS", "CONFIRM_OUTFIT", "REQUEST_EXPLANATION"]
    )
    def test_allowed_intents(self, intent):
        assert TripShoppingRequestPatch.model_validate({"intent": intent}).intent == intent

    @pytest.mark.parametrize("intent", ["MODIFY", "confirm_outfit", "CHITCHAT", ""])
    def test_illegal_intent_rejected(self, intent):
        with pytest.raises(ValidationError):
            TripShoppingRequestPatch.model_validate({"intent": intent})

    def test_unknown_request_field_in_set_rejected(self):
        with pytest.raises(ValidationError, match="budgett"):
            TripShoppingRequestPatch.model_validate(
                {"intent": "MODIFY_REQUIREMENTS", "set": {"budgett": 400}}
            )

    def test_add_on_non_list_field_rejected(self):
        with pytest.raises(ValidationError):
            TripShoppingRequestPatch.model_validate(
                {"intent": "MODIFY_REQUIREMENTS", "add": {"budget": [100]}}
            )

    def test_add_value_must_be_list(self):
        with pytest.raises(ValidationError):
            TripShoppingRequestPatch.model_validate(
                {"intent": "MODIFY_REQUIREMENTS", "add": {"style": "casual"}}
            )

    def test_unknown_field_in_remove_rejected(self):
        with pytest.raises(ValidationError):
            TripShoppingRequestPatch.model_validate(
                {"intent": "MODIFY_REQUIREMENTS", "remove": {"colour": ["red"]}}
            )

    @pytest.mark.parametrize(
        "operations,field",
        [
            ({"set": {"style": ["casual"]}, "add": {"style": ["minimalist"]}}, "style"),
            (
                {"add": {"excluded_items": ["boots"]}, "remove": {"excluded_items": ["boots"]}},
                "excluded_items",
            ),
            ({"set": {"budget": 400}, "remove": {"budget": None}}, "budget"),
        ],
        ids=["set_add_style", "add_remove_excluded_items", "set_remove_budget"],
    )
    def test_overlapping_operations_rejected(self, operations, field):
        with pytest.raises(ValidationError, match=f"more than one.*{field}"):
            TripShoppingRequestPatch.model_validate({"intent": "MODIFY_REQUIREMENTS", **operations})

    def test_non_overlapping_operations_accepted(self):
        patch = TripShoppingRequestPatch.model_validate(
            {
                "intent": "MODIFY_REQUIREMENTS",
                "set": {"budget": 400, "style": ["minimalist"]},
                "add": {"excluded_items": ["long coat"]},
                "remove": {"activities": ["hiking"]},
            }
        )
        assert set(patch.set) == {"budget", "style"}
        assert set(patch.add) == {"excluded_items"}
        assert set(patch.remove) == {"activities"}

    def test_confirm_outfit_patch_must_not_carry_changes(self):
        with pytest.raises(ValidationError):
            TripShoppingRequestPatch.model_validate(
                {"intent": "CONFIRM_OUTFIT", "set": {"budget": 300}}
            )

    def test_unknown_top_level_field_rejected(self):
        with pytest.raises(ValidationError, match="extra"):
            TripShoppingRequestPatch.model_validate(
                {"intent": "MODIFY_REQUIREMENTS", "replace": {}}
            )

    def test_default_dicts_not_shared(self):
        a = TripShoppingRequestPatch.model_validate({"intent": "MODIFY_REQUIREMENTS"})
        b = TripShoppingRequestPatch.model_validate({"intent": "MODIFY_REQUIREMENTS"})
        a.set["budget"] = 1
        assert b.set == {}

    def test_round_trip(self):
        patch = TripShoppingRequestPatch.model_validate(
            {
                "intent": "MODIFY_REQUIREMENTS",
                "set": {"budget": 400},
                "add": {"style": ["minimalist"]},
            }
        )
        assert TripShoppingRequestPatch.model_validate(patch.model_dump()) == patch
        assert TripShoppingRequestPatch.model_validate_json(patch.model_dump_json()) == patch


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------


class TestProduct:
    def test_valid_product(self):
        p = Product.model_validate(product_data())
        assert p.category is ProductCategory.TOP
        assert p.price == Decimal("50.00")
        assert p.availability is True
        assert p.is_user_owned is False
        assert p.available_sizes is None

    def test_negative_price_rejected(self):
        with pytest.raises(ValidationError):
            Product.model_validate(product_data(price="-0.01"))

    def test_zero_price_allowed(self):
        assert Product.model_validate(product_data(price="0")).price == Decimal("0")

    @pytest.mark.parametrize("score", [-1.0, -0.3, 0.0, 1.0])
    def test_product_score_boundaries_accepted(self, score):
        assert Product.model_validate(product_data(product_score=score)).product_score == score

    @pytest.mark.parametrize("score", [-1.0001, 1.0001, float("nan"), float("inf")])
    def test_product_score_out_of_range_rejected(self, score):
        with pytest.raises(ValidationError):
            Product.model_validate(product_data(product_score=score))

    @pytest.mark.parametrize("category", ["DRESS", "top", "ACCESSORY"])
    def test_illegal_category_rejected(self, category):
        with pytest.raises(ValidationError):
            Product.model_validate(product_data(category=category))

    def test_availability_must_be_explicit_boolean(self):
        data = product_data()
        del data["availability"]
        with pytest.raises(ValidationError):
            Product.model_validate(data)
        with pytest.raises(ValidationError):
            Product.model_validate(product_data(availability="in stock"))

    def test_user_owned_item_keeps_original_price(self):
        p = Product.model_validate(
            product_data(is_user_owned=True, product_url=None, price="120.00")
        )
        assert p.is_user_owned is True
        assert p.price == Decimal("120.00")

    def test_purchasable_product_requires_product_url(self):
        with pytest.raises(ValidationError, match="product_url"):
            Product.model_validate(product_data(product_url=None))

    def test_sizes_and_embedding_reference(self):
        p = Product.model_validate(
            product_data(available_sizes=["S", "M"], embedding_ref="emb/p_top.npy")
        )
        assert p.available_sizes == ["S", "M"]
        assert p.embedding_ref == "emb/p_top.npy"

    def test_size_data_unreliable_by_default(self):
        assert Product.model_validate(product_data()).size_data_reliable is False

    def test_reliable_size_data_requires_size_list(self):
        with pytest.raises(ValidationError, match="available_sizes"):
            Product.model_validate(product_data(size_data_reliable=True))

    @pytest.mark.parametrize("available_sizes", [[], ["M"]])
    def test_reliable_size_data_with_list(self, available_sizes):
        p = Product.model_validate(
            product_data(size_data_reliable=True, available_sizes=available_sizes)
        )
        assert p.size_data_reliable is True

    def test_unreliable_size_data_may_still_list_sizes(self):
        p = Product.model_validate(product_data(available_sizes=["M"]))
        assert p.size_data_reliable is False

    def test_size_data_reliable_must_be_boolean(self):
        with pytest.raises(ValidationError):
            Product.model_validate(product_data(size_data_reliable="yes", available_sizes=["M"]))

    def test_raw_embedding_vector_not_accepted(self):
        with pytest.raises(ValidationError, match="extra"):
            Product.model_validate(product_data(embedding=[0.1, 0.2]))

    def test_decimal_price_json_serialisation(self):
        p = Product.model_validate(product_data(price="19.90"))
        assert json.loads(p.model_dump_json())["price"] == "19.90"
        assert Product.model_validate_json(p.model_dump_json()).price == Decimal("19.90")

    def test_round_trip(self):
        p = Product.model_validate(
            product_data(
                description="wool",
                brand="B",
                colour="black",
                material="wool",
                collected_at="2026-10-01T08:00:00Z",
            )
        )
        assert Product.model_validate(p.model_dump()) == p
        assert Product.model_validate_json(p.model_dump_json()) == p


# ---------------------------------------------------------------------------
# CandidateOutfit / FeasibleCandidateOutfit
# ---------------------------------------------------------------------------


class TestCandidateOutfit:
    def test_three_item_outfit_without_compatibility_score(self):
        outfit = CandidateOutfit.model_validate({"outfit_id": "c1", "items": outfit_items()})
        assert outfit.compatibility_score is None
        assert {i.category for i in outfit.items} == {
            ProductCategory.TOP,
            ProductCategory.BOTTOM,
            ProductCategory.FOOTWEAR,
        }

    def test_optional_outerwear(self):
        outfit = CandidateOutfit.model_validate(
            {"outfit_id": "c1", "items": outfit_items(include_outerwear=True)}
        )
        assert len(outfit.items) == 4

    @pytest.mark.parametrize("missing", ["TOP", "BOTTOM", "FOOTWEAR"])
    def test_missing_required_role_rejected(self, missing):
        items = [i for i in outfit_items(include_outerwear=True) if i["category"] != missing]
        with pytest.raises(ValidationError, match=missing):
            CandidateOutfit.model_validate({"outfit_id": "c1", "items": items})

    def test_duplicate_role_rejected(self):
        items = outfit_items() + [product_data("TOP", product_id="p_top_2")]
        with pytest.raises(ValidationError, match="TOP"):
            CandidateOutfit.model_validate({"outfit_id": "c1", "items": items})

    def test_duplicate_product_id_rejected(self):
        items = outfit_items()
        items[1]["product_id"] = items[0]["product_id"]
        with pytest.raises(ValidationError, match="product_id"):
            CandidateOutfit.model_validate({"outfit_id": "c1", "items": items})

    @pytest.mark.parametrize("outfit_id", ["", "   "])
    def test_blank_outfit_id_rejected(self, outfit_id):
        with pytest.raises(ValidationError):
            CandidateOutfit.model_validate({"outfit_id": outfit_id, "items": outfit_items()})

    def test_outfit_id_any_non_empty_string(self):
        outfit = CandidateOutfit.model_validate({"outfit_id": "cand-7f3a", "items": outfit_items()})
        assert outfit.outfit_id == "cand-7f3a"

    @pytest.mark.parametrize("score", [0.0, 1.0, 0.42])
    def test_compatibility_score_in_range(self, score):
        outfit = CandidateOutfit.model_validate(
            {"outfit_id": "c1", "items": outfit_items(), "compatibility_score": score}
        )
        assert outfit.compatibility_score == score

    @pytest.mark.parametrize("score", [-0.01, 1.01])
    def test_compatibility_score_out_of_range_rejected(self, score):
        with pytest.raises(ValidationError):
            CandidateOutfit.model_validate(
                {"outfit_id": "c1", "items": outfit_items(), "compatibility_score": score}
            )

    def test_feasible_outfit_rejects_unavailable_purchase(self):
        items = outfit_items()
        items[0]["availability"] = False
        CandidateOutfit.model_validate({"outfit_id": "c1", "items": items})
        with pytest.raises(ValidationError, match="available"):
            FeasibleCandidateOutfit.model_validate({"outfit_id": "c1", "items": items})

    def test_feasible_outfit_accepts_user_owned_item(self):
        items = outfit_items()
        items[2] = product_data("FOOTWEAR", is_user_owned=True, product_url=None)
        outfit = FeasibleCandidateOutfit.model_validate({"outfit_id": "c1", "items": items})
        assert outfit.items[2].is_user_owned

    def test_round_trip(self):
        outfit = FeasibleCandidateOutfit.model_validate(
            {"outfit_id": "c1", "items": outfit_items(True)}
        )
        assert FeasibleCandidateOutfit.model_validate(outfit.model_dump()) == outfit
        assert FeasibleCandidateOutfit.model_validate_json(outfit.model_dump_json()) == outfit


# ---------------------------------------------------------------------------
# ScoredOutfit
# ---------------------------------------------------------------------------


def scored_data(**overrides):
    data = {
        "outfit_id": "c1",
        "items": outfit_items(),
        "compatibility_score": 0.8,
        "outfit_product_relevance": 0.5,
        "final_outfit_score": 0.65,
    }
    data.update(overrides)
    return data


class TestScoredOutfit:
    def test_valid(self):
        s = ScoredOutfit.model_validate(scored_data())
        assert s.compatibility_score == 0.8
        assert s.final_outfit_score == 0.65

    def test_compatibility_score_required(self):
        data = scored_data()
        del data["compatibility_score"]
        with pytest.raises(ValidationError):
            ScoredOutfit.model_validate(data)

    @pytest.mark.parametrize("score", [-0.1, 1.1])
    def test_compatibility_score_out_of_range_rejected(self, score):
        with pytest.raises(ValidationError):
            ScoredOutfit.model_validate(scored_data(compatibility_score=score))

    def test_negative_relevance_and_final_score_allowed(self):
        # ProductScore is cosine-based, so OutfitProductRelevance and FinalOutfitScore
        # can be negative (e.g. beta=0.5, compat=0.0, relevance=-0.4 -> -0.2).
        s = ScoredOutfit.model_validate(
            scored_data(
                compatibility_score=0.0, outfit_product_relevance=-0.4, final_outfit_score=-0.2
            )
        )
        assert s.final_outfit_score == -0.2

    @pytest.mark.parametrize("field", ["outfit_product_relevance", "final_outfit_score"])
    @pytest.mark.parametrize("value", [-1.01, 1.01])
    def test_scores_outside_minus_one_to_one_rejected(self, field, value):
        with pytest.raises(ValidationError):
            ScoredOutfit.model_validate(scored_data(**{field: value}))

    def test_round_trip(self):
        s = ScoredOutfit.model_validate(scored_data())
        assert ScoredOutfit.model_validate(s.model_dump()) == s
        assert ScoredOutfit.model_validate_json(s.model_dump_json()) == s


# ---------------------------------------------------------------------------
# RecommendationItem / RecommendationSet
# ---------------------------------------------------------------------------


class TestRecommendationItem:
    def test_valid(self):
        item = RecommendationItem.model_validate(recommendation_item_data())
        assert item.total_purchase_cost == Decimal("80.00")
        assert all(p.is_user_owned for p in item.existing_items)
        assert not any(p.is_user_owned for p in item.purchase_items)

    def test_total_purchase_cost_excludes_existing_items(self):
        # 50 + 30 purchased; the 80.00 user-owned footwear must not be counted.
        with pytest.raises(ValidationError, match="total_purchase_cost"):
            RecommendationItem.model_validate(
                {**recommendation_item_data(), "total_purchase_cost": "160.00"}
            )

    def test_existing_items_must_be_user_owned(self):
        data = recommendation_item_data()
        data["existing_items"][0]["is_user_owned"] = False
        data["existing_items"][0]["product_url"] = "https://example.com/f"
        with pytest.raises(ValidationError, match="existing_items"):
            RecommendationItem.model_validate(data)

    def test_purchase_items_must_not_be_user_owned(self):
        data = recommendation_item_data()
        data["purchase_items"][0]["is_user_owned"] = True
        with pytest.raises(ValidationError, match="purchase_items"):
            RecommendationItem.model_validate(data)

    def test_roles_validated_across_both_lists(self):
        data = recommendation_item_data()
        data["existing_items"] = []
        with pytest.raises(ValidationError, match="FOOTWEAR"):
            RecommendationItem.model_validate(data)

    def test_unavailable_purchase_item_rejected(self):
        data = recommendation_item_data()
        data["purchase_items"][1]["availability"] = False
        with pytest.raises(ValidationError, match="not available"):
            RecommendationItem.model_validate(data)

    def test_unavailable_existing_item_accepted(self):
        # Existing items are the user's own clothes, so stock availability does not apply.
        data = recommendation_item_data()
        data["existing_items"][0]["availability"] = False
        item = RecommendationItem.model_validate(data)
        assert item.existing_items[0].availability is False

    def test_all_purchase_items_available_passes(self):
        item = RecommendationItem.model_validate(recommendation_item_data())
        assert all(p.availability for p in item.purchase_items)

    def test_display_position_starts_at_one(self):
        with pytest.raises(ValidationError):
            RecommendationItem.model_validate(recommendation_item_data(position=0))

    def test_all_purchases_currency_must_match(self):
        data = recommendation_item_data()
        data["purchase_items"][0]["currency"] = "USD"
        with pytest.raises(ValidationError, match="currency"):
            RecommendationItem.model_validate(data)

    def test_decimal_json_serialisation(self):
        item = RecommendationItem.model_validate(recommendation_item_data())
        assert json.loads(item.model_dump_json())["total_purchase_cost"] == "80.00"

    def test_round_trip(self):
        item = RecommendationItem.model_validate(recommendation_item_data())
        assert RecommendationItem.model_validate(item.model_dump()) == item
        assert RecommendationItem.model_validate_json(item.model_dump_json()) == item


class TestRecommendationSet:
    def make(self, n):
        return {
            "recommendation_round": 1,
            "recommendations": [
                recommendation_item_data(f"o{i}", i, final_score=0.9 - 0.1 * i)
                for i in range(1, n + 1)
            ],
        }

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_up_to_three(self, n):
        assert len(RecommendationSet.model_validate(self.make(n)).recommendations) == n

    def test_more_than_three_rejected(self):
        with pytest.raises(ValidationError):
            RecommendationSet.model_validate(self.make(4))

    def test_empty_rejected(self):
        with pytest.raises(ValidationError):
            RecommendationSet.model_validate(self.make(0))

    def test_duplicate_display_position_rejected(self):
        data = self.make(3)
        data["recommendations"][2]["display_position"] = 2
        with pytest.raises(ValidationError, match="display_position"):
            RecommendationSet.model_validate(data)

    def test_positions_must_start_at_one(self):
        data = self.make(2)
        data["recommendations"][0]["display_position"] = 2
        data["recommendations"][1]["display_position"] = 3
        with pytest.raises(ValidationError, match="display_position"):
            RecommendationSet.model_validate(data)

    def test_duplicate_outfit_id_rejected(self):
        data = self.make(2)
        data["recommendations"][1]["outfit_id"] = "o1"
        with pytest.raises(ValidationError, match="outfit_id"):
            RecommendationSet.model_validate(data)

    def test_scores_must_be_non_increasing_by_position(self):
        data = self.make(2)
        data["recommendations"][1]["final_outfit_score"] = 0.95
        with pytest.raises(ValidationError, match="final_outfit_score"):
            RecommendationSet.model_validate(data)

    def test_equal_scores_allowed(self):
        data = self.make(2)
        data["recommendations"][1]["final_outfit_score"] = data["recommendations"][0][
            "final_outfit_score"
        ]
        RecommendationSet.model_validate(data)

    def test_round_trip(self):
        rs = RecommendationSet.model_validate(self.make(3))
        assert RecommendationSet.model_validate(rs.model_dump()) == rs
        assert RecommendationSet.model_validate_json(rs.model_dump_json()) == rs


# ---------------------------------------------------------------------------
# ConversationalRecommendationState
# ---------------------------------------------------------------------------


class TestConversationalRecommendationState:
    def test_appendix_a_state(self):
        s = ConversationalRecommendationState.model_validate(state_data())
        assert s.session_status is SessionStatus.AWAITING_CONFIRMATION
        assert s.selected_outfit_id is None
        assert s.current_request.budget == Decimal("400")
        assert s.current_request.excluded_items == ["long coat"]

    def test_proposal_field_names_preserved(self):
        dumped = ConversationalRecommendationState.model_validate(state_data()).model_dump(
            mode="json"
        )
        assert set(dumped) == {
            "session_id",
            "request_version",
            "current_request",
            "recommendation_round",
            "last_recommended_outfit_ids",
            "selected_outfit_id",
            "session_status",
        }

    def test_confirmed_state(self):
        s = ConversationalRecommendationState.model_validate(
            state_data(selected_outfit_id="outfit_202", session_status="CONFIRMED")
        )
        assert s.session_status is SessionStatus.CONFIRMED

    def test_confirmed_requires_selected_outfit_id(self):
        with pytest.raises(ValidationError, match="selected_outfit_id"):
            ConversationalRecommendationState.model_validate(state_data(session_status="CONFIRMED"))

    def test_awaiting_confirmation_must_not_have_selection(self):
        with pytest.raises(ValidationError, match="selected_outfit_id"):
            ConversationalRecommendationState.model_validate(
                state_data(selected_outfit_id="outfit_201")
            )

    def test_selected_id_must_be_in_last_recommended(self):
        with pytest.raises(ValidationError, match="last_recommended_outfit_ids"):
            ConversationalRecommendationState.model_validate(
                state_data(selected_outfit_id="outfit_999", session_status="CONFIRMED")
            )

    def test_illegal_status_rejected(self):
        with pytest.raises(ValidationError):
            ConversationalRecommendationState.model_validate(state_data(session_status="DONE"))

    def test_more_than_three_last_ids_rejected(self):
        with pytest.raises(ValidationError):
            ConversationalRecommendationState.model_validate(
                state_data(last_recommended_outfit_ids=["a", "b", "c", "d"])
            )

    def test_duplicate_last_ids_rejected(self):
        with pytest.raises(ValidationError):
            ConversationalRecommendationState.model_validate(
                state_data(last_recommended_outfit_ids=["a", "a"])
            )

    @pytest.mark.parametrize("field,value", [("request_version", 0), ("recommendation_round", -1)])
    def test_counters_lower_bounds(self, field, value):
        with pytest.raises(ValidationError):
            ConversationalRecommendationState.model_validate(state_data(**{field: value}))

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="extra"):
            ConversationalRecommendationState.model_validate(state_data(status="x"))

    def test_assignment_is_validated(self):
        s = ConversationalRecommendationState.model_validate(state_data())
        with pytest.raises(ValidationError):
            s.request_version = 0

    def test_default_lists_not_shared(self):
        data = state_data()
        del data["last_recommended_outfit_ids"]
        a = ConversationalRecommendationState.model_validate({**data, "recommendation_round": 0})
        b = ConversationalRecommendationState.model_validate({**data, "recommendation_round": 0})
        a.last_recommended_outfit_ids.append("x")
        assert b.last_recommended_outfit_ids == []

    def test_round_trip(self):
        s = ConversationalRecommendationState.model_validate(state_data())
        assert ConversationalRecommendationState.model_validate(s.model_dump()) == s
        assert ConversationalRecommendationState.model_validate_json(s.model_dump_json()) == s


# ---------------------------------------------------------------------------
# Stage 14: PreviewItemReference / PreviewRequest / PreviewResult / ConfirmedOutfit
# ---------------------------------------------------------------------------


def preview_item(product_id="p_top", category="TOP", **overrides):
    data = {
        "product_id": product_id,
        "category": category,
        "image_reference": f"https://example.com/{product_id}.jpg",
        "title": f"Mock {category.lower()}",
        "description": None,
        "is_user_owned": False,
    }
    data.update(overrides)
    return data


def preview_items():
    return [
        preview_item("p_top", "TOP", description="Wool knit sweater"),
        preview_item("p_bottom", "BOTTOM", is_user_owned=True, image_reference=None),
        preview_item("p_footwear", "FOOTWEAR"),
    ]


def preview_request(**overrides):
    data = {"selected_outfit_id": "o1", "items": preview_items(), "preview_mode": "PRODUCT_CARDS"}
    data.update(overrides)
    return data


class TestPreviewItemReference:
    def test_valid(self):
        ref = PreviewItemReference.model_validate(preview_item(description="Black coat"))
        assert ref.category is ProductCategory.TOP
        assert ref.image_reference == "https://example.com/p_top.jpg"
        assert ref.title == "Mock top"
        assert ref.description == "Black coat"
        assert ref.is_user_owned is False

    def test_image_reference_may_be_null(self):
        assert (
            PreviewItemReference.model_validate(preview_item(image_reference=None)).image_reference
            is None
        )

    def test_description_may_be_null(self):
        ref = PreviewItemReference.model_validate(preview_item(description=None))
        assert ref.description is None
        assert ref.title == "Mock top"  # title is not copied into description

    def test_description_omitted_defaults_to_null(self):
        data = preview_item()
        del data["description"]
        assert PreviewItemReference.model_validate(data).description is None

    def test_title_missing_rejected(self):
        data = preview_item()
        del data["title"]
        with pytest.raises(ValidationError, match="title"):
            PreviewItemReference.model_validate(data)

    @pytest.mark.parametrize("field", ["title", "description", "product_id"])
    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_text_rejected(self, field, value):
        with pytest.raises(ValidationError, match=field):
            PreviewItemReference.model_validate(preview_item(**{field: value}))

    def test_built_from_product_without_inventing_description(self):
        product = Product.model_validate(product_data("BOTTOM", description=None))
        ref = PreviewItemReference.model_validate(
            {
                "product_id": product.product_id,
                "category": product.category,
                "image_reference": product.image_url,
                "title": product.title,
                "description": product.description,
                "is_user_owned": product.is_user_owned,
            }
        )
        assert ref.title == product.title
        assert ref.description is None
        assert ref.image_reference == product.image_url

    def test_raw_image_bytes_rejected(self):
        with pytest.raises(ValidationError, match="image_reference"):
            PreviewItemReference.model_validate(preview_item(image_reference=b"\x89PNG\r\n"))

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError, match="extra"):
            PreviewItemReference.model_validate(preview_item(image_bytes="..."))

    def test_round_trip(self):
        for description in (None, "Black coat"):
            ref = PreviewItemReference.model_validate(preview_item(description=description))
            assert PreviewItemReference.model_validate(ref.model_dump()) == ref
            assert PreviewItemReference.model_validate_json(ref.model_dump_json()) == ref


class TestPreviewRequest:
    def test_portrait_with_consent(self):
        req = PreviewRequest.model_validate(
            preview_request(
                preview_mode="PORTRAIT",
                user_portrait="tmp/uploads/portrait.jpg",
                portrait_consent=True,
            )
        )
        assert req.preview_mode is PreviewMode.PORTRAIT
        assert req.portrait_consent is True

    def test_portrait_mode_without_portrait_rejected(self):
        with pytest.raises(ValidationError, match="user_portrait"):
            PreviewRequest.model_validate(
                preview_request(preview_mode="PORTRAIT", portrait_consent=True)
            )

    def test_portrait_without_consent_rejected(self):
        with pytest.raises(ValidationError, match="portrait_consent"):
            PreviewRequest.model_validate(
                preview_request(
                    preview_mode="PORTRAIT", user_portrait="p.jpg", portrait_consent=False
                )
            )

    def test_portrait_with_consent_omitted_rejected(self):
        with pytest.raises(ValidationError, match="portrait_consent"):
            PreviewRequest.model_validate(
                preview_request(preview_mode="PORTRAIT", user_portrait="p.jpg")
            )

    @pytest.mark.parametrize("mode", ["MANNEQUIN", "FLAT_LAY"])
    def test_portrait_without_consent_rejected_in_any_mode(self, mode):
        with pytest.raises(ValidationError, match="portrait_consent"):
            PreviewRequest.model_validate(preview_request(preview_mode=mode, user_portrait="p.jpg"))

    @pytest.mark.parametrize("mode", ["PRODUCT_CARDS", "MANNEQUIN", "FLAT_LAY"])
    def test_modes_without_portrait(self, mode):
        req = PreviewRequest.model_validate(preview_request(preview_mode=mode))
        assert req.user_portrait is None
        assert req.portrait_consent is False

    def test_consent_without_portrait_allowed(self):
        req = PreviewRequest.model_validate(
            preview_request(preview_mode="MANNEQUIN", portrait_consent=True)
        )
        assert req.user_portrait is None

    def test_consent_must_be_boolean(self):
        with pytest.raises(ValidationError):
            PreviewRequest.model_validate(
                preview_request(
                    user_portrait="p.jpg", portrait_consent="yes", preview_mode="PORTRAIT"
                )
            )

    def test_items_keep_title_and_optional_description(self):
        req = PreviewRequest.model_validate(preview_request())
        assert [(i.title, i.description) for i in req.items] == [
            ("Mock top", "Wool knit sweater"),
            ("Mock bottom", None),
            ("Mock footwear", None),
        ]

    def test_empty_items_rejected(self):
        with pytest.raises(ValidationError, match="items"):
            PreviewRequest.model_validate(preview_request(items=[]))

    def test_duplicate_product_id_rejected(self):
        items = preview_items() + [preview_item("p_top", "OUTERWEAR")]
        with pytest.raises(ValidationError, match="p_top"):
            PreviewRequest.model_validate(preview_request(items=items))

    def test_item_order_preserved(self):
        items = [
            preview_item("p_shoes", "FOOTWEAR"),
            preview_item("p_coat", "OUTERWEAR"),
            preview_item("p_shirt", "TOP"),
        ]
        req = PreviewRequest.model_validate(preview_request(items=items))
        assert [i.product_id for i in req.items] == ["p_shoes", "p_coat", "p_shirt"]

    def test_raw_portrait_bytes_rejected(self):
        with pytest.raises(ValidationError, match="user_portrait"):
            PreviewRequest.model_validate(
                preview_request(
                    preview_mode="PORTRAIT", user_portrait=b"\xff\xd8\xff", portrait_consent=True
                )
            )

    @pytest.mark.parametrize("extra", ["product_images", "product_descriptions", "api_key"])
    def test_extra_and_removed_fields_rejected(self, extra):
        with pytest.raises(ValidationError, match="extra"):
            PreviewRequest.model_validate(preview_request(**{extra: ["x"]}))

    def test_illegal_mode_rejected(self):
        with pytest.raises(ValidationError):
            PreviewRequest.model_validate(preview_request(preview_mode="VIRTUAL_FIT"))

    def test_round_trip(self):
        req = PreviewRequest.model_validate(
            preview_request(
                preview_mode="PORTRAIT", user_portrait="tmp/p.jpg", portrait_consent=True
            )
        )
        assert PreviewRequest.model_validate(req.model_dump()) == req
        assert PreviewRequest.model_validate_json(req.model_dump_json()) == req

    def test_input_not_modified(self):
        data = preview_request(preview_mode="FLAT_LAY")
        snapshot = json.dumps(data, sort_keys=True)
        req = PreviewRequest.model_validate(data)
        assert json.dumps(data, sort_keys=True) == snapshot
        assert data["items"][0] is not req.items[0]


GENERATIVE_MODES = ["PORTRAIT", "MANNEQUIN", "FLAT_LAY"]


def preview_result(**overrides):
    data = {
        "selected_outfit_id": "o1",
        "status": "SUCCESS",
        "requested_mode": "FLAT_LAY",
        "delivered_mode": "FLAT_LAY",
        "output_reference": "tmp/previews/o1.png",
        "provider_name": "mock",
        "model_name": None,
        "disclaimer": "Visual reference only.",
        "warnings": [],
        "generation_time_seconds": 0.5,
    }
    data.update(overrides)
    return data


class TestPreviewResultSuccess:
    @pytest.mark.parametrize("mode", GENERATIVE_MODES)
    def test_generative_success_with_output(self, mode):
        result = PreviewResult.model_validate(
            preview_result(requested_mode=mode, delivered_mode=mode)
        )
        assert result.status is PreviewStatus.SUCCESS
        assert result.delivered_mode is PreviewMode(mode)

    @pytest.mark.parametrize("mode", GENERATIVE_MODES)
    def test_generative_success_without_output_rejected(self, mode):
        with pytest.raises(ValidationError, match="output_reference"):
            PreviewResult.model_validate(
                preview_result(requested_mode=mode, delivered_mode=mode, output_reference=None)
            )

    def test_product_cards_success_without_output(self):
        result = PreviewResult.model_validate(
            preview_result(
                requested_mode="PRODUCT_CARDS",
                delivered_mode="PRODUCT_CARDS",
                output_reference=None,
                provider_name="product_cards",
            )
        )
        assert result.output_reference is None

    def test_product_cards_success_with_output_allowed(self):
        assert PreviewResult.model_validate(
            preview_result(
                requested_mode="PRODUCT_CARDS",
                delivered_mode="PRODUCT_CARDS",
                output_reference="tmp/previews/cards.html",
            )
        ).output_reference

    def test_success_without_delivered_mode_rejected(self):
        with pytest.raises(ValidationError, match="delivered_mode"):
            PreviewResult.model_validate(preview_result(delivered_mode=None))

    def test_success_with_different_delivered_mode_rejected(self):
        with pytest.raises(ValidationError, match="delivered_mode"):
            PreviewResult.model_validate(preview_result(requested_mode="PORTRAIT"))


class TestPreviewResultFallback:
    def test_fallback_portrait_to_product_cards_without_output(self):
        result = PreviewResult.model_validate(
            preview_result(
                status="FALLBACK",
                requested_mode="PORTRAIT",
                delivered_mode="PRODUCT_CARDS",
                output_reference=None,
                provider_name="product_card_fallback",
                warnings=["Preview generation unavailable; showing product cards."],
            )
        )
        assert result.requested_mode is PreviewMode.PORTRAIT
        assert result.delivered_mode is PreviewMode.PRODUCT_CARDS

    def test_fallback_to_generative_mode_with_output(self):
        result = PreviewResult.model_validate(
            preview_result(status="FALLBACK", requested_mode="PORTRAIT", delivered_mode="MANNEQUIN")
        )
        assert result.delivered_mode is PreviewMode.MANNEQUIN

    @pytest.mark.parametrize("mode", GENERATIVE_MODES)
    def test_fallback_to_generative_mode_without_output_rejected(self, mode):
        with pytest.raises(ValidationError, match="output_reference"):
            PreviewResult.model_validate(
                preview_result(
                    status="FALLBACK",
                    requested_mode="PORTRAIT",
                    delivered_mode=mode,
                    output_reference=None,
                )
            )

    def test_fallback_without_delivered_mode_rejected(self):
        with pytest.raises(ValidationError, match="delivered_mode"):
            PreviewResult.model_validate(
                preview_result(status="FALLBACK", delivered_mode=None, output_reference=None)
            )


class TestPreviewResultFailed:
    def test_failed_without_mode_or_output(self):
        result = PreviewResult.model_validate(
            preview_result(
                status="FAILED",
                delivered_mode=None,
                output_reference=None,
                generation_time_seconds=None,
                warnings=["provider error"],
            )
        )
        assert result.status is PreviewStatus.FAILED
        assert result.delivered_mode is None
        assert result.output_reference is None

    def test_delivered_mode_defaults_to_null(self):
        data = preview_result(status="FAILED", output_reference=None)
        del data["delivered_mode"]
        assert PreviewResult.model_validate(data).delivered_mode is None

    def test_failed_with_delivered_mode_rejected(self):
        with pytest.raises(ValidationError, match="delivered_mode"):
            PreviewResult.model_validate(
                preview_result(
                    status="FAILED", delivered_mode="PRODUCT_CARDS", output_reference=None
                )
            )

    def test_failed_with_output_rejected(self):
        with pytest.raises(ValidationError, match="output_reference"):
            PreviewResult.model_validate(preview_result(status="FAILED", delivered_mode=None))


class TestPreviewResultFields:
    @pytest.mark.parametrize("field", ["provider_name", "disclaimer"])
    @pytest.mark.parametrize("value", ["", "  "])
    def test_required_text_not_blank(self, field, value):
        with pytest.raises(ValidationError, match=field):
            PreviewResult.model_validate(preview_result(**{field: value}))

    def test_disclaimer_required(self):
        data = preview_result()
        del data["disclaimer"]
        with pytest.raises(ValidationError, match="disclaimer"):
            PreviewResult.model_validate(data)

    @pytest.mark.parametrize("seconds", [-0.001, float("nan"), float("inf"), float("-inf")])
    def test_invalid_generation_time_rejected(self, seconds):
        with pytest.raises(ValidationError, match="generation_time_seconds"):
            PreviewResult.model_validate(preview_result(generation_time_seconds=seconds))

    def test_zero_and_null_generation_time_allowed(self):
        assert (
            PreviewResult.model_validate(
                preview_result(generation_time_seconds=0)
            ).generation_time_seconds
            == 0
        )
        assert (
            PreviewResult.model_validate(
                preview_result(generation_time_seconds=None)
            ).generation_time_seconds
            is None
        )

    def test_blank_warning_rejected(self):
        with pytest.raises(ValidationError, match="warnings"):
            PreviewResult.model_validate(preview_result(warnings=["colours may differ", " "]))

    def test_raw_output_bytes_rejected(self):
        with pytest.raises(ValidationError, match="output_reference"):
            PreviewResult.model_validate(preview_result(output_reference=b"\x89PNG"))

    def test_default_warnings_not_shared(self):
        data = preview_result()
        del data["warnings"]
        a = PreviewResult.model_validate(data)
        b = PreviewResult.model_validate(data)
        a.warnings.append("x")
        assert b.warnings == []

    @pytest.mark.parametrize(
        "overrides",
        [
            {"model_name": "some-model", "warnings": ["colours may differ"]},
            {"status": "FAILED", "delivered_mode": None, "output_reference": None},
            {
                "status": "FALLBACK",
                "requested_mode": "PORTRAIT",
                "delivered_mode": "PRODUCT_CARDS",
                "output_reference": None,
            },
        ],
    )
    def test_round_trip(self, overrides):
        result = PreviewResult.model_validate(preview_result(**overrides))
        assert PreviewResult.model_validate(result.model_dump()) == result
        assert PreviewResult.model_validate_json(result.model_dump_json()) == result

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError, match="extra"):
            PreviewResult.model_validate(preview_result(image_bytes="..."))


class TestConfirmedOutfit:
    def test_valid(self):
        c = ConfirmedOutfit.model_validate(
            {
                "session_id": "session_001",
                "selected_outfit_id": "o1",
                "recommendation": recommendation_item_data("o1"),
            }
        )
        assert c.recommendation.outfit_id == c.selected_outfit_id

    def test_selected_id_must_match_recommendation(self):
        with pytest.raises(ValidationError, match="selected_outfit_id"):
            ConfirmedOutfit.model_validate(
                {
                    "session_id": "session_001",
                    "selected_outfit_id": "o2",
                    "recommendation": recommendation_item_data("o1"),
                }
            )

    def test_round_trip(self):
        c = ConfirmedOutfit.model_validate(
            {
                "session_id": "s",
                "selected_outfit_id": "o1",
                "recommendation": recommendation_item_data("o1"),
            }
        )
        assert ConfirmedOutfit.model_validate(c.model_dump()) == c
        assert ConfirmedOutfit.model_validate_json(c.model_dump_json()) == c
