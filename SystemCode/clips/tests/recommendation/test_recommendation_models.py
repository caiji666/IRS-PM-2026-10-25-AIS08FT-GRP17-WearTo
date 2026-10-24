"""Tests for the proposal data contract models (Module 1)."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from travel_agent.recommendation.student5.models import (
    CandidateOutfit,
    ConfirmedOutfit,
    ConversationalRecommendationState,
    FeasibleCandidateOutfit,
    PreviewMode,
    PreviewRequest,
    Product,
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

    def test_end_date_before_start_date_rejected(self):
        with pytest.raises(ValidationError, match="end_date"):
            TripShoppingRequest.model_validate(
                request_data(start_date="2026-12-27", end_date="2026-12-20")
            )

    @pytest.mark.parametrize("budget", [0])
    def test_non_positive_budget_rejected(self, budget):
        with pytest.raises(ValidationError):
            TripShoppingRequest.model_validate(request_data(budget=budget))

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="extra"):
            TripShoppingRequest.model_validate(request_data(weather_class="COLD"))


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

    @pytest.mark.parametrize("intent", ["MODIFY", ""])
    def test_illegal_intent_rejected(self, intent):
        with pytest.raises(ValidationError):
            TripShoppingRequestPatch.model_validate({"intent": intent})

    @pytest.mark.parametrize(
        "operations,field",
        [({"set": {"style": ["casual"]}, "add": {"style": ["minimalist"]}}, "style")],
        ids=["set_add_style"],
    )
    def test_overlapping_operations_rejected(self, operations, field):
        with pytest.raises(ValidationError, match=f"more than one.*{field}"):
            TripShoppingRequestPatch.model_validate({"intent": "MODIFY_REQUIREMENTS", **operations})

    def test_confirm_outfit_patch_must_not_carry_changes(self):
        with pytest.raises(ValidationError):
            TripShoppingRequestPatch.model_validate(
                {"intent": "CONFIRM_OUTFIT", "set": {"budget": 300}}
            )

    def test_add_on_non_list_field_rejected(self):
        with pytest.raises(ValidationError):
            TripShoppingRequestPatch.model_validate(
                {"intent": "MODIFY_REQUIREMENTS", "add": {"budget": [100]}}
            )


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------


class TestProduct:
    @pytest.mark.parametrize("score", [-1.0, 0.0, 1.0])
    def test_product_score_boundaries_accepted(self, score):
        assert Product.model_validate(product_data(product_score=score)).product_score == score

    @pytest.mark.parametrize("score", [-1.0001, 1.0001, float("nan")])
    def test_product_score_out_of_range_rejected(self, score):
        with pytest.raises(ValidationError):
            Product.model_validate(product_data(product_score=score))

    def test_availability_must_be_explicit_boolean(self):
        data = product_data()
        del data["availability"]
        with pytest.raises(ValidationError):
            Product.model_validate(data)
        with pytest.raises(ValidationError):
            Product.model_validate(product_data(availability="in stock"))

    def test_purchasable_product_requires_product_url(self):
        with pytest.raises(ValidationError, match="product_url"):
            Product.model_validate(product_data(product_url=None))

    def test_reliable_size_data_requires_size_list(self):
        with pytest.raises(ValidationError, match="available_sizes"):
            Product.model_validate(product_data(size_data_reliable=True))

    def test_raw_embedding_vector_not_accepted(self):
        with pytest.raises(ValidationError, match="extra"):
            Product.model_validate(product_data(embedding=[0.1, 0.2]))


# ---------------------------------------------------------------------------
# CandidateOutfit / FeasibleCandidateOutfit
# ---------------------------------------------------------------------------


class TestCandidateOutfit:
    def test_optional_outerwear(self):
        outfit = CandidateOutfit.model_validate(
            {"outfit_id": "c1", "items": outfit_items(include_outerwear=True)}
        )
        assert len(outfit.items) == 4

    @pytest.mark.parametrize("missing", ["TOP"])
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

    def test_feasible_outfit_rejects_unavailable_purchase(self):
        items = outfit_items()
        items[0]["availability"] = False
        CandidateOutfit.model_validate({"outfit_id": "c1", "items": items})
        with pytest.raises(ValidationError, match="available"):
            FeasibleCandidateOutfit.model_validate({"outfit_id": "c1", "items": items})


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
    @pytest.mark.parametrize("field", ["outfit_product_relevance", "final_outfit_score"])
    @pytest.mark.parametrize("value", [1.01])
    def test_scores_outside_minus_one_to_one_rejected(self, field, value):
        with pytest.raises(ValidationError):
            ScoredOutfit.model_validate(scored_data(**{field: value}))


# ---------------------------------------------------------------------------
# RecommendationItem / RecommendationSet
# ---------------------------------------------------------------------------


class TestRecommendationItem:
    def test_total_purchase_cost_excludes_existing_items(self):
        # 50 + 30 purchased; the 80.00 user-owned footwear must not be counted.
        with pytest.raises(ValidationError, match="total_purchase_cost"):
            RecommendationItem.model_validate(
                {**recommendation_item_data(), "total_purchase_cost": "160.00"}
            )

    @pytest.mark.parametrize(
        "field, owned",
        [("existing_items", False), ("purchase_items", True)],
        ids=["existing_not_owned", "purchase_owned"],
    )
    def test_ownership_must_match_item_list(self, field, owned):
        data = recommendation_item_data()
        data[field][0]["is_user_owned"] = owned
        data[field][0]["product_url"] = "https://example.com/f"
        with pytest.raises(ValidationError, match=field):
            RecommendationItem.model_validate(data)

    def test_all_purchases_currency_must_match(self):
        data = recommendation_item_data()
        data["purchase_items"][0]["currency"] = "USD"
        with pytest.raises(ValidationError, match="currency"):
            RecommendationItem.model_validate(data)

    def test_unavailable_purchase_item_rejected(self):
        data = recommendation_item_data()
        data["purchase_items"][1]["availability"] = False
        with pytest.raises(ValidationError, match="not available"):
            RecommendationItem.model_validate(data)


class TestRecommendationSet:
    def make(self, n):
        return {
            "recommendation_round": 1,
            "recommendations": [
                recommendation_item_data(f"o{i}", i, final_score=0.9 - 0.1 * i)
                for i in range(1, n + 1)
            ],
        }

    def test_more_than_three_rejected(self):
        with pytest.raises(ValidationError):
            RecommendationSet.model_validate(self.make(4))

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

    def test_duplicate_display_position_rejected(self):
        data = self.make(3)
        data["recommendations"][2]["display_position"] = 2
        with pytest.raises(ValidationError, match="display_position"):
            RecommendationSet.model_validate(data)


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

    def test_duplicate_last_ids_rejected(self):
        with pytest.raises(ValidationError):
            ConversationalRecommendationState.model_validate(
                state_data(last_recommended_outfit_ids=["a", "a"])
            )


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
    data = {"selected_outfit_id": "o1", "items": preview_items(), "preview_mode": "FLAT_LAY"}
    data.update(overrides)
    return data


class TestPreviewRequest:
    def test_only_flat_lay_and_portrait_are_user_selectable_modes(self):
        assert {mode.value for mode in PreviewMode} == {"FLAT_LAY", "PORTRAIT"}

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

    @pytest.mark.parametrize("mode", ["FLAT_LAY"])
    def test_portrait_without_consent_rejected_in_any_mode(self, mode):
        with pytest.raises(ValidationError, match="portrait_consent"):
            PreviewRequest.model_validate(preview_request(preview_mode=mode, user_portrait="p.jpg"))

    def test_consent_must_be_boolean(self):
        with pytest.raises(ValidationError):
            PreviewRequest.model_validate(
                preview_request(
                    user_portrait="p.jpg", portrait_consent="yes", preview_mode="PORTRAIT"
                )
            )

    def test_duplicate_product_id_rejected(self):
        items = preview_items() + [preview_item("p_top", "OUTERWEAR")]
        with pytest.raises(ValidationError, match="p_top"):
            PreviewRequest.model_validate(preview_request(items=items))


GENERATIVE_MODES = ["PORTRAIT", "FLAT_LAY"]


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


class TestConfirmedOutfit:
    def test_selected_id_must_match_recommendation(self):
        with pytest.raises(ValidationError, match="selected_outfit_id"):
            ConfirmedOutfit.model_validate(
                {
                    "session_id": "session_001",
                    "selected_outfit_id": "o2",
                    "recommendation": recommendation_item_data("o1"),
                }
            )
