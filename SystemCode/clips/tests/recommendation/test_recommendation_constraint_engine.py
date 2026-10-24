"""Tests for Stage 10 Hard Constraint Filtering (Module 2)."""

from decimal import Decimal

import pytest

from travel_agent.recommendation.student5.constraint_engine import (
    ConstraintFilterResult,
    ConstraintType,
    CurrencyMismatchError,
    RejectedOutfit,
    filter_candidate_outfits,
    purchase_total,
)
from travel_agent.recommendation.student5.models import (
    CandidateOutfit,
    TripShoppingRequest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_request(budget="500", user_size=None, currency="SGD"):
    return TripShoppingRequest.model_validate(
        {
            "destination": "Seoul",
            "start_date": "2026-12-20",
            "end_date": "2026-12-27",
            "budget": budget,
            "currency": currency,
            "user_size": user_size,
        }
    )


def product(category, price="50.00", **overrides):
    data = {
        "product_id": f"p_{category.lower()}",
        "source": "mock_catalogue",
        "title": f"Mock {category.lower()}",
        "product_url": f"https://example.com/{category.lower()}",
        "price": price,
        "currency": "SGD",
        "availability": True,
        "category": category,
        "product_score": 0.5,
    }
    if overrides.get("is_user_owned"):
        data["product_url"] = None
    data.update(overrides)
    return data


def make_outfit(outfit_id="c1", top=None, bottom=None, footwear=None, outerwear=None):
    items = [
        product("TOP", **(top or {})),
        product("BOTTOM", **(bottom or {})),
        product("FOOTWEAR", **(footwear or {})),
    ]
    if outerwear is not None:
        items.append(product("OUTERWEAR", **outerwear))
    return CandidateOutfit.model_validate({"outfit_id": outfit_id, "items": items})


def only_reason_types(rejected: RejectedOutfit):
    return [r.constraint for r in rejected.reasons]


# ---------------------------------------------------------------------------
# A. Budget
# ---------------------------------------------------------------------------


class TestBudget:
    def test_below_budget_passes(self):
        result = filter_candidate_outfits(make_request("500"), [make_outfit()])  # 150 total
        assert result.feasible_count == 1
        assert result.rejected_outfits == []

    def test_exactly_budget_passes(self):
        result = filter_candidate_outfits(make_request("150.00"), [make_outfit()])
        assert result.feasible_count == 1

    def test_over_budget_rejected(self):
        result = filter_candidate_outfits(make_request("149.99"), [make_outfit()])
        assert result.feasible_outfits == []
        [rejected] = result.rejected_outfits
        assert only_reason_types(rejected) == [ConstraintType.BUDGET]
        assert rejected.total_purchase_cost == Decimal("150.00")
        assert "149.99" in rejected.reasons[0].message

    def test_user_owned_price_not_counted(self):
        outfit = make_outfit(footwear={"is_user_owned": True, "price": "900.00"})
        assert purchase_total(outfit) == Decimal("100.00")
        result = filter_candidate_outfits(make_request("100"), [outfit])
        assert result.feasible_count == 1


# ---------------------------------------------------------------------------
# B. Availability
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_unavailable_purchase_item_rejected(self):
        result = filter_candidate_outfits(
            make_request(), [make_outfit(bottom={"availability": False})]
        )
        [rejected] = result.rejected_outfits
        assert only_reason_types(rejected) == [ConstraintType.AVAILABILITY]
        assert rejected.reasons[0].product_ids == ["p_bottom"]

    def test_unavailable_user_owned_item_passes(self):
        outfit = make_outfit(bottom={"is_user_owned": True, "availability": False})
        result = filter_candidate_outfits(make_request(), [outfit])
        assert result.feasible_count == 1
        assert result.feasible_outfits[0].items[1].availability is False


# ---------------------------------------------------------------------------
# C. Size (PROVISIONAL single user_size)
# ---------------------------------------------------------------------------


def sizes(*values, reliable=True):
    return {"available_sizes": list(values), "size_data_reliable": reliable}


class TestSize:
    def test_unreliable_size_data_not_treated_as_mismatch(self):
        # Sizes are listed but flagged unreliable: the item is skipped, not rejected.
        outfit = make_outfit(top=sizes("S", reliable=False), bottom=sizes(reliable=False))
        assert filter_candidate_outfits(make_request(user_size="M"), [outfit]).feasible_count == 1

    @pytest.mark.parametrize(
        "user_size,available",
        [
            ("m", ["M"]),
            ("  M ", ["M"]),
        ],
    )
    def test_case_and_whitespace_insensitive(self, user_size, available):
        outfit = make_outfit(top=sizes(*available))
        result = filter_candidate_outfits(make_request(user_size=user_size), [outfit])
        assert result.feasible_count == 1

    def test_exact_match_not_substring(self):
        outfit = make_outfit(top=sizes("XL"))
        result = filter_candidate_outfits(make_request(user_size="L"), [outfit])
        assert result.feasible_count == 0

    def test_mismatching_size_rejected(self):
        outfit = make_outfit(top=sizes("S"), bottom=sizes("M"))
        [rejected] = filter_candidate_outfits(
            make_request(user_size="M"), [outfit]
        ).rejected_outfits
        assert only_reason_types(rejected) == [ConstraintType.SIZE]
        assert rejected.reasons[0].product_ids == ["p_top"]


# ---------------------------------------------------------------------------
# D. Currency
# ---------------------------------------------------------------------------


class TestCurrency:
    def test_purchase_currency_mismatch_fails_loudly(self):
        outfit = make_outfit("c7", bottom={"currency": "USD"})
        with pytest.raises(CurrencyMismatchError) as exc:
            filter_candidate_outfits(make_request(currency="SGD"), [outfit])
        message = str(exc.value)
        for expected in ("c7", "p_bottom", "USD", "SGD"):
            assert expected in message


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


class TestFilterResult:
    def test_all_violations_collected(self):
        outfit = make_outfit(
            "bad",
            top={"price": "400.00", **sizes("S")},
            bottom={"availability": False},
        )
        [rejected] = filter_candidate_outfits(
            make_request("100", user_size="M"), [outfit]
        ).rejected_outfits
        assert rejected.outfit_id == "bad"
        assert only_reason_types(rejected) == [
            ConstraintType.BUDGET,
            ConstraintType.AVAILABILITY,
            ConstraintType.SIZE,
        ]
        assert rejected.total_purchase_cost == Decimal("500.00")

    def test_all_candidates_filtered_returns_conflict_summary(self):
        candidates = [
            make_outfit("o1", top={"price": "999"}),
            make_outfit("o2", top={"availability": False}),
            make_outfit("o3", top={"price": "999", "availability": False}),
        ]
        result = filter_candidate_outfits(make_request("200"), candidates)
        assert isinstance(result, ConstraintFilterResult)
        assert result.feasible_outfits == []
        assert result.total_candidates == 3
        assert result.feasible_count == 0
        assert result.rejected_count == 3
        assert result.conflict_summary == {
            ConstraintType.BUDGET: 2,
            ConstraintType.AVAILABILITY: 2,
            ConstraintType.SIZE: 0,
        }

    def test_duplicate_outfit_id_in_candidates_fails(self):
        candidates = [make_outfit("dup"), make_outfit("ok"), make_outfit("dup")]
        with pytest.raises(ValueError, match="dup"):
            filter_candidate_outfits(make_request(), candidates)


# ---------------------------------------------------------------------------
# Structured conflict explanation (no feasible outfit)
# ---------------------------------------------------------------------------


class TestConflictExplanation:
    def test_min_budget_ignores_outfits_failing_other_constraints(self):
        candidates = [
            make_outfit("cheap_but_unavailable", top={"price": "60.00", "availability": False}),
            make_outfit("budget_only", top={"price": "300.00"}),
        ]
        conflict = filter_candidate_outfits(make_request("150"), candidates).conflict
        assert conflict.violated_constraints == [ConstraintType.BUDGET, ConstraintType.AVAILABILITY]
        assert conflict.min_budget_to_make_feasible == Decimal("400.00")

    def test_min_budget_none_when_budget_alone_cannot_help(self):
        candidates = [
            make_outfit("o1", top={"availability": False}),
            make_outfit("o2", top=sizes("S")),
        ]
        conflict = filter_candidate_outfits(make_request("500", user_size="M"), candidates).conflict
        assert conflict.violated_constraints == [ConstraintType.AVAILABILITY, ConstraintType.SIZE]
        assert conflict.min_budget_to_make_feasible is None

    def test_constraints_not_relaxed(self):
        # Even when nothing is feasible, no outfit is returned as feasible.
        candidates = [make_outfit("o1", top={"price": "100.01"})]
        result = filter_candidate_outfits(make_request("150"), candidates)
        assert result.feasible_outfits == []
        assert result.conflict.min_budget_to_make_feasible == Decimal("200.01")


# ---------------------------------------------------------------------------
# ConstraintFilterResult / RejectedOutfit consistency validation
# ---------------------------------------------------------------------------


def mixed_result_dump():
    candidates = [
        make_outfit("o1"),
        make_outfit("o2", top={"price": "999"}),
        make_outfit("o3", top={"price": "999", "availability": False}),
    ]
    return filter_candidate_outfits(make_request("500"), candidates).model_dump()


def all_rejected_dump():
    return filter_candidate_outfits(
        make_request("100"), [make_outfit("o1", top={"price": "999"})]
    ).model_dump()
