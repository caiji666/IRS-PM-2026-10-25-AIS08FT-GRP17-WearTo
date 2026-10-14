"""Tests for Stage 10 Hard Constraint Filtering (Module 2)."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

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
    FeasibleCandidateOutfit,
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

    def test_user_owned_price_kept_on_product(self):
        outfit = make_outfit(footwear={"is_user_owned": True, "price": "900.00"})
        result = filter_candidate_outfits(make_request("100"), [outfit])
        assert result.feasible_outfits[0].items[2].price == Decimal("900.00")

    def test_decimal_arithmetic_is_exact(self):
        # 0.10 + 0.20 + 0.00 == 0.30 exactly; float would give 0.30000000000000004 > 0.30.
        outfit = make_outfit(
            top={"price": "0.10"}, bottom={"price": "0.20"}, footwear={"price": "0.00"}
        )
        assert purchase_total(outfit) == Decimal("0.30")
        assert isinstance(purchase_total(outfit), Decimal)
        assert filter_candidate_outfits(make_request("0.30"), [outfit]).feasible_count == 1
        rejected = filter_candidate_outfits(make_request("0.29"), [outfit]).rejected_outfits[0]
        assert rejected.total_purchase_cost == Decimal("0.30")

    def test_all_user_owned_outfit_costs_zero(self):
        owned = {"is_user_owned": True}
        outfit = make_outfit(top=owned, bottom=owned, footwear=owned)
        assert purchase_total(outfit) == Decimal("0")
        assert filter_candidate_outfits(make_request("1"), [outfit]).feasible_count == 1


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

    def test_all_unavailable_items_listed(self):
        outfit = make_outfit(top={"availability": False}, footwear={"availability": False})
        [rejected] = filter_candidate_outfits(make_request(), [outfit]).rejected_outfits
        assert rejected.reasons[0].product_ids == ["p_top", "p_footwear"]


# ---------------------------------------------------------------------------
# C. Size (PROVISIONAL single user_size)
# ---------------------------------------------------------------------------


def sizes(*values, reliable=True):
    return {"available_sizes": list(values), "size_data_reliable": reliable}


class TestSize:
    def test_no_user_size_skips_size_constraint(self):
        outfit = make_outfit(top=sizes(), bottom=sizes("XS"))
        assert filter_candidate_outfits(make_request(user_size=None), [outfit]).feasible_count == 1

    def test_no_size_data_skips_item(self):
        outfit = make_outfit()  # size_data_reliable=False, available_sizes=None on every item
        assert filter_candidate_outfits(make_request(user_size="M"), [outfit]).feasible_count == 1

    def test_unreliable_size_data_not_treated_as_mismatch(self):
        # Sizes are listed but flagged unreliable: the item is skipped, not rejected.
        outfit = make_outfit(top=sizes("S", reliable=False), bottom=sizes(reliable=False))
        assert filter_candidate_outfits(make_request(user_size="M"), [outfit]).feasible_count == 1

    def test_mixed_reliable_and_unreliable_size_data(self):
        outfit = make_outfit(top=sizes("M", "L"), bottom=sizes("S", reliable=False))
        assert filter_candidate_outfits(make_request(user_size="M"), [outfit]).feasible_count == 1

    def test_matching_size_passes(self):
        outfit = make_outfit(top=sizes("S", "M", "L"), bottom=sizes("M"), footwear=sizes("m"))
        assert filter_candidate_outfits(make_request(user_size="M"), [outfit]).feasible_count == 1

    @pytest.mark.parametrize(
        "user_size,available",
        [
            ("m", ["M"]),
            ("M", ["m"]),
            ("  M ", ["M"]),
            ("M", [" m  "]),
            ("xl", ["XL"]),
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

    def test_empty_reliable_sizes_rejected(self):
        outfit = make_outfit(footwear=sizes())
        [rejected] = filter_candidate_outfits(
            make_request(user_size="M"), [outfit]
        ).rejected_outfits
        assert only_reason_types(rejected) == [ConstraintType.SIZE]
        assert rejected.reasons[0].product_ids == ["p_footwear"]

    def test_user_owned_item_not_size_checked(self):
        outfit = make_outfit(top={"is_user_owned": True, **sizes("XS")})
        assert filter_candidate_outfits(make_request(user_size="M"), [outfit]).feasible_count == 1


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

    def test_currency_mismatch_is_value_error(self):
        assert issubclass(CurrencyMismatchError, ValueError)

    def test_user_owned_item_currency_ignored(self):
        outfit = make_outfit(bottom={"is_user_owned": True, "currency": "USD"})
        assert filter_candidate_outfits(make_request(currency="SGD"), [outfit]).feasible_count == 1


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

    def test_counts_and_compatibility_calls_saved(self):
        candidates = [make_outfit("o1"), make_outfit("o2", top={"price": "999"}), make_outfit("o3")]
        result = filter_candidate_outfits(make_request("500"), candidates)
        assert (result.total_candidates, result.feasible_count, result.rejected_count) == (3, 2, 1)
        assert result.compatibility_calls_saved == 1

    def test_input_order_preserved(self):
        ids = ["z", "a", "m", "b", "y"]
        candidates = [
            make_outfit(i, top={"price": "999"} if i in {"a", "y"} else None) for i in ids
        ]
        result = filter_candidate_outfits(make_request("500"), candidates)
        assert [o.outfit_id for o in result.feasible_outfits] == ["z", "m", "b"]
        assert [o.outfit_id for o in result.rejected_outfits] == ["a", "y"]

    def test_inputs_not_modified(self):
        request = make_request("100", user_size="M")
        candidates = [
            make_outfit("o1"),
            make_outfit(
                "o2", top={"price": "1.00"}, bottom={"price": "1.00"}, footwear={"price": "1.00"}
            ),
        ]
        request_before = request.model_dump()
        candidates_before = [c.model_dump() for c in candidates]
        result = filter_candidate_outfits(request, candidates)
        assert request.model_dump() == request_before
        assert [c.model_dump() for c in candidates] == candidates_before
        assert result.feasible_outfits[0] is not candidates[1]
        assert result.feasible_outfits[0].items[0] is not candidates[1].items[0]

    def test_empty_candidate_list(self):
        result = filter_candidate_outfits(make_request(), [])
        assert result.feasible_outfits == [] and result.rejected_outfits == []
        assert (
            result.total_candidates,
            result.feasible_count,
            result.rejected_count,
            result.compatibility_calls_saved,
        ) == (0, 0, 0, 0)
        assert all(n == 0 for n in result.conflict_summary.values())

    def test_feasible_outfit_matches_candidate(self):
        candidate = make_outfit(
            "keep",
            outerwear={"price": "80.00", **sizes("M")},
            top={"description": "wool knit", "embedding_ref": "emb/top"},
        )
        [feasible] = filter_candidate_outfits(
            make_request("500", user_size="M"), [candidate]
        ).feasible_outfits
        assert type(feasible) is FeasibleCandidateOutfit
        assert feasible.model_dump() == candidate.model_dump()

    def test_compatibility_score_not_touched(self):
        candidate = make_outfit()
        [feasible] = filter_candidate_outfits(make_request(), [candidate]).feasible_outfits
        assert feasible.compatibility_score is None

    def test_duplicate_outfit_id_in_candidates_fails(self):
        candidates = [make_outfit("dup"), make_outfit("ok"), make_outfit("dup")]
        with pytest.raises(ValueError, match="dup"):
            filter_candidate_outfits(make_request(), candidates)

    def test_result_round_trip(self):
        candidates = [make_outfit("o1"), make_outfit("o2", top={"price": "999"})]
        result = filter_candidate_outfits(make_request("500"), candidates)
        assert ConstraintFilterResult.model_validate(result.model_dump()) == result
        assert ConstraintFilterResult.model_validate_json(result.model_dump_json()) == result
        assert (
            result.model_dump(mode="json")["rejected_outfits"][0]["total_purchase_cost"]
            == "1099.00"
        )


# ---------------------------------------------------------------------------
# Structured conflict explanation (no feasible outfit)
# ---------------------------------------------------------------------------


class TestConflictExplanation:
    def test_present_when_all_candidates_rejected(self):
        candidates = [
            make_outfit("o1", top={"price": "200.00"}),
            make_outfit("o2", top={"price": "150.00"}),
        ]
        result = filter_candidate_outfits(make_request("200"), candidates)
        conflict = result.conflict
        assert conflict is not None
        assert conflict.violated_constraints == [ConstraintType.BUDGET]
        assert conflict.budget == Decimal("200")
        assert conflict.currency == "SGD"
        # o2 costs 250.00 and fails only on budget: the smallest budget that would admit it.
        assert conflict.min_budget_to_make_feasible == Decimal("250.00")
        assert "budget" in conflict.message.lower()

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

    def test_absent_when_some_outfit_feasible(self):
        candidates = [make_outfit("o1"), make_outfit("o2", top={"price": "999"})]
        assert filter_candidate_outfits(make_request("500"), candidates).conflict is None

    def test_absent_for_empty_candidate_list(self):
        assert filter_candidate_outfits(make_request(), []).conflict is None

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


class TestResultValidation:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("feasible_count", 5),
            ("rejected_count", 5),
            ("total_candidates", 5),
        ],
    )
    def test_inconsistent_counts_rejected(self, field, value):
        data = mixed_result_dump()
        data[field] = value
        with pytest.raises(ValidationError, match=field):
            ConstraintFilterResult.model_validate(data)

    def test_compatibility_calls_saved_must_equal_rejected_count(self):
        data = mixed_result_dump()
        data["compatibility_calls_saved"] = 0
        with pytest.raises(ValidationError, match="compatibility_calls_saved"):
            ConstraintFilterResult.model_validate(data)

    def test_conflict_summary_must_cover_every_constraint(self):
        data = mixed_result_dump()
        del data["conflict_summary"][ConstraintType.SIZE]
        with pytest.raises(ValidationError, match="conflict_summary"):
            ConstraintFilterResult.model_validate(data)

    def test_conflict_summary_must_match_rejection_reasons(self):
        data = mixed_result_dump()
        data["conflict_summary"][ConstraintType.BUDGET] = 1  # actually 2
        with pytest.raises(ValidationError, match="conflict_summary"):
            ConstraintFilterResult.model_validate(data)

    def test_outfit_ids_unique_across_result(self):
        data = mixed_result_dump()
        data["rejected_outfits"][0]["outfit_id"] = "o1"
        with pytest.raises(ValidationError, match="outfit_id"):
            ConstraintFilterResult.model_validate(data)

    def test_conflict_required_when_nothing_feasible(self):
        data = all_rejected_dump()
        data["conflict"] = None
        with pytest.raises(ValidationError, match="conflict"):
            ConstraintFilterResult.model_validate(data)

    def test_conflict_forbidden_when_something_feasible(self):
        data = mixed_result_dump()
        data["conflict"] = all_rejected_dump()["conflict"]
        with pytest.raises(ValidationError, match="conflict"):
            ConstraintFilterResult.model_validate(data)

    def test_conflict_constraints_must_match_summary(self):
        data = all_rejected_dump()
        data["conflict"]["violated_constraints"] = [ConstraintType.SIZE]
        with pytest.raises(ValidationError, match="violated_constraints"):
            ConstraintFilterResult.model_validate(data)

    def test_wrong_min_budget_rejected(self):
        data = all_rejected_dump()  # one budget-only outfit costing 1099.00
        data["conflict"]["min_budget_to_make_feasible"] = Decimal("1000.00")
        with pytest.raises(ValidationError, match="min_budget_to_make_feasible"):
            ConstraintFilterResult.model_validate(data)

    def test_min_budget_given_when_it_should_be_null_rejected(self):
        data = filter_candidate_outfits(
            make_request("500"), [make_outfit("o1", top={"availability": False})]
        ).model_dump()
        assert data["conflict"]["min_budget_to_make_feasible"] is None
        data["conflict"]["min_budget_to_make_feasible"] = Decimal("150.00")
        with pytest.raises(ValidationError, match="min_budget_to_make_feasible"):
            ConstraintFilterResult.model_validate(data)

    def test_min_budget_null_when_it_should_have_value_rejected(self):
        data = all_rejected_dump()
        data["conflict"]["min_budget_to_make_feasible"] = None
        with pytest.raises(ValidationError, match="min_budget_to_make_feasible"):
            ConstraintFilterResult.model_validate(data)

    def test_correct_min_budget_with_mixed_rejections_accepted(self):
        candidates = [
            make_outfit("o1", top={"price": "60.00", "availability": False}),
            make_outfit("o2", top={"price": "300.00"}),
            make_outfit("o3", top={"price": "250.00"}),
        ]
        data = filter_candidate_outfits(make_request("150"), candidates).model_dump()
        result = ConstraintFilterResult.model_validate(data)
        assert result.conflict.min_budget_to_make_feasible == Decimal("350.00")

    def test_rejected_outfit_reason_types_unique(self):
        reason = {"constraint": "BUDGET", "message": "over budget"}
        with pytest.raises(ValidationError, match="duplicate"):
            RejectedOutfit.model_validate(
                {"outfit_id": "o1", "reasons": [reason, reason], "total_purchase_cost": "1"}
            )

    def test_all_rejected_round_trip(self):
        data = all_rejected_dump()
        result = ConstraintFilterResult.model_validate(data)
        assert ConstraintFilterResult.model_validate_json(result.model_dump_json()) == result
