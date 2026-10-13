"""Tests for TripShoppingRequestPatch merging (Module 6, patch_processor)."""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from travel_agent.recommendation.student5.models import (
    TripShoppingRequest,
    TripShoppingRequestPatch,
)
from travel_agent.recommendation.student5.patch_processor import (
    PatchApplicationError,
    RequirementUpdateResult,
    apply_requirement_patch,
)


def make_request(**overrides):
    data = {
        "destination": "Seoul",
        "start_date": "2026-12-20",
        "end_date": "2026-12-27",
        "activities": ["sightseeing", "dinner"],
        "budget": 500,
        "currency": "SGD",
        "style": ["minimalist"],
        "excluded_items": ["long coat"],
    }
    data.update(overrides)
    return TripShoppingRequest.model_validate(data)


def patch(**operations):
    return TripShoppingRequestPatch.model_validate({"intent": "MODIFY_REQUIREMENTS", **operations})


def apply(request=None, **operations):
    return apply_requirement_patch(request or make_request(), patch(**operations))


# ---------------------------------------------------------------------------
# set
# ---------------------------------------------------------------------------


class TestSet:
    def test_set_budget(self):
        result = apply(set={"budget": 400})
        assert result.updated_request.budget == Decimal("400")
        assert result.changed_fields == ["budget"]
        assert result.changed is True

    def test_set_destination(self):
        result = apply(set={"destination": "Tokyo"})
        assert result.updated_request.destination == "Tokyo"
        assert result.changed_fields == ["destination"]

    def test_set_replaces_whole_list(self):
        result = apply(set={"style": ["casual", "smart casual"]})
        assert result.updated_request.style == ["casual", "smart casual"]

    def test_set_optional_size(self):
        assert apply(set={"user_size": "L"}).updated_request.user_size == "L"

    def test_set_dates_together(self):
        result = apply(set={"start_date": "2027-01-05", "end_date": "2027-01-10"})
        assert result.updated_request.start_date == date(2027, 1, 5)
        assert result.changed_fields == ["start_date", "end_date"]

    def test_changed_fields_follow_request_field_order(self):
        result = apply(set={"budget": 300, "destination": "Osaka"})
        assert result.changed_fields == ["destination", "budget"]


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_activities(self):
        result = apply(add={"activities": ["hiking"]})
        assert result.updated_request.activities == ["sightseeing", "dinner", "hiking"]
        assert result.changed_fields == ["activities"]

    def test_add_style(self):
        assert apply(add={"style": ["casual"]}).updated_request.style == ["minimalist", "casual"]

    def test_add_excluded_items(self):
        result = apply(add={"excluded_items": ["boots"]})
        assert result.updated_request.excluded_items == ["long coat", "boots"]

    def test_add_deduplicates_and_keeps_order(self):
        result = apply(add={"activities": ["dinner", "hiking", "museum", "hiking"]})
        assert result.updated_request.activities == ["sightseeing", "dinner", "hiking", "museum"]

    def test_add_ignores_surrounding_whitespace_when_deduplicating(self):
        result = apply(add={"activities": [" dinner ", "  hiking"]})
        assert result.updated_request.activities == ["sightseeing", "dinner", "hiking"]

    def test_add_is_case_sensitive(self):
        result = apply(add={"activities": ["Dinner"]})
        assert result.updated_request.activities == ["sightseeing", "dinner", "Dinner"]

    def test_add_to_empty_list(self):
        result = apply(make_request(style=[]), add={"style": ["casual"]})
        assert result.updated_request.style == ["casual"]

    def test_add_existing_value_is_not_a_change(self):
        result = apply(add={"activities": ["dinner"]})
        assert result.changed is False
        assert result.changed_fields == []

    def test_add_invalid_value_rejected(self):
        with pytest.raises(ValidationError):
            apply(add={"style": [""]})


# ---------------------------------------------------------------------------
# remove (PROVISIONAL semantics)
# ---------------------------------------------------------------------------


class TestRemove:
    def test_remove_listed_values(self):
        result = apply(add={}, remove={"activities": ["dinner"]})
        assert result.updated_request.activities == ["sightseeing"]
        assert result.changed_fields == ["activities"]

    def test_remove_strips_whitespace(self):
        result = apply(remove={"excluded_items": ["  long coat "]})
        assert result.updated_request.excluded_items == []

    def test_remove_absent_value_is_not_a_change(self):
        result = apply(remove={"activities": ["hiking"]})
        assert result.updated_request.activities == ["sightseeing", "dinner"]
        assert result.changed is False

    @pytest.mark.parametrize("field", ["activities", "style", "excluded_items"])
    def test_remove_empty_array_clears_list_field(self, field):
        # Team contract (docs/contracts.md): "Empty remove arrays explicitly clear that list".
        result = apply(remove={field: []})
        assert getattr(result.updated_request, field) == []
        assert result.changed_fields == [field]

    def test_remove_empty_array_on_empty_list_is_not_a_change(self):
        result = apply(make_request(style=[]), remove={"style": []})
        assert result.changed is False

    def test_remove_empty_array_leaves_other_fields_unchanged(self):
        original = make_request(user_size="M")
        result = apply(original, remove={"style": []})
        before, after = original.model_dump(), result.updated_request.model_dump()
        assert {f for f in before if before[f] != after[f]} == {"style"}

    def test_remove_null_clears_list_field(self):
        # Backward-compatible form accepted by Student 5 (equivalent to remove: []).
        result = apply(remove={"style": None})
        assert result.updated_request.style == []
        assert result.changed_fields == ["style"]

    def test_remove_null_resets_optional_field(self):
        result = apply(make_request(user_size="M"), remove={"user_size": None})
        assert result.updated_request.user_size is None
        assert result.changed_fields == ["user_size"]

    @pytest.mark.parametrize(
        "field", ["destination", "start_date", "end_date", "budget", "currency"]
    )
    def test_remove_required_field_rejected(self, field):
        with pytest.raises(PatchApplicationError, match=field):
            apply(remove={field: None})

    def test_remove_list_field_with_non_list_value_rejected(self):
        with pytest.raises(PatchApplicationError, match="activities"):
            apply(remove={"activities": "dinner"})

    def test_remove_optional_field_with_value_rejected(self):
        with pytest.raises(PatchApplicationError, match="user_size"):
            apply(make_request(user_size="M"), remove={"user_size": "M"})


# ---------------------------------------------------------------------------
# Preservation, validation and result structure
# ---------------------------------------------------------------------------


class TestMergeBehaviour:
    def test_fields_absent_from_patch_unchanged(self):
        original = make_request(user_size="M")
        result = apply(original, set={"budget": 400}, add={"style": ["casual"]})
        before, after = original.model_dump(), result.updated_request.model_dump()
        for field in before:
            if field not in {"budget", "style"}:
                assert after[field] == before[field], field

    def test_combined_operations(self):
        result = apply(
            set={"budget": 400},
            add={"excluded_items": ["boots"]},
            remove={"activities": ["dinner"]},
        )
        assert result.changed_fields == ["activities", "budget", "excluded_items"]

    def test_unknown_field_rejected_by_patch_model(self):
        with pytest.raises(ValidationError, match="colour"):
            patch(set={"colour": "black"})

    @pytest.mark.parametrize("budget", [0, -10])
    def test_invalid_budget_rejected(self, budget):
        with pytest.raises(ValidationError, match="budget"):
            apply(set={"budget": budget})

    def test_end_before_start_rejected(self):
        with pytest.raises(ValidationError, match="end_date"):
            apply(set={"end_date": "2026-12-01"})

    def test_blank_currency_rejected(self):
        with pytest.raises(ValidationError):
            apply(set={"currency": "  "})

    def test_invalid_date_value_rejected(self):
        with pytest.raises(ValidationError):
            apply(set={"start_date": "next week"})

    @pytest.mark.parametrize("intent", ["CONFIRM_OUTFIT", "REQUEST_EXPLANATION"])
    def test_non_modify_intent_rejected(self, intent):
        other = TripShoppingRequestPatch.model_validate({"intent": intent})
        with pytest.raises(PatchApplicationError, match="MODIFY_REQUIREMENTS"):
            apply_requirement_patch(make_request(), other)

    def test_dict_patch_rejected(self):
        with pytest.raises(TypeError, match="TripShoppingRequestPatch"):
            apply_requirement_patch(make_request(), {"intent": "MODIFY_REQUIREMENTS"})

    def test_previous_request_not_modified(self):
        original = make_request()
        snapshot = original.model_dump()
        result = apply(original, add={"activities": ["hiking"]}, remove={"style": None})
        assert original.model_dump() == snapshot
        assert result.previous_request == original
        assert result.previous_request is not original
        assert result.updated_request.activities is not original.activities

    def test_patch_not_modified(self):
        p = patch(add={"activities": ["hiking"]})
        snapshot = p.model_dump()
        apply_requirement_patch(make_request(), p)
        assert p.model_dump() == snapshot


class TestChangedFields:
    def test_set_to_same_value_is_not_a_change(self):
        result = apply(set={"destination": "Seoul", "budget": "500.00"})
        assert result.changed is False
        assert result.changed_fields == []
        assert result.updated_request == result.previous_request

    def test_empty_patch_is_not_a_change(self):
        assert apply().changed is False

    def test_only_really_changed_fields_listed(self):
        result = apply(set={"destination": "Seoul", "budget": 450}, add={"activities": ["dinner"]})
        assert result.changed_fields == ["budget"]


class TestRequirementUpdateResult:
    def test_round_trip(self):
        result = apply(set={"budget": 400}, add={"style": ["casual"]})
        assert RequirementUpdateResult.model_validate(result.model_dump()) == result
        assert RequirementUpdateResult.model_validate_json(result.model_dump_json()) == result

    def test_changed_fields_must_match_requests(self):
        data = apply(set={"budget": 400}).model_dump()
        data["changed_fields"] = ["destination"]
        with pytest.raises(ValidationError, match="changed_fields"):
            RequirementUpdateResult.model_validate(data)

    def test_changed_flag_must_match_changed_fields(self):
        data = apply(set={"budget": 400}).model_dump()
        data["changed"] = False
        with pytest.raises(ValidationError, match="changed"):
            RequirementUpdateResult.model_validate(data)

    def test_unknown_field_rejected(self):
        data = apply(set={"budget": 400}).model_dump()
        data["affected_stages"] = [10, 12]
        with pytest.raises(ValidationError, match="extra"):
            RequirementUpdateResult.model_validate(data)
