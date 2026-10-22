"""Tests for TripShoppingRequestPatch merging (Module 6, patch_processor)."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from travel_agent.recommendation.student5.models import (
    TripShoppingRequest,
    TripShoppingRequestPatch,
)
from travel_agent.recommendation.student5.patch_processor import (
    PatchApplicationError,
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


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_excluded_items(self):
        result = apply(add={"excluded_items": ["boots"]})
        assert result.updated_request.excluded_items == ["long coat", "boots"]

    def test_add_existing_value_is_not_duplicated(self):
        result = apply(add={"excluded_items": [" long coat ", "boots", "boots"]})
        assert result.updated_request.excluded_items == ["long coat", "boots"]


# ---------------------------------------------------------------------------
# remove (PROVISIONAL semantics)
# ---------------------------------------------------------------------------


class TestRemove:
    def test_remove_listed_values(self):
        result = apply(add={}, remove={"activities": ["dinner"]})
        assert result.updated_request.activities == ["sightseeing"]
        assert result.changed_fields == ["activities"]

    def test_remove_list_field_with_non_list_value_rejected(self):
        with pytest.raises(PatchApplicationError, match="activities"):
            apply(remove={"activities": "dinner"})

    def test_remove_null_clears_list_field(self):
        # Backward-compatible form accepted by Student 5 (equivalent to remove: []).
        result = apply(remove={"style": None})
        assert result.updated_request.style == []
        assert result.changed_fields == ["style"]

    def test_remove_null_resets_optional_field(self):
        result = apply(make_request(user_size="M"), remove={"user_size": None})
        assert result.updated_request.user_size is None
        assert result.changed_fields == ["user_size"]

    def test_remove_optional_field_with_value_rejected(self):
        with pytest.raises(PatchApplicationError, match="user_size"):
            apply(make_request(user_size="M"), remove={"user_size": "M"})

    @pytest.mark.parametrize("field", ["budget"])
    def test_remove_required_field_rejected(self, field):
        with pytest.raises(PatchApplicationError, match=field):
            apply(remove={field: None})


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

    def test_unknown_field_rejected_by_patch_model(self):
        with pytest.raises(ValidationError, match="colour"):
            patch(set={"colour": "black"})

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


class TestChangedFields:
    def test_set_to_same_value_is_not_a_change(self):
        result = apply(set={"destination": "Seoul", "budget": "500.00"})
        assert result.changed is False
        assert result.changed_fields == []
        assert result.updated_request == result.previous_request
