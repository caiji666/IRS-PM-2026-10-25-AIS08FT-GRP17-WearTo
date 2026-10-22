"""Tests for the Selective Re-execution Planner (Module 7, rerun_planner)."""

import itertools

import pytest

from travel_agent.recommendation.student5.models import (
    TripShoppingRequest,
    TripShoppingRequestPatch,
)
from travel_agent.recommendation.student5.patch_processor import apply_requirement_patch
from travel_agent.recommendation.student5.rerun_planner import (
    FIELD_STAGE_MAP,
    NoChangesError,
    plan_from_update_result,
    plan_selective_rerun,
)

FULL = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def make_request():
    return TripShoppingRequest.model_validate(
        {
            "destination": "Seoul",
            "start_date": "2026-12-20",
            "end_date": "2026-12-27",
            "activities": ["sightseeing"],
            "budget": 500,
            "currency": "SGD",
            "style": ["minimalist"],
        }
    )


def update(**operations):
    patch = TripShoppingRequestPatch.model_validate({"intent": "MODIFY_REQUIREMENTS", **operations})
    return apply_requirement_patch(make_request(), patch)


# ---------------------------------------------------------------------------
# Single-field mappings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,stages",
    [
        ("budget", [10, 11, 12]),
        ("user_size", [8, 9, 10, 11, 12]),
        ("excluded_items", [8, 9, 10, 11, 12]),
        ("currency", [8, 9, 10, 11, 12]),
        ("style", [4, 5, 6, 7, 8, 9, 10, 11, 12]),
        ("activities", [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        ("destination", FULL),
        ("start_date", FULL),
        ("end_date", FULL),
    ],
)
def test_single_field_mapping(field, stages):
    plan = plan_selective_rerun([field])
    assert plan.changed_fields == [field]
    assert plan.affected_stages == stages
    assert plan.rerun_from_stage == stages[0]


def test_mapping_covers_every_request_field():
    assert set(FIELD_STAGE_MAP) == set(TripShoppingRequest.model_fields)


# ---------------------------------------------------------------------------
# Dependency guarantees (checked for every combination of changed fields)
# ---------------------------------------------------------------------------

ALL_PLANS = [
    plan_selective_rerun(list(combo))
    for r in range(1, len(FIELD_STAGE_MAP) + 1)
    for combo in itertools.combinations(TripShoppingRequest.model_fields, r)
]


class TestDependencyGuarantees:
    def test_every_plan_ends_with_final_ranking(self):
        assert all(plan.affected_stages[-1] == 12 for plan in ALL_PLANS)


# ---------------------------------------------------------------------------
# Multi-field plans
# ---------------------------------------------------------------------------


class TestMultipleFields:
    def test_union_of_mappings(self):
        plan = plan_selective_rerun(["excluded_items", "style"])
        assert plan.affected_stages == sorted(
            set(FIELD_STAGE_MAP["excluded_items"]) | set(FIELD_STAGE_MAP["style"])
        )
        assert plan.affected_stages == [4, 5, 6, 7, 8, 9, 10, 11, 12]


# ---------------------------------------------------------------------------
# Invalid input and no-change handling
# ---------------------------------------------------------------------------


class TestInvalidInput:
    def test_empty_changed_fields_rejected(self):
        with pytest.raises(NoChangesError):
            plan_selective_rerun([])
        assert issubclass(NoChangesError, ValueError)

    @pytest.mark.parametrize("fields", [["size"]])
    def test_unknown_field_rejected(self, fields):
        with pytest.raises(ValueError, match="unknown"):
            plan_selective_rerun(fields)

    def test_non_update_result_rejected(self):
        with pytest.raises(TypeError, match="RequirementUpdateResult"):
            plan_from_update_result(["budget"])

    def test_single_string_rejected(self):
        with pytest.raises(TypeError):
            plan_selective_rerun("budget")

    def test_unchanged_update_result_rejected(self):
        result = update(set={"budget": 500})
        assert result.changed is False
        with pytest.raises(NoChangesError):
            plan_from_update_result(result)


# ---------------------------------------------------------------------------
# From RequirementUpdateResult
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Determinism and immutability
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RerunPlan model
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The planner never executes pipeline stages
# ---------------------------------------------------------------------------


class TestNoExecution:
    def test_pipeline_functions_not_called(self, monkeypatch):
        import travel_agent.recommendation.student5.constraint_engine as ce
        import travel_agent.recommendation.student5.ranking_engine as re_

        def boom(*args, **kwargs):
            raise AssertionError("pipeline stage was executed")

        monkeypatch.setattr(ce, "filter_candidate_outfits", boom)
        monkeypatch.setattr(re_, "rank_outfits", boom)
        monkeypatch.setattr(re_, "score_outfit", boom)
        plan_from_update_result(update(set={"budget": 400, "destination": "Tokyo"}))
        plan_selective_rerun(list(TripShoppingRequest.model_fields))
