"""Tests for the Selective Re-execution Planner (Module 7, rerun_planner)."""

import ast
import inspect
import itertools

import pytest
from pydantic import ValidationError

import travel_agent.recommendation.student5.rerun_planner as rerun_planner
from travel_agent.recommendation.student5.dialogue_state import (
    create_initial_state,
    modify_requirements,
)
from travel_agent.recommendation.student5.models import (
    TripShoppingRequest,
    TripShoppingRequestPatch,
)
from travel_agent.recommendation.student5.patch_processor import apply_requirement_patch
from travel_agent.recommendation.student5.rerun_planner import (
    FIELD_STAGE_MAP,
    NoChangesError,
    RerunPlan,
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


def test_budget_example_from_spec():
    plan = plan_selective_rerun(["budget"])
    assert plan.model_dump() == {
        "changed_fields": ["budget"],
        "rerun_from_stage": 10,
        "affected_stages": [10, 11, 12],
        "reason": plan.reason,
    }
    assert "budget" in plan.reason
    assert "hard-constraint filtering" in plan.reason
    assert "outfit compatibility scoring" in plan.reason
    assert "final ranking" in plan.reason


@pytest.mark.parametrize(
    "field,stage_name",
    [
        ("style", "social reference retrieval (Stage 5)"),
        ("activities", "social reference reranking (Stage 6)"),
        ("user_size", "candidate outfit generation (Stage 9)"),
        ("currency", "outfit compatibility scoring (Stage 11)"),
    ],
)
def test_reason_names_dependency_stages(field, stage_name):
    assert stage_name in plan_selective_rerun([field]).reason


# ---------------------------------------------------------------------------
# Dependency guarantees (checked for every combination of changed fields)
# ---------------------------------------------------------------------------

ALL_PLANS = [
    plan_selective_rerun(list(combo))
    for r in range(1, len(FIELD_STAGE_MAP) + 1)
    for combo in itertools.combinations(TripShoppingRequest.model_fields, r)
]


class TestDependencyGuarantees:
    def test_every_combination_planned(self):
        assert len(ALL_PLANS) == 2 ** len(FIELD_STAGE_MAP) - 1

    def test_candidate_change_rescored_before_ranking(self):
        # Stage 9 or 10 can change the feasible set; Stage 12 rejects compatibility_score=None.
        for plan in ALL_PLANS:
            if 12 in plan.affected_stages and {9, 10} & set(plan.affected_stages):
                assert 11 in plan.affected_stages, plan.changed_fields

    def test_query_change_refreshes_social_references(self):
        for plan in ALL_PLANS:
            if 4 in plan.affected_stages:
                assert {5, 6} <= set(plan.affected_stages), plan.changed_fields

    def test_product_pool_change_regenerates_outfits(self):
        for plan in ALL_PLANS:
            if 8 in plan.affected_stages:
                assert 9 in plan.affected_stages, plan.changed_fields

    def test_every_plan_ends_with_final_ranking(self):
        assert all(plan.affected_stages[-1] == 12 for plan in ALL_PLANS)

    def test_every_mapping_satisfies_rules(self):
        for field, stages in FIELD_STAGE_MAP.items():
            s = set(stages)
            if 4 in s:
                assert {5, 6} <= s, field
            if 8 in s:
                assert 9 in s, field
            if {9, 10} & s:
                assert 11 in s, field

    def test_budget_rescored(self):
        assert plan_selective_rerun(["budget"]).affected_stages == [10, 11, 12]


# ---------------------------------------------------------------------------
# Multi-field plans
# ---------------------------------------------------------------------------


class TestMultipleFields:
    def test_budget_and_user_size(self):
        plan = plan_selective_rerun(["budget", "user_size"])
        assert plan.changed_fields == ["budget", "user_size"]
        assert plan.affected_stages == [8, 9, 10, 11, 12]
        assert plan.rerun_from_stage == 8

    def test_destination_and_budget(self):
        plan = plan_selective_rerun(["budget", "destination"])
        assert plan.changed_fields == ["destination", "budget"]
        assert plan.affected_stages == FULL
        assert plan.rerun_from_stage == 2

    def test_union_of_mappings(self):
        plan = plan_selective_rerun(["excluded_items", "style"])
        assert plan.affected_stages == sorted(
            set(FIELD_STAGE_MAP["excluded_items"]) | set(FIELD_STAGE_MAP["style"])
        )
        assert plan.affected_stages == [4, 5, 6, 7, 8, 9, 10, 11, 12]

    def test_all_fields(self):
        plan = plan_selective_rerun(list(TripShoppingRequest.model_fields))
        assert plan.affected_stages == FULL
        assert plan.changed_fields == list(TripShoppingRequest.model_fields)

    def test_stages_sorted_unique_and_min_is_start(self):
        plan = plan_selective_rerun(["currency", "activities", "user_size", "budget"])
        assert plan.affected_stages == sorted(set(plan.affected_stages))
        assert all(isinstance(s, int) for s in plan.affected_stages)
        assert plan.rerun_from_stage == min(plan.affected_stages) == 3

    def test_fields_in_request_order(self):
        plan = plan_selective_rerun(["user_size", "currency", "budget", "activities"])
        assert plan.changed_fields == ["activities", "budget", "currency", "user_size"]

    def test_input_order_does_not_matter(self):
        assert plan_selective_rerun(["style", "budget", "end_date"]) == plan_selective_rerun(
            ["end_date", "style", "budget"]
        )

    def test_duplicates_removed(self):
        plan = plan_selective_rerun(["budget", "budget", "user_size", "budget"])
        assert plan.changed_fields == ["budget", "user_size"]

    def test_reason_explains_each_field_in_canonical_order(self):
        plan = plan_selective_rerun(["user_size", "budget"])
        assert plan.reason.index("budget") < plan.reason.index("user_size")
        assert "Stage 8" in plan.reason and "Stage 10" in plan.reason


# ---------------------------------------------------------------------------
# Invalid input and no-change handling
# ---------------------------------------------------------------------------


class TestInvalidInput:
    def test_empty_changed_fields_rejected(self):
        with pytest.raises(NoChangesError):
            plan_selective_rerun([])
        assert issubclass(NoChangesError, ValueError)

    @pytest.mark.parametrize("fields", [["size"], ["budget", "colour"], ["Budget"]])
    def test_unknown_field_rejected(self, fields):
        with pytest.raises(ValueError, match="unknown"):
            plan_selective_rerun(fields)

    def test_single_string_rejected(self):
        with pytest.raises(TypeError):
            plan_selective_rerun("budget")

    def test_unchanged_update_result_rejected(self):
        result = update(set={"budget": 500})
        assert result.changed is False
        with pytest.raises(NoChangesError):
            plan_from_update_result(result)

    def test_non_update_result_rejected(self):
        with pytest.raises(TypeError, match="RequirementUpdateResult"):
            plan_from_update_result(["budget"])


# ---------------------------------------------------------------------------
# From RequirementUpdateResult
# ---------------------------------------------------------------------------


class TestFromUpdateResult:
    def test_uses_changed_fields(self):
        result = update(set={"budget": 400, "user_size": "M"})
        plan = plan_from_update_result(result)
        assert plan.changed_fields == result.changed_fields == ["budget", "user_size"]
        assert plan.affected_stages == [8, 9, 10, 11, 12]

    def test_module6_state_update_to_plan(self):
        state = create_initial_state("session_001", make_request())
        patch = TripShoppingRequestPatch.model_validate(
            {
                "intent": "MODIFY_REQUIREMENTS",
                "set": {"budget": 400},
                "add": {"excluded_items": ["long coat"], "style": ["casual"]},
            }
        )
        new_state, result = modify_requirements(state, patch)
        plan = plan_from_update_result(result)
        assert result.changed_fields == ["budget", "style", "excluded_items"]
        assert plan.changed_fields == result.changed_fields
        assert plan.affected_stages == [4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert plan.rerun_from_stage == 4
        assert new_state.request_version == 2  # planning itself changes nothing further

    def test_update_result_not_modified(self):
        result = update(set={"destination": "Tokyo"}, add={"style": ["casual"]})
        snapshot = result.model_dump()
        plan_from_update_result(result)
        assert result.model_dump() == snapshot


# ---------------------------------------------------------------------------
# Determinism and immutability
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_repeated_calls_identical(self):
        plans = [plan_selective_rerun(["style", "budget"]) for _ in range(3)]
        assert plans[0] == plans[1] == plans[2]
        assert plans[0].model_dump_json() == plans[2].model_dump_json()

    def test_reason_non_empty_and_deterministic(self):
        a = plan_selective_rerun(["excluded_items"]).reason
        b = plan_selective_rerun(["excluded_items"]).reason
        assert a and a.strip() and a == b

    def test_input_list_not_modified(self):
        fields = ["user_size", "budget", "budget"]
        plan_selective_rerun(fields)
        assert fields == ["user_size", "budget", "budget"]

    def test_mapping_not_mutated_by_plan(self):
        plan = plan_selective_rerun(["budget"])
        plan.affected_stages.append(99)
        assert FIELD_STAGE_MAP["budget"] == (10, 11, 12)
        assert plan_selective_rerun(["budget"]).affected_stages == [10, 11, 12]


# ---------------------------------------------------------------------------
# RerunPlan model
# ---------------------------------------------------------------------------


class TestRerunPlanModel:
    def test_round_trip(self):
        plan = plan_selective_rerun(["destination", "budget"])
        assert RerunPlan.model_validate(plan.model_dump()) == plan
        assert RerunPlan.model_validate_json(plan.model_dump_json()) == plan

    def test_unknown_field_rejected(self):
        data = plan_selective_rerun(["budget"]).model_dump()
        data["stages_executed"] = [10, 12]
        with pytest.raises(ValidationError, match="extra"):
            RerunPlan.model_validate(data)

    @pytest.mark.parametrize(
        "changes,message",
        [
            ({"changed_fields": []}, "changed_fields"),
            ({"changed_fields": ["size"]}, "changed_fields"),
            ({"changed_fields": ["budget", "budget"]}, "changed_fields"),
            (
                {
                    "changed_fields": ["user_size", "budget"],
                    "affected_stages": [8, 10, 12],
                    "rerun_from_stage": 8,
                },
                "changed_fields",
            ),
            ({"affected_stages": [12, 10]}, "affected_stages"),
            ({"affected_stages": [10, 10, 12]}, "affected_stages"),
            ({"affected_stages": [10, 12]}, "affected_stages"),
            ({"rerun_from_stage": 12}, "rerun_from_stage"),
            ({"reason": "budget changed"}, "reason"),
        ],
    )
    def test_inconsistent_plan_rejected(self, changes, message):
        data = plan_selective_rerun(["budget"]).model_dump()
        data.update(changes)
        with pytest.raises(ValidationError, match=message):
            RerunPlan.model_validate(data)

    def test_stage_strings_rejected(self):
        data = plan_selective_rerun(["budget"]).model_dump()
        data["affected_stages"] = ["10", "12"]
        with pytest.raises(ValidationError):
            RerunPlan.model_validate(data)


# ---------------------------------------------------------------------------
# The planner never executes pipeline stages
# ---------------------------------------------------------------------------


class TestNoExecution:
    def test_imports_no_pipeline_modules(self):
        tree = ast.parse(inspect.getsource(rerun_planner))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        forbidden = {
            "travel_agent.recommendation.student5.constraint_engine",
            "travel_agent.recommendation.student5.ranking_engine",
            "travel_agent.recommendation.student5.recommendation_formatter",
            "travel_agent.recommendation.student5.dialogue_state",
            "travel_agent.recommendation.student5.confirmation_handler",
            "travel_agent.recommendation.student5.explanation_handler",
        }
        assert not imported & forbidden

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
