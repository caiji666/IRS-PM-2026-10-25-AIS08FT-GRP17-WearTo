import copy
import json
from pathlib import Path

import pytest

from evaluation.qwen_benchmark import evaluate_candidate, score_output, summarize, validate_dataset

DATA = Path(__file__).resolve().parents[1] / "evaluation/datasets/qwen_understanding_v1.json"


@pytest.fixture
def cases():
    dataset = json.loads(DATA.read_text())
    return {case["id"]: case for scenario in dataset["scenarios"] for case in scenario["turns"]}


def test_dataset_has_complete_unique_labels_and_continuous_dialogue_facts():
    dataset = json.loads(DATA.read_text())
    assert validate_dataset(dataset) == 40
    assert len(dataset["scenarios"]) == 32
    for scenario in dataset["scenarios"]:
        if scenario["kind"] == "dialogue":
            for previous, current in zip(scenario["turns"], scenario["turns"][1:]):
                assert (
                    current["previous_state"]["current_request"] == previous["expected"]["request"]
                )
                assert (
                    current["previous_state"]["selected_outfit_id"]
                    == (previous["expected"]["selected_outfit_id"])
                )


def test_correct_request_scores_both_stages_without_altering_raw(cases):
    case = cases["S01"]
    raw = json.dumps(case["expected"]["request"])
    result = score_output(raw, case)
    assert result["raw"]["passed"] and result["guarded"]["passed"]
    assert result["strict_json_object"]
    assert result["raw_candidate"] == case["expected"]["request"]


def test_code_fences_are_reported_separately_from_strict_json(cases):
    case = cases["S01"]
    result = score_output("```json\n" + json.dumps(case["expected"]["request"]) + "\n```", case)
    assert not result["strict_json_object"]
    assert result["parseable_object"] and result["raw"]["passed"]


def test_raw_currency_hallucination_is_not_hidden_by_guard_success(cases):
    case = cases["S09"]
    candidate = case["expected"]["request"] | {"currency": "USD"}
    result = score_output(json.dumps(candidate), case)
    assert not result["raw"]["passed"]
    assert result["raw"]["mismatches"]["currency"]["actual"] == "USD"
    assert result["guarded"]["passed"]
    assert result["raw_candidate"]["currency"] == "USD"


def test_invalid_input_rejection_is_not_a_successful_guarded_extraction(cases):
    case = cases["S20"]
    result = score_output(json.dumps(case["expected"]["request"]), case)
    assert result["raw"]["passed"]  # Faithful invalid date plus expected rejection.
    assert not result["raw"]["contract_valid"]
    assert not result["guarded"]["passed"]
    assert result["guarded"]["safe_rejection"]
    assert not score_output("not json", case)["raw"]["passed"]


def test_equivalent_patch_operations_are_scored_on_merged_facts(cases):
    case = cases["P03"]
    removal = {"intent": "MODIFY_REQUIREMENTS", "remove": {"excluded_items": ["skirt"]}}
    replacement = {"intent": "MODIFY_REQUIREMENTS", "set": {"excluded_items": ["boots"]}}
    assert evaluate_candidate(removal, case)["passed"]
    assert evaluate_candidate(replacement, case)["passed"]
    assert not evaluate_candidate(case["expected"]["request"], case)["passed"]


def test_unchanged_owned_items_and_extra_weather_fields_cannot_be_ignored(cases):
    case = cases["P01"]
    candidate = {
        "intent": "MODIFY_REQUIREMENTS",
        "set": {"budget": 175, "currency": "USD", "owned_item_ids": []},
    }
    result = evaluate_candidate(candidate, case)
    assert not result["passed"]
    assert "owned_item_ids" in result["unrequested_edits"]
    case = cases["S17"]
    assert not evaluate_candidate(case["expected"]["request"] | {"snow": True}, case)["passed"]


def test_rollout_uses_actual_state_and_does_not_restore_gold(cases):
    case = cases["D01_T2"]
    wrong_state = copy.deepcopy(case["previous_state"])
    wrong_state["current_request"]["destination"] = "Osaka"
    patch = {
        "intent": "MODIFY_REQUIREMENTS",
        "set": {"start_date": "2026-11-04", "end_date": "2026-11-08"},
    }
    result = evaluate_candidate(patch, case, wrong_state)
    assert not result["passed"]
    assert result["next_state"]["current_request"]["destination"] == "Osaka"
    assert result["mismatches"]["destination"]["actual"] == "Osaka"


def test_confirmation_resolves_positions_against_fixture_without_fake_products(cases):
    case = cases["D03_T2"]
    assert evaluate_candidate({"intent": "CONFIRM_OUTFIT", "display_position": 2}, case)["passed"]
    assert not evaluate_candidate({"intent": "CONFIRM_OUTFIT", "display_position": 1}, case)[
        "passed"
    ]


def test_summary_separates_changed_fields_and_missing_fields(cases):
    case = cases["P01"]
    result = score_output('{"intent":"MODIFY_REQUIREMENTS","set":{},"add":{},"remove":{}}', case)
    row = result | {"category": case["category"], "expected": case["expected"]}
    summary = summarize([row])
    assert summary["stages"]["raw"]["changed_field_total"] == 2
    assert summary["stages"]["raw"]["changed_field_correct"] == 0
    assert summary["stages"]["raw"]["passed"] == 0


def test_guard_failure_preserves_rollout_state_instead_of_creating_empty_session(cases):
    case = cases["S20"]
    result = score_output(json.dumps(case["expected"]["request"]), case)
    assert result["guarded"]["next_state"]["request_version"] == 0
    assert result["guarded"]["validation_outcome"] == "reject"
    assert result["guarded"]["validation_match"]


def test_semantically_translated_but_dangling_dress_code_keys_fail_integration(cases):
    case = cases["S02"]
    candidate = copy.deepcopy(case["expected"]["request"])
    candidate["dress_codes"] = {"晚餐": "smart casual"}
    result = evaluate_candidate(candidate, case)
    assert result["mismatches"] == {}  # Lexically equivalent concepts.
    assert result["dangling_dress_code_keys"] == ["晚餐"]
    assert not result["passed"]  # Actual service looks up the English activity key.


def test_noop_proposal_preserves_awaiting_confirmation_like_real_service(cases):
    case = cases["D03_T1"]
    result = evaluate_candidate({"intent": "MODIFY_REQUIREMENTS", "set": {}}, case)
    assert not result["passed"]  # Wrong intent remains a failure.
    assert result["next_state"]["session_status"] == "AWAITING_CONFIRMATION"
    assert result["next_state"]["last_recommended_outfit_ids"] == [
        "outfit_301",
        "outfit_302",
        "outfit_303",
    ]


def test_saved_generation_audit_preserves_raw_outputs_and_original_file(cases, tmp_path):
    from evaluation.qwen_benchmark import digest, rescore_saved

    case = cases["S02"]
    candidate = copy.deepcopy(case["expected"]["request"])
    candidate["dress_codes"] = {"晚餐": "smart casual"}
    raw = json.dumps(candidate, ensure_ascii=False)
    row = score_output(raw, case) | {
        "id": case["id"],
        "category": case["category"],
        "language": case["language"],
        "message": case["message"],
        "expected": case["expected"],
        "raw_output": raw,
    }
    source = tmp_path / "baseline.json"
    source.write_text(
        json.dumps(
            {
                "dataset_sha256": digest(DATA),
                "started_at": "test",
                "teacher_forced": [row],
                "rollout": [],
            }
        )
    )
    original = source.read_bytes()
    audited = rescore_saved(source, DATA, tmp_path / "audit.json")
    assert source.read_bytes() == original
    assert audited["teacher_forced"][0]["raw_output"] == raw
    assert audited["audit"]["new_model_calls"] == 0
    assert not audited["audit"]["gold_changed"]
    assert not audited["teacher_forced"][0]["raw"]["passed"]
    assert audited["teacher_forced"][0]["guarded"]["passed"]  # Canonical key repair now exists.
