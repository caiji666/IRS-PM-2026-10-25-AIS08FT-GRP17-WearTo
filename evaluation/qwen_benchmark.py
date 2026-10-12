"""Frozen synthetic evaluation: raw Qwen, guarded output and dialogue rollout.

No prompt tuning, weather calls, database access or user-session writes occur.
Gold labels are independently authored and require human review.
"""

import argparse
import copy
import hashlib
import json
import platform
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from travel_agent.schemas import ConversationState, TripShoppingRequest
from travel_agent.understanding.extraction import (
    DIALOGUE_SCHEMA,
    MODEL_ID,
    MODEL_REVISION,
    QwenExtractor,
    ground_extraction,
    parse_json,
)
from travel_agent.understanding.state import apply_patch, confirm_outfit, update_state

DEFAULTS = TripShoppingRequest().model_dump(mode="json")
REQUIRED = ("destination", "start_date", "end_date", "activities", "budget", "currency")
# Lexical equivalences only, independent of the production grounding guards.
ALIASES = {
    "首尔": "seoul",
    "东京": "tokyo",
    "大阪": "osaka",
    "新加坡": "singapore",
    "观光": "sightseeing",
    "徒步": "hiking",
    "博物馆参观": "museum visit",
    "museum visits": "museum visit",
    "晚餐": "dinner",
    "boho": "bohemian",
    "极简": "minimalist",
    "极简风": "minimalist",
    "简约": "minimalist",
    "休闲": "casual",
    "休闲风": "casual",
    "运动风": "sporty",
    "skirts": "skirt",
    "靴子": "boots",
    "高跟鞋": "high heels",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(value):
    if isinstance(value, str):
        text = " ".join(value.strip().casefold().replace("_", " ").split())
        return ALIASES.get(text, text)
    if isinstance(value, list):
        return sorted({json.dumps(normalize(v), sort_keys=True, ensure_ascii=False) for v in value})
    if isinstance(value, dict):
        return {normalize(k): normalize(v) for k, v in value.items()}
    return value


def equivalent(actual, expected):
    # JSON booleans/strings are not equivalent to numeric money.
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return False
    return normalize(actual) == normalize(expected)


def missing_fields(request):
    return [field for field in REQUIRED if not request.get(field)]


def plain_patch(before, candidate):
    """Observe proposed values for grading even when business validation rejects.

    This does not accept or fix a malformed patch. The actual state transition is
    independently handled by the real contract/state code below.
    """
    values = copy.deepcopy(before)
    for key, value in candidate.get("set", {}).items():
        values[key] = value
    for key, additions in candidate.get("add", {}).items():
        if (
            key not in values
            or not isinstance(values[key], list)
            or not isinstance(additions, list)
        ):
            raise ValueError("Invalid list addition")
        values[key] = list(dict.fromkeys([*values[key], *additions]))
    for key, removals in candidate.get("remove", {}).items():
        if isinstance(values.get(key), list):
            if not isinstance(removals, list):
                raise ValueError("Invalid list removal")
            values[key] = [v for v in values[key] if v not in removals] if removals else []
        else:
            values[key] = {} if key == "dress_codes" else None
    return values


def evaluate_candidate(candidate, case, previous_state=None):
    """Grade candidate semantics, validation, intent and actual state separately."""
    expected = case["expected"]
    supplied_state = previous_state if previous_state is not None else case.get("previous_state")
    state = ConversationState.model_validate(supplied_state or {})
    before = state.current_request.model_dump(mode="json")
    observed = copy.deepcopy(before if supplied_state else DEFAULTS)
    existing = supplied_state is not None and state.request_version > 0
    predicted_intent = candidate.get("intent")
    if not existing and predicted_intent is None:
        predicted_intent = "CREATE_REQUEST"
    schema_valid, outcome, error = False, "reject", None
    next_state = state.model_copy(deep=True)
    try:
        if not existing:
            observed = copy.deepcopy(DEFAULTS | candidate)
            request = TripShoppingRequest.model_validate(candidate)
            schema_valid = True
            observed = request.model_dump(mode="json")
            next_state = update_state(state, request)
            if not next_state.request_version:
                next_state.request_version = 1
        else:
            # Observe values without turning a full-request-shaped response into
            # a patch or treating an omitted intent as a successful classification.
            if predicted_intent == "MODIFY_REQUIREMENTS":
                observed = plain_patch(before, candidate)
            action = DIALOGUE_SCHEMA.validate_python(candidate)
            schema_valid = True
            if action.intent == "MODIFY_REQUIREMENTS":
                request = apply_patch(state.current_request, action)
                observed = request.model_dump(mode="json")
                next_state = update_state(state, request)
            elif action.intent == "CONFIRM_OUTFIT":
                next_state = confirm_outfit(state, action.outfit_id, action.display_position)
        outcome = "accept"
    except (ValidationError, ValueError, KeyError, TypeError, AttributeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    expected_request = expected["request"]
    mismatches = {
        field: {"expected": value, "actual": observed.get(field)}
        for field, value in expected_request.items()
        if not equivalent(observed.get(field), value)
    }
    extra_fields = sorted(set(observed) - set(DEFAULTS))
    changed = [
        field
        for field, value in expected_request.items()
        if supplied_state and not equivalent(before.get(field), value)
    ]
    changed_matches = {field: field not in mismatches for field in changed}
    unrequested = [
        field
        for field in DEFAULTS
        if supplied_state
        and field not in changed
        and not equivalent(observed.get(field), before.get(field))
    ]
    dress_codes = observed.get("dress_codes")
    activities = observed.get("activities")
    dangling_codes = (
        [key for key in dress_codes if key not in activities]
        if isinstance(dress_codes, dict) and isinstance(activities, list)
        else []
    )
    mandatory = missing_fields(expected_request)
    actual_missing = missing_fields(observed)
    intent_match = predicted_intent == expected["intent"]
    validation_match = outcome == expected["validation"]
    selection_match = next_state.selected_outfit_id == expected.get("selected_outfit_id")
    if outcome == "accept" and predicted_intent in {"CREATE_REQUEST", "MODIFY_REQUIREMENTS"}:
        if next_state.current_request == state.current_request and state.session_status in {
            "CONFIRMED",
            "AWAITING_CONFIRMATION",
        }:
            next_state.session_status = state.session_status
        else:
            next_state.session_status = "COLLECTING_REQUIREMENTS" if actual_missing else "READY"
    return {
        "passed": (
            not mismatches
            and not extra_fields
            and not dangling_codes
            and intent_match
            and validation_match
            and selection_match
        ),
        "predicted_intent": predicted_intent,
        "intent_match": intent_match,
        "contract_valid": schema_valid,
        "validation_outcome": outcome,
        "validation_match": validation_match,
        "observed_request": observed,
        "field_matches": {field: field not in mismatches for field in expected_request},
        "mismatches": mismatches,
        "extra_fields": extra_fields,
        "dangling_dress_code_keys": dangling_codes,
        "changed_field_matches": changed_matches,
        "unrequested_edits": unrequested,
        "missing_fields_expected": mandatory,
        "missing_fields_observed": actual_missing,
        "missing_fields_match": set(mandatory) == set(actual_missing),
        "selection_match": selection_match,
        "error": error,
        "next_state": next_state.model_dump(mode="json"),
    }


def failed_stage(case, message, previous_state=None, parsed=None):
    supplied = previous_state if previous_state is not None else case.get("previous_state")
    state = ConversationState.model_validate(supplied or {})
    result = evaluate_candidate({}, case, previous_state)
    observed = state.current_request.model_dump(mode="json")
    intent = parsed.get("intent") if isinstance(parsed, dict) else None
    if parsed is not None and not state.request_version and intent is None:
        intent = "CREATE_REQUEST"
    expected = case["expected"]
    result.update(
        passed=False,
        contract_valid=False,
        error=message,
        candidate_available=False,
        predicted_intent=intent,
        intent_match=intent == expected["intent"],
        validation_outcome="reject",
        validation_match=expected["validation"] == "reject",
        next_state=state.model_dump(mode="json"),
        observed_request=observed,
        selection_match=state.selected_outfit_id == expected.get("selected_outfit_id"),
        field_matches={
            field: equivalent(observed[field], value)
            for field, value in expected["request"].items()
        },
        mismatches={
            field: {"expected": value, "actual": observed[field]}
            for field, value in expected["request"].items()
            if not equivalent(observed[field], value)
        },
        unrequested_edits=[],
        missing_fields_observed=missing_fields(observed),
        missing_fields_match=set(missing_fields(observed))
        == set(missing_fields(expected["request"])),
    )
    result["changed_field_matches"] = {
        field: result["field_matches"][field] for field in result["changed_field_matches"]
    }
    return result


def score_output(raw, case, previous_state=None):
    try:
        direct = json.loads(raw)
        strict_json = isinstance(direct, dict)
    except (ValueError, TypeError):
        strict_json = False
    parsed = None
    try:
        parsed = parse_json(raw)
        raw_score = evaluate_candidate(copy.deepcopy(parsed), case, previous_state)
        raw_score["candidate_available"] = True
    except (ValueError, TypeError, AttributeError) as exc:
        raw_score = failed_stage(case, f"Parse error: {exc}", previous_state)
    supplied_state = previous_state if previous_state is not None else case.get("previous_state")
    try:
        if parsed is None:
            raise ValueError("No parsed candidate")
        guarded, warnings = ground_extraction(
            copy.deepcopy(parsed),
            case["message"],
            bool(supplied_state and supplied_state["request_version"]),
            TripShoppingRequest.model_validate(supplied_state["current_request"])
            if supplied_state and supplied_state["request_version"]
            else None,
        )
        guarded_score = evaluate_candidate(guarded, case, previous_state)
        guarded_score.update(candidate_available=True, grounding_warnings=warnings)
    except (ValueError, TypeError, AttributeError) as exc:
        guarded_score = failed_stage(
            case, f"Guard/validation error: {exc}", previous_state, parsed=parsed
        )
        # Rejection is a desired control-layer outcome for invalid inputs, but it
        # is not counted as a faithful extraction if no candidate survived.
        guarded_score["safe_rejection"] = case["expected"]["validation"] == "reject"
    return {
        "strict_json_object": strict_json,
        "parseable_object": parsed is not None,
        "raw_candidate": parsed,
        "raw": raw_score,
        "guarded": guarded_score,
    }


def summarize(rows):
    summary = {
        "score_version": "state-fields-and-joins-v1.1",
        "total": len(rows),
        "strict_json_objects": sum(r["strict_json_object"] for r in rows),
        "parseable_objects": sum(r["parseable_object"] for r in rows),
        "stages": {},
    }
    for stage in ("raw", "guarded"):
        field_counts, facts_correct, facts_total = Counter(), 0, 0
        changed_correct, changed_total = 0, 0
        groups = defaultdict(lambda: {"passed": 0, "total": 0})
        for row in rows:
            scores, expected = row[stage], row["expected"]["request"]
            groups[row["category"]]["total"] += 1
            groups[row["category"]]["passed"] += int(scores["passed"])
            for field, matched in scores["field_matches"].items():
                field_counts[(field, "total")] += 1
                field_counts[(field, "correct")] += int(matched)
                if expected[field] not in (None, [], {}):
                    facts_total += 1
                    facts_correct += int(matched)
            changed_total += len(scores["changed_field_matches"])
            changed_correct += sum(scores["changed_field_matches"].values())
        followups = [r for r in rows if r["expected"]["intent"] != "CREATE_REQUEST"]
        accept_cases = [r for r in rows if r["expected"]["validation"] == "accept"]
        summary["stages"][stage] = {
            "passed": sum(r[stage]["passed"] for r in rows),
            "total": len(rows),
            "contract_valid_on_accept_cases": sum(r[stage]["contract_valid"] for r in accept_cases),
            "accept_case_total": len(accept_cases),
            "followup_intent_correct": sum(r[stage]["intent_match"] for r in followups),
            "followup_intent_total": len(followups),
            "populated_field_correct": facts_correct,
            "populated_field_total": facts_total,
            "changed_field_correct": changed_correct,
            "changed_field_total": changed_total,
            "validation_outcome_correct": sum(r[stage]["validation_match"] for r in rows),
            "unrequested_edit_turns": sum(bool(r[stage]["unrequested_edits"]) for r in rows),
            "dangling_dress_code_turns": sum(
                bool(r[stage].get("dangling_dress_code_keys")) for r in rows
            ),
            "safe_rejections_without_candidate": sum(
                r[stage].get("safe_rejection", False) for r in rows
            ),
            "by_category": dict(groups),
            "by_field": {
                field: {
                    "correct": field_counts[(field, "correct")],
                    "total": field_counts[(field, "total")],
                }
                for field in DEFAULTS
            },
        }
    return summary


def write_report(output, path):
    s = output["teacher_forced_summary"]
    lines = [
        f"# {output['dataset_id']} — observed results",
        "",
        f"Run: {output['started_at']}",
        "",
        "Assistant-authored synthetic gold labels require human review. See the evaluation role "
        "and audit provenance below for whether results informed development. "
        "This is a small diagnostic set, not an independent deployment accuracy claim.",
        "",
        f"Evaluation role: `{output.get('evaluation_role', 'historical_diagnostic')}`. "
        + str(output.get("dataset_scope") or "Refer to the dataset documentation for scope."),
        "",
        f"Model: `{MODEL_ID}`; revision `{MODEL_REVISION}`.",
        "",
        f"Dataset SHA-256: `{output['dataset_sha256']}`.",
        "",
        f"Raw strict JSON objects: **{s['strict_json_objects']}/{s['total']}**. "
        f"Objects accepted by the code-fence parser: **{s['parseable_objects']}/{s['total']}**.",
        "",
        "| Metric | Raw Qwen | After existing guards |",
        "|---|---|---|",
    ]
    for label, numerator, denominator in [
        ("Full proposal correctness", "passed", "total"),
        (
            "Contract-valid output on valid-input cases",
            "contract_valid_on_accept_cases",
            "accept_case_total",
        ),
        ("Follow-up intent classification", "followup_intent_correct", "followup_intent_total"),
        ("Populated expected fields", "populated_field_correct", "populated_field_total"),
        ("Intended changed fields", "changed_field_correct", "changed_field_total"),
    ]:
        values = [
            f"{s['stages'][stage][numerator]}/{s['stages'][stage][denominator]}"
            for stage in ("raw", "guarded")
        ]
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    if output.get("audit"):
        lines += [
            "",
            "Scoring audit: the same saved model outputs were regraded, with exact "
            "activity/dress-code key consistency and service-compatible no-op session "
            "transitions checked. The original baseline is preserved; gold labels, "
            "model prompts and generated outputs were not changed. Current production guards "
            "may differ; this is a post-inspection regression audit, not a new held-out run.",
        ]
    lines += [
        "",
        "Safe rejection of invalid input is reported separately; it is not successful "
        "extraction. Invalid-input cases require faithful raw facts and validator rejection. "
        "Semantic equivalences are lexical translations/plurals only. Dress-code keys must "
        "also exactly reference activity values in the submitted request, matching the actual "
        "downstream dictionary lookup. Partial/empty fields "
        "must remain empty; extra weather/schema fields and unintended edits fail.",
        "",
        "## Per-turn results",
        "",
        "| ID | Coverage | Language | Raw | Guarded |",
        "|---|---|---|---|---|",
    ]
    for row in output["teacher_forced"]:
        lines.append(
            f"| {row['id']} | {row['category']} | {row['language']} | "
            f"{'PASS' if row['raw']['passed'] else 'FAIL'} | "
            f"{'PASS' if row['guarded']['passed'] else 'FAIL'} |"
        )
    if output["rollout"]:
        lines += [
            "",
            "## Dialogue rollout",
            "",
            "Teacher-forced turns receive the gold previous state, isolating each extraction. "
            "Rollout carries the actual guarded state, including mistakes and failed edits. "
            "This simulates proposals without writing any real user session.",
            "",
        ]
        for dialogue in output["rollout"]:
            lines.append(
                f"- {dialogue['id']}: {sum(t['guarded']['passed'] for t in dialogue['turns'])}"
                f"/{len(dialogue['turns'])} turns; final gold-state match: "
                f"{dialogue['final_state_match']}."
            )
    lines += [
        "",
        "## Human review and next steps",
        "",
        "Review all gold labels before reporting formal metrics, especially ambiguous dates, "
        "negation, quoted instructions, dress-code scope, resets and confirmation transitions. "
        "Inspect raw output and field diffs in the result JSON. Keep this baseline result; "
        "prompt/guard changes should produce a new run, not overwrite or adjust labels. "
        "Use a fresh held-out set for subsequent prompt tuning evaluation.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def validate_dataset(dataset):
    ids = [case["id"] for scenario in dataset["scenarios"] for case in scenario["turns"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs")
    for scenario in dataset["scenarios"]:
        for case in scenario["turns"]:
            expected = case["expected"]
            if set(expected["request"]) != set(DEFAULTS):
                raise ValueError(f"Incomplete gold request: {case['id']}")
            if case.get("previous_state"):
                ConversationState.model_validate(case["previous_state"])
            if expected["validation"] == "accept":
                TripShoppingRequest.model_validate(expected["request"])
            if case.get("label_status") != "assistant_authored_pending_human_review":
                raise ValueError("Gold-label review status must be explicit")
    return len(ids)


def rescore_saved(source_path, dataset_path, target_path):
    """Audit existing raw generations without changing labels or rerunning Qwen."""
    dataset = json.loads(dataset_path.read_text())
    validate_dataset(dataset)
    source = json.loads(source_path.read_text())
    if source["dataset_sha256"] != digest(dataset_path):
        raise ValueError("Gold dataset differs from the generation baseline")
    output = copy.deepcopy(source)
    output.setdefault("dataset_id", dataset["dataset_id"])
    output["audit"] = {
        "source": str(source_path),
        "source_sha256": digest(source_path),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "evaluator_sha256": digest(Path(__file__)),
        "current_evidence_sha256": digest(
            Path(__file__).resolve().parents[1] / "travel_agent/understanding/english_evidence.py"
        ),
        "production_guards_may_differ": True,
        "changes": [
            "Check exact dress-code/activity dictionary joins",
            "Preserve awaiting/confirmed state on no-op proposals as the real service does",
        ],
        "new_model_calls": 0,
        "gold_changed": False,
        "prompt_changed": False,
    }
    cases = {case["id"]: case for scenario in dataset["scenarios"] for case in scenario["turns"]}
    for index, original in enumerate(source["teacher_forced"]):
        case = cases[original["id"]]
        row = copy.deepcopy(original)
        row.update(score_output(original["raw_output"], case))
        row["input_state"] = copy.deepcopy(case.get("previous_state"))
        output["teacher_forced"][index] = row
    scenarios = {scenario["id"]: scenario for scenario in dataset["scenarios"]}
    for index, original_dialogue in enumerate(source["rollout"]):
        scenario = scenarios[original_dialogue["id"]]
        actual = copy.deepcopy(scenario["turns"][0].get("previous_state"))
        original_prior = copy.deepcopy(actual)
        turns = []
        for case, original in zip(scenario["turns"], original_dialogue["turns"], strict=True):
            # The production prompt receives only request facts. Status-only
            # audit corrections must never substitute different model inputs.
            actual_request = actual["current_request"] if actual else None
            original_request = original_prior["current_request"] if original_prior else None
            if actual_request != original_request:
                raise ValueError("Audit changes model input facts; a new rollout is required")
            row = copy.deepcopy(original)
            row.update(score_output(original["raw_output"], case, actual))
            row["input_state"] = copy.deepcopy(original_prior)
            row["scoring_state"] = copy.deepcopy(actual)
            turns.append(row)
            actual = row["guarded"]["next_state"]
            original_prior = original["guarded"]["next_state"]
        expected = scenario["turns"][-1]["expected"]
        final_match = (
            equivalent(actual["current_request"], expected["request"])
            and actual["selected_outfit_id"] == expected["selected_outfit_id"]
        )
        output["rollout"][index] = {
            "id": scenario["id"],
            "turns": turns,
            "final_state_match": final_match,
        }
    output["teacher_forced_summary"] = summarize(output["teacher_forced"])
    if target_path.resolve() == source_path.resolve():
        raise ValueError("Preserve the original baseline; choose a new audited output path")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    write_report(output, target_path.with_suffix(".md"))
    print(f"Audited saved raw outputs; no model calls. Saved {target_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("evaluation/datasets/qwen_understanding_v1.json")
    )
    parser.add_argument("--model-path", default=MODEL_ID)
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/results/qwen_understanding_v1.json")
    )
    parser.add_argument("--rollout", action="store_true")
    parser.add_argument(
        "--rescore", type=Path, help="Audit saved raw outputs; do not call the model"
    )
    parser.add_argument(
        "--limit", type=int, help="Debug subset; excluded from full baseline claims"
    )
    args = parser.parse_args()
    if args.rescore:
        rescore_saved(args.rescore, args.dataset, args.output)
        return
    dataset = json.loads(args.dataset.read_text())
    validate_dataset(dataset)
    dataset_hash = digest(args.dataset)
    extractor_file = (
        Path(__file__).resolve().parents[1] / "travel_agent/understanding/extraction.py"
    )
    extractor_hash = digest(extractor_file)
    extractor = QwenExtractor(args.model_path, local_files_only=True)
    output = {
        "dataset_id": dataset["dataset_id"],
        "dataset_sha256": dataset_hash,
        "extractor_sha256": extractor_hash,
        "label_status": dataset["label_status"],
        "evaluation_role": dataset.get("evaluation_role", "synthetic_diagnostic"),
        "dataset_scope": dataset.get("scope"),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "subset_limit": args.limit,
        "teacher_forced": [],
        "rollout": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def execute(case, previous_state=None):
        start = time.monotonic()
        prior = previous_state if previous_state is not None else case.get("previous_state")
        request = (
            TripShoppingRequest.model_validate(prior["current_request"])
            if prior and prior["request_version"]
            else None
        )
        try:
            raw, metadata = extractor.generate_raw(case["message"], request)
            row = score_output(raw, case, previous_state)
        except Exception as exc:
            raw, metadata = "", {"generation_error": f"{type(exc).__name__}: {exc}"}
            row = score_output(raw, case, previous_state)
        row.update(
            {key: case[key] for key in ("id", "category", "language", "message", "expected")}
        )
        row.update(
            raw_output=raw,
            metadata=metadata,
            input_state=copy.deepcopy(prior),
            seconds=round(time.monotonic() - start, 2),
        )
        return row

    def checkpoint():
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")

    cases = [case for scenario in dataset["scenarios"] for case in scenario["turns"]]
    if args.limit:
        cases = cases[: args.limit]
    for index, case in enumerate(cases, 1):
        row = execute(case)
        output["teacher_forced"].append(row)
        print(
            f"{index}/{len(cases)} {case['id']}: raw={row['raw']['passed']} "
            f"guarded={row['guarded']['passed']}",
            flush=True,
        )
        checkpoint()
    output["teacher_forced_summary"] = summarize(output["teacher_forced"])
    if args.rollout and not args.limit:
        for scenario in dataset["scenarios"]:
            if scenario["kind"] != "dialogue":
                continue
            actual_state = scenario["turns"][0].get("previous_state")
            turns = []
            for case in scenario["turns"]:
                # Empty rollout state is supplied explicitly after failed first
                # turns so that actual omissions, not gold state, carry forward.
                row = execute(case, actual_state)
                turns.append(row)
                actual_state = row["guarded"]["next_state"]
                print(f"rollout {case['id']}: guarded={row['guarded']['passed']}", flush=True)
            gold = scenario["turns"][-1]["expected"]
            matches = (
                equivalent(actual_state["current_request"], gold["request"])
                and actual_state["selected_outfit_id"] == gold["selected_outfit_id"]
            )
            output["rollout"].append(
                {"id": scenario["id"], "turns": turns, "final_state_match": matches}
            )
            checkpoint()
    if digest(args.dataset) != dataset_hash or digest(extractor_file) != extractor_hash:
        raise RuntimeError("Dataset/extractor changed during run; baseline is invalid")
    output["finished_at"] = datetime.now(timezone.utc).isoformat()
    checkpoint()
    report = args.output.with_suffix(".md")
    write_report(output, report)
    print(json.dumps(output["teacher_forced_summary"], ensure_ascii=False), flush=True)
    print(f"Saved {args.output} and {report}", flush=True)


if __name__ == "__main__":
    main()
