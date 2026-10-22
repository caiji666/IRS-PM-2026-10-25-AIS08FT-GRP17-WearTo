"""Real-data refinement demo: change the budget, then recommend again with the same candidates.

Run from the repository root (the contact sheet needs Pillow, e.g. the `preview` extra):

    uv run --extra preview python -m scripts.run_student5_real_refinement \\
        PATH/TO/ranked_outfits.json --budget 200 [--select OUTFIT_ID] \\
        [--preview-mode FLAT_LAY] [--output-dir DIR]

Flow, through `travel_agent.recommendation.service.run` only:

    recommend (the temporary 500 USD test request of scripts.run_student4_real_top3)
    -> modify_requirements with patch {"intent": "MODIFY_REQUIREMENTS", "set": {"budget": ...}}
    -> checks: ok, changed, changed_fields == ["budget"], new budget, rerun plan has Stages 10-12
    -> recommend again with the updated request and the updated state
    -> image folders and contact sheet (scripts.export_student5_top3_assets)
    -> confirm one outfit of the NEW Top-3 (default: the first) with the latest state
    -> preview it, only if the confirm succeeded (scripts.run_student5_real_flow)

Formal CLI runs automatically load the repository `.env`, require the real OpenAI provider and
fail if the result is a mock or product-card fallback. Mock providers remain available only to
offline unit tests through the Python function arguments.

Only the budget changes, so the same Student 4 candidate outfits and compatibility scores are
reused: Student 5's filtering and ranking run again; Students 1-4 are NOT re-run. (The rerun
plan lists Stage 11 because a full pipeline would re-score; this demo reuses Student 4's saved
scores.) Results go to `artifacts/student4_real_data/refinement_budget_<budget>/` and never
overwrite the 500 USD result. Nothing in the output is hard-coded: Top-3 IDs, scores and counts
come from the run.

Exit codes: 0 when every step succeeds and a real OpenAI image file is generated; 1 when
modify_requirements, a recommend, the selection, the confirm or real image generation fails. Nothing
later runs after a failed step: no re-recommendation after a failed modification, no confirm
for an outfit outside the new Top-3, no preview after a failed confirm.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from scripts.export_student5_top3_assets import contact_sheet, export_images, fetch
from scripts.run_student4_real_top3 import (
    TEST_REQUEST,
    build_payload,
    load_ranked_outfits,
    summarise,
)
from scripts.run_student5_real_flow import (
    PREVIEW_MODES,
    FlowError,
    _check_options,
    confirm_then_preview,
    load_real_preview_environment,
    require_real_preview,
)

DEFAULT_ROOT = Path("artifacts/student4_real_data")
REQUIRED_STAGES = (10, 11, 12)
SCOPE_NOTE = (
    "Only the budget changed. Student 5 re-ran its budget-related hard-constraint filtering "
    "and final ranking on the same Student 4 candidate outfits and compatibility scores. "
    "Students 1-4 were not re-run. The 500 USD starting request is the temporary test request "
    "of scripts.run_student4_real_top3, not a budget from an upstream module."
)

ServiceRun = Callable[[dict], Any]


class RefinementError(RuntimeError):
    """A step failed; the flow stops before re-recommending."""


def _service_run() -> ServiceRun:
    from travel_agent.recommendation.service import run

    return run


def _data(result) -> dict:
    return result.model_dump(mode="json")["data"] or {}


def check_modification(data: dict, budget: Decimal) -> list[str]:
    """Problems with the modify_requirements result (empty when it is as expected)."""
    problems = []
    if data.get("changed") is not True:
        problems.append("changed is not true")
    if data.get("changed_fields") != ["budget"]:
        problems.append(f"changed_fields is {data.get('changed_fields')}, expected ['budget']")
    if Decimal(str((data.get("updated_request") or {}).get("budget"))) != budget:
        problems.append("updated_request.budget does not equal the requested budget")
    stages = (data.get("rerun_plan") or {}).get("affected_stages") or []
    missing = [s for s in REQUIRED_STAGES if s not in stages]
    if missing:
        problems.append(f"rerun_plan misses Stages {missing}")
    return problems


def _scores(items: list[dict]) -> list[dict]:
    return [
        {
            "outfit_id": r["outfit_id"],
            "display_position": r["display_position"],
            "total_purchase_cost": f"{Decimal(str(r['total_purchase_cost'])):.2f}",
            "currency": r["currency"],
            "final_outfit_score": r["final_outfit_score"],
        }
        for r in items
    ]


def compare(original: dict, refined: dict, original_budget: str, new_budget: str) -> dict:
    """Before/after view of the two recommendation rounds."""
    before = original["recommendation_set"]["recommendations"]
    after = refined["recommendation_set"]["recommendations"]
    before_ids = [r["outfit_id"] for r in before]
    after_ids = [r["outfit_id"] for r in after]
    fs_before, fs_after = original["filter_summary"], refined["filter_summary"]
    return {
        "original_budget": original_budget,
        "new_budget": new_budget,
        "budget_currency": TEST_REQUEST["currency"],
        "display_currency": after[0]["currency"] if after else TEST_REQUEST["currency"],
        "original_top3": before_ids,
        "new_top3": after_ids,
        "original": {
            "feasible_count": fs_before["feasible_count"],
            "rejected_count": fs_before["rejected_count"],
            "budget_rejections": fs_before["conflict_summary"]["BUDGET"],
            "outfits": _scores(before),
        },
        "refined": {
            "feasible_count": fs_after["feasible_count"],
            "rejected_count": fs_after["rejected_count"],
            "budget_rejections": fs_after["conflict_summary"]["BUDGET"],
            "outfits": _scores(after),
        },
        "added_outfits": [i for i in after_ids if i not in before_ids],
        "removed_outfits": [i for i in before_ids if i not in after_ids],
        "unchanged_outfits": [i for i in after_ids if i in before_ids],
    }


def run_refinement(ranked_outfits: list, budget: str, run: ServiceRun | None = None) -> dict:
    """recommend → modify_requirements → recommend; raises RefinementError on any failure."""
    run = run or _service_run()
    try:
        new_budget = Decimal(budget)
    except InvalidOperation:
        raise RefinementError(f"budget {budget!r} is not a number") from None

    first_payload = build_payload(ranked_outfits)
    first = run(first_payload)
    if first.status != "ok":
        raise RefinementError(f"original recommend returned {first.status}: {first.reason}")
    original = _data(first)

    patch = {"intent": "MODIFY_REQUIREMENTS", "set": {"budget": budget}}
    modified = run({"action": "modify_requirements", "state": original["state"], "patch": patch})
    modification = _data(modified)
    if modified.status != "ok":
        raise RefinementError(f"modify_requirements returned {modified.status}: {modified.reason}")
    problems = check_modification(modification, new_budget)
    if problems:
        raise RefinementError(f"unexpected modify_requirements result: {problems}")

    second = run(
        {
            "action": "recommend",
            "request": modification["updated_request"],
            "student4": first_payload["student4"],
            "state": modification["state"],
        }
    )
    if second.status != "ok":
        raise RefinementError(f"refined recommend returned {second.status}: {second.reason}")
    refined = _data(second)

    refined_top3 = summarise(refined)
    refined_top3["test_request"] = modification["updated_request"]
    refined_top3["note"] = SCOPE_NOTE
    refined_top3["recommendation_round"] = refined["state"]["recommendation_round"]
    return {
        "modification_result": {
            "note": SCOPE_NOTE,
            "original_request": first_payload["request"],
            "patch": patch,
            "changed": modification["changed"],
            "changed_fields": modification["changed_fields"],
            "updated_request": modification["updated_request"],
            "rerun_plan": modification["rerun_plan"],
            "students_rerun": ["student5"],
        },
        "student5_top3": refined_top3,
        "comparison": compare(original, refined, str(first_payload["request"]["budget"]), budget),
        # Latest round (state + recommendation set) for confirm; not written to disk as is.
        "latest_round": {
            "state": refined["state"],
            "recommendation_set": refined["recommendation_set"],
        },
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_results(results: dict, out_dir: Path, fetcher=fetch, sheet: bool = True) -> dict:
    """Write the three JSON files, image folders and (optionally) the contact sheet."""
    for name in ("modification_result", "student5_top3", "comparison"):
        _write_json(out_dir / f"{name}.json", results[name])
    summary = export_images(results["student5_top3"], out_dir / "top3_images", fetcher)
    if sheet:
        budget = results["comparison"]["new_budget"]
        currency = results["comparison"]["budget_currency"]
        title = (
            f"Student 5 Top-3 after budget change to {budget} {currency} "
            "(test result; not the team UI)"
        )
        contact_sheet(
            results["student5_top3"],
            out_dir / "top3_images",
            out_dir / "top3_contact_sheet.png",
            title,
        )
    return summary


def run_demo(
    ranked_outfits: list,
    budget: str,
    out_dir: Path,
    select: str | None = None,
    preview_mode: str = "FLAT_LAY",
    provider: str = "server",
    run: ServiceRun | None = None,
    fetcher=fetch,
    sheet: bool = True,
) -> dict:
    """recommend → modify_requirements → recommend → export cards → confirm → preview.

    Raises RefinementError (modification or recommend) or FlowError (selection, options). The
    confirm uses the second round's state; preview runs only after a successful confirm.
    """
    _check_options(preview_mode, provider)
    run = run or _service_run()
    results = run_refinement(ranked_outfits, budget, run=run)
    results["download_summary"] = save_results(results, out_dir, fetcher=fetcher, sheet=sheet)
    choice = select if select is not None else results["comparison"]["new_top3"][0]
    confirmed, preview = confirm_then_preview(
        results["latest_round"], choice, preview_mode, provider, run, note=SCOPE_NOTE
    )
    _write_json(out_dir / "confirmed_outfit.json", confirmed)
    _write_json(out_dir / "preview_result.json", preview)
    results.update(confirmed_outfit=confirmed, preview_result=preview)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ranked_outfits", type=Path, help="Student 4 output (JSON list)")
    parser.add_argument("--budget", required=True, help="new budget in the request currency")
    parser.add_argument("--select", default=None, help="outfit_id of the new Top-3 (default: #1)")
    parser.add_argument("--preview-mode", default="FLAT_LAY", choices=PREVIEW_MODES)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.output_dir or DEFAULT_ROOT / f"refinement_budget_{args.budget}"
    if out.resolve() == DEFAULT_ROOT.resolve():
        print("error: refusing to write over the original 500 USD result folder")
        return 1

    try:
        load_real_preview_environment()
        results = run_demo(
            load_ranked_outfits(args.ranked_outfits),
            args.budget,
            out,
            select=args.select,
            preview_mode=args.preview_mode,
            provider="server",
        )
    except (RefinementError, FlowError) as exc:
        print(f"error: {exc}")
        return 1
    summary = results["download_summary"]

    mod, cmp_ = results["modification_result"], results["comparison"]
    print(f"note: {SCOPE_NOTE}")
    print(
        f"modify_requirements: changed={mod['changed']} fields={mod['changed_fields']} "
        f"budget {cmp_['original_budget']} -> {cmp_['new_budget']} "
        f"{cmp_['budget_currency']}, "
        f"affected stages {mod['rerun_plan']['affected_stages']}"
    )
    for label, key in (("original", "original"), ("refined", "refined")):
        part = cmp_[key]
        print(
            f"{label}: feasible={part['feasible_count']} rejected={part['rejected_count']} "
            f"(BUDGET {part['budget_rejections']}) top3={[o['outfit_id'] for o in part['outfits']]}"
        )
    print(f"filter_summary: {json.dumps(results['student5_top3']['filter_summary'])}")
    for r in results["student5_top3"]["top3"]:
        print(
            f"#{r['display_position']} {r['outfit_id']}: "
            f"compatibility={r['compatibility_score']:.6f} "
            f"relevance={r['outfit_product_relevance']:.6f} "
            f"final={r['final_outfit_score']:.6f} total={r['total_purchase_cost']} {r['currency']}"
        )
    print(
        f"added={cmp_['added_outfits']} removed={cmp_['removed_outfits']} "
        f"unchanged={cmp_['unchanged_outfits']}"
    )
    print(f"images: {summary['downloaded']} downloaded, {summary['failed']} failed")
    confirmed, preview = results["confirmed_outfit"], results["preview_result"]
    try:
        generated_image = require_real_preview(preview)
    except FlowError as exc:
        print(f"error: {exc}")
        return 1
    print(
        f"confirm {confirmed['selected_outfit_id']}: status={confirmed['confirm_status']} "
        f"succeeded={confirmed['confirm_succeeded']} session_status={confirmed['session_status']} "
        f"(round {confirmed['recommendation_round']})"
    )
    print(
        f"preview: completed={preview['preview_completed']} status={preview['preview_status']} "
        f"result={preview.get('preview_result_status')} source={preview['preview_source']} "
        f"requested={preview['requested_mode']} delivered={preview['delivered_mode']} "
        f"image_generated={preview['image_generated']}"
    )
    if preview["fallback_reason"]:
        print(f"  fallback reason: {preview['fallback_reason']}")
    print(f"  {preview.get('explanation')}")
    print(f"saved: {out.resolve()}")
    print(f"generated image: {generated_image.resolve()}")
    ok = confirmed["confirm_succeeded"] and preview["preview_completed"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
