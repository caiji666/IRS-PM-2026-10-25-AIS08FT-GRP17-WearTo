"""Backend-only Student 3 -> 4 invocation and optional Student 5 contract check."""

import argparse
import json
import os
from pathlib import Path

from travel_agent.outfits.output import recommendation_input, save_artifact
from travel_agent.outfits.service import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--ranking-report", type=Path, required=True)
    parser.add_argument("--understanding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-recommendation", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("EBAY_MANAGED_ROOT", str(Path.cwd().resolve()))
    report = json.loads(args.ranking_report.read_text())
    understanding = json.loads(args.understanding.read_text())
    result = run(
        {
            "product_result_path": str(args.products),
            "understanding": understanding,
            "ranking_request": report["request_used"],
        }
    )
    if args.output.resolve() in {
        args.products.resolve(),
        args.ranking_report.resolve(),
        args.understanding.resolve(),
    }:
        parser.error("Output must not overwrite inputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_artifact(result.model_dump(mode="json"), args.output, report)
    print(
        json.dumps(
            {
                "status": result.status,
                "output": str(args.output),
                "summary": result.data.get("summary"),
                "reason": result.reason,
            }
        )
    )
    if result.status != "ok":
        return 4
    input5 = recommendation_input(result, understanding)
    input5_path = args.output.with_name("student5_input.json")
    if input5_path.resolve() in {
        args.products.resolve(),
        args.ranking_report.resolve(),
        args.understanding.resolve(),
    }:
        parser.error("Student5 input output must not overwrite an upstream input")
    save_artifact(input5, input5_path, report)
    if args.verify_recommendation:
        from travel_agent.recommendation.service import run as recommend

        recommendation = recommend(input5)
        target = args.output.with_name(args.output.stem + "_student5.json")
        save_artifact(recommendation.model_dump(mode="json"), target, report)
        print(
            json.dumps(
                {
                    "student5_status": recommendation.status,
                    "output": str(target),
                    "reason": recommendation.reason,
                }
            )
        )
        return 0 if recommendation.status in {"ok", "no_candidates"} else 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
