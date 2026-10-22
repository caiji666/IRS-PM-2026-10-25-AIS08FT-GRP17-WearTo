"""Run Student 5 once on a real Student 4 output file and save the Top-3 for review.

Run from the repository root:

    uv run python -m scripts.run_student4_real_top3 PATH/TO/ranked_outfits.json [--output FILE]

The input is Student 4's `run()` result: a JSON list of outfits (`outfit_id`,
`compatibility_score`, category-keyed `products` with each product's `product_score`). It is
wrapped here in the Student 4 hand-off shape the adapter expects (PROVISIONAL until Student 4
fixes its own wrapper) and passed to `travel_agent.recommendation.service.run`.

TEMPORARY TEST SETTING: the request below uses a 500 USD budget. Student 5 preserves that user
input in its conversation state and converts the budget and candidate prices to SGD with one
daily ECB rate snapshot for hard-constraint filtering. The 500 USD value is chosen for this test
only; it is not a budget from an upstream module.

The first run of a day may access the public ECB feed; later runs use the Git-ignored local
cache. No model credentials are needed for recommendation. The service, ranking formula and
default beta are used unchanged. The saved file holds only the request, beta, filter summary
and the Top-3 as mapped by Student 5 (no embeddings or provenance).

Exit codes: 0 when the service returns `ok`; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_OUTPUT = Path("artifacts/student4_real_data/student5_top3.json")

TEST_REQUEST = {
    "destination": "Seoul",
    "start_date": "2026-12-20",
    "end_date": "2026-12-27",
    "activities": ["sightseeing", "dinner"],
    "budget": "500",
    "currency": "USD",
    "style": ["minimalist"],
}

TEST_BUDGET_NOTE = (
    "TEMPORARY TEST SETTING: the budget of 500 USD is a test value chosen for this run. The "
    "user input remains USD in state; Student 5 converts it and candidate prices to SGD using "
    "the recorded daily ECB snapshot for filtering. It is not a budget passed from an upstream "
    "module."
)

STUDENT4_WRAPPER = {"status": "ok", "source": "student4", "model_version": "bofei-compatibility"}


def load_ranked_outfits(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, str):  # Student 4's run() returns a JSON string
        data = json.loads(data)
    if not isinstance(data, list):
        raise SystemExit(f"{path.name}: expected a JSON list of outfits at the root")
    return data


def build_payload(ranked_outfits: list) -> dict:
    return {
        "action": "recommend",
        "request": dict(TEST_REQUEST),
        "student4": {**STUDENT4_WRAPPER, "outfits": ranked_outfits},
    }


def summarise(result_data: dict) -> dict:
    recommendations = result_data["recommendation_set"]["recommendations"]
    return {
        "note": TEST_BUDGET_NOTE,
        "test_request": dict(TEST_REQUEST),
        "beta": result_data["beta"],
        "currency_conversion": result_data.get("currency_conversion"),
        "filter_summary": result_data["filter_summary"],
        "top3": recommendations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ranked_outfits", type=Path, help="Student 4 output (JSON list)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    from travel_agent.recommendation.service import run

    ranked_outfits = load_ranked_outfits(args.ranked_outfits)
    result = run(build_payload(ranked_outfits))
    print(f"status: {result.status}")
    if result.status != "ok":
        print(json.dumps(result.model_dump(mode="json")["data"], ensure_ascii=False)[:2000])
        return 1

    data = result.model_dump(mode="json")["data"]
    summary = summarise(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8")

    reasons = sorted(
        {reason["constraint"] for o in data["rejected_outfits"] for reason in o["reasons"]}
    )
    print(f"note: {TEST_BUDGET_NOTE}")
    print(f"beta: {summary['beta']}")
    print(f"filter_summary: {json.dumps(summary['filter_summary'])}")
    print(f"rejection constraints: {reasons}")
    for item in summary["top3"]:
        print(
            f"#{item['display_position']} {item['outfit_id']}: "
            f"compatibility={item['compatibility_score']:.6f} "
            f"relevance={item['outfit_product_relevance']:.6f} "
            f"final={item['final_outfit_score']:.6f} "
            f"total={item['total_purchase_cost']} {item['currency']}"
        )
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
