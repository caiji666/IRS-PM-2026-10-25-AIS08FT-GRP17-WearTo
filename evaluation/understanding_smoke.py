"""Small developer-labelled local-Qwen smoke set, not an independent benchmark."""

import argparse
import json
import platform
import time
from pathlib import Path

from travel_agent.schemas import TripShoppingRequest, TripShoppingRequestPatch
from travel_agent.understanding.extraction import QwenExtractor
from travel_agent.understanding.state import apply_patch

CASES = [
    {
        "id": "complete_en",
        "message": "I am going to Seoul from 2026-12-20 to 2026-12-27 for "
        "sightseeing and dinner. My shopping budget is 500 SGD. I prefer minimalist outfits. "
        "The dinner dress code is smart casual.",
        "expected": {
            "destination": "Seoul",
            "start_date": "2026-12-20",
            "end_date": "2026-12-27",
            "activities": ["sightseeing", "dinner"],
            "budget": 500,
            "currency": "SGD",
            "style": ["minimalist"],
            "dress_codes": {"dinner": "smart casual"},
            "price_preference": None,
            "enter_interior": None,
        },
    },
    {
        "id": "complete_zh",
        "message": "2026年12月20日至2026年12月27日去首尔观光，购物预算500新币，喜欢极简风。",
        "expected": {
            "destination": "Seoul",
            "start_date": "2026-12-20",
            "end_date": "2026-12-27",
            "activities": ["sightseeing"],
            "budget": 500,
            "currency": "SGD",
            "style": ["minimalist"],
            "price_preference": None,
        },
    },
    {
        "id": "missing_currency_dates",
        "message": "Going to Tokyo for sightseeing, budget 500.",
        "expected": {
            "destination": "Tokyo",
            "activities": ["sightseeing"],
            "budget": 500,
            "currency": None,
            "start_date": None,
            "end_date": None,
            "style": [],
        },
    },
    {
        "id": "cheaper_patch",
        "message": "Make it cheaper and do not include a long coat.",
        "patch": True,
        "expected": {
            "budget": 500,
            "currency": "SGD",
            "destination": "Seoul",
            "excluded_items": ["long coat"],
            "price_preference": "lower",
        },
    },
    {
        "id": "budget_patch_zh",
        "message": "预算改成400新币，其他不变。",
        "patch": True,
        "expected": {
            "budget": 400,
            "currency": "SGD",
            "destination": "Seoul",
            "activities": ["sightseeing", "dinner"],
            "style": ["minimalist"],
        },
    },
    {
        "id": "remove_style",
        "message": "Remove my minimalist style preference. Budget unchanged.",
        "patch": True,
        "expected": {"style": [], "budget": 500, "destination": "Seoul"},
    },
    {
        "id": "unknown_activity",
        "message": "Tokyo, 2026-10-15 to 2026-10-17. I plan scuba diving. Budget 200 SGD.",
        "expected": {
            "destination": "Tokyo",
            "activities": ["scuba diving"],
            "budget": 200,
            "currency": "SGD",
            "style": [],
        },
    },
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen-smoke-results.json"))
    args = parser.parse_args()
    extractor = QwenExtractor(args.model_path, local_files_only=True)
    previous = TripShoppingRequest(
        destination="Seoul",
        start_date="2026-12-20",
        end_date="2026-12-27",
        activities=["sightseeing", "dinner"],
        budget=500,
        currency="SGD",
        style=["minimalist"],
        dress_codes={"dinner": "smart casual"},
    )
    results = []
    for case in CASES:
        start = time.monotonic()
        metadata = {}
        try:
            parsed, metadata = extractor.extract(
                case["message"], previous if case.get("patch") else None
            )
            request = (
                apply_patch(previous, TripShoppingRequestPatch.model_validate(parsed))
                if case.get("patch")
                else TripShoppingRequest.model_validate(parsed)
            )
            values = request.model_dump(mode="json")
            failures = {
                field: {"expected": expected, "actual": values[field]}
                for field, expected in case["expected"].items()
                if values[field] != expected
            }
            result = {
                "id": case["id"],
                "message": case["message"],
                "expected": case["expected"],
                "passed": not failures,
                "failures": failures,
                "parsed": parsed,
                "request": values,
                "metadata": metadata,
            }
        except Exception as exc:
            result = {
                "id": case["id"],
                "message": case["message"],
                "expected": case["expected"],
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "raw_output": getattr(exc, "raw_output", metadata.get("raw_output")),
                "metadata": metadata,
            }
        result["seconds"] = round(time.monotonic() - start, 2)
        results.append(result)
        print(case["id"], "PASS" if result["passed"] else "FAIL", flush=True)
    output = {
        "scope": "Developer smoke set; prompted on these cases; not a blind accuracy estimate",
        "platform": platform.platform(),
        "cases": results,
        "passed": sum(r["passed"] for r in results),
        "total": len(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {args.output}: {output['passed']}/{output['total']} smoke cases", flush=True)


if __name__ == "__main__":
    main()
