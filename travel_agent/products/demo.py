"""Invoke the actual LangChain tool with an explicit local payload."""

import argparse
import json
from pathlib import Path

from travel_agent.products.tools import search_products


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--understanding", type=Path, help="Student 1 ModuleResult JSON")
    parser.add_argument("--wardrobe-from", help="Existing store path under STUDENT3_DATA_ROOT")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    if args.understanding:
        payload["understanding"] = json.loads(args.understanding.read_text())
    if args.wardrobe_from:
        from travel_agent.products.retrieval.integration import Settings
        from travel_agent.products.retrieval.phase2.service import load_store

        settings = Settings.from_environment()
        old = load_store(settings.input_path(args.wardrobe_from))
        payload["wardrobe_images"] = [str(settings.input_path(i.image_path)) for i in old.items]
    result = search_products.invoke({"payload": payload})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    groups = result.get("data", {}).get("bundle", {}).get("top_k_products", {})
    print(
        json.dumps(
            {
                "status": result["status"],
                "reason": result.get("reason"),
                "returned_per_category": {k: len(v) for k, v in groups.items()},
            }
        )
    )
    return 1 if result["status"] in {"error", "needs_input"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
