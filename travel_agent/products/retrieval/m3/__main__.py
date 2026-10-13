import argparse
import json
import sys
from pathlib import Path

from ..catalogue import CatalogueError
from .pipeline import process_catalogue, process_records


def main():
    parser = argparse.ArgumentParser(description="M3 normalization demo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--catalogue", type=Path)
    group.add_argument("--input", type=Path, help="JSON array of unprocessed M3 records")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.input:
            rows = json.loads(args.input.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError("input must be a JSON array")
            report = process_records(rows, source=str(args.input.resolve()))
        else:
            report = process_catalogue(args.catalogue or Path("data/catalogue.json"))
        payload = report.model_dump_json(indent=2)
        if args.output:
            args.output.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        return 0
    except (OSError, UnicodeError, ValueError, CatalogueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
