import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from .catalogue import CatalogueError, PreparedCatalogueProvider
from .schemas import ProductRetrievalRequest


def main():
    parser = argparse.ArgumentParser(description="Student 3 M1/M2 prepared catalogue demo")
    parser.add_argument("--catalogue", type=Path, default=Path("data/catalogue.json"))
    parser.add_argument("--request", type=Path, default=Path("fixtures/request.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        request = ProductRetrievalRequest.model_validate_json(
            args.request.read_text(encoding="utf-8")
        )
        result = PreparedCatalogueProvider(args.catalogue).retrieve(request)
        payload = result.model_dump_json(indent=2)
        if args.output:
            args.output.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        return 0
    except (OSError, UnicodeError, ValidationError, CatalogueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
