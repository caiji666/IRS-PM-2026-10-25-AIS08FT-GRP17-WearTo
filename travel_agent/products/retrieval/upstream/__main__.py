import argparse
import json
import sys
from pathlib import Path

from .adapter import adapt_files


def main():
    p = argparse.ArgumentParser(
        description="Adapt Student 1/2 JSON, text and safe PT into M1 request + provenance"
    )
    p.add_argument("--pt", type=Path, required=True)
    p.add_argument("--text", type=Path, required=True)
    p.add_argument("--context", type=Path, required=True)
    p.add_argument("--model-info", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    try:
        outputs = [
            a.output_dir / n for n in ["request.json", "request_text_only.json", "metadata.json"]
        ]
        inputs = [a.pt, a.text, a.context] + ([a.model_info] if a.model_info else [])
        if {x.resolve() for x in outputs} & {x.resolve() for x in inputs}:
            raise ValueError("output would overwrite input")
        req, fallback, metadata = adapt_files(a.pt, a.text, a.context, a.model_info)
        a.output_dir.mkdir(parents=True, exist_ok=True)
        for path, data in zip(
            outputs, [req.model_dump(mode="json"), fallback.model_dump(mode="json"), metadata]
        ):
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        print(
            f"Saved {a.output_dir}; references={len(req.top_social_references)}; status={metadata['status']}"
        )
        if metadata["status"] != "declared_compatible":
            print(
                "Imported-vector ranking blocked by existing compatibility checks; await confirmed model metadata."
            )
        return 0
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
