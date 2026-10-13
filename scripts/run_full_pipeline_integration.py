"""Module 11: run the five-module pipeline once and save a run record.

Uses the same harness as tests/test_full_pipeline_integration.py. Run from the repository root:

    uv run python -m scripts.run_full_pipeline_integration [--output-dir DIR] [--allow-models]

Offline by default: no model, network, credentials, `.env` or image API. The run then stops at
Student 2 with `prerequisite_missing`.

`--allow-models` (or RUN_FULL_PIPELINE_TESTS=1) enables the live stages:
- Student 2: FashionCLIP plus Unsplash retrieval over the network; needs a locally cached model
  and UNSPLASH_ACCESS_KEY in the environment. Student 2 itself also calls `load_dotenv()`.
- Student 3: FashionCLIP product retrieval on its configured data source.
Prerequisites are checked offline first; the harness downloads nothing.

Only a run in which every real module completed writes `full_pipeline_run.json`. Otherwise
`full_pipeline_run.pending.json` is written. It is never reported as passed, and its status says
why the run stopped:
- `blocked_by_missing_prerequisites`: model, extra, credentials, data or opt-in missing;
- `blocked_by_undefined_handoff`: no agreed contract yet;
- `blocked_by_missing_modules`: a scaffold returned `not_implemented`;
- `stopped_by_module_status`: a real module returned `needs_input` / `no_candidates` / `error`.

Exit codes:
- 0: completed and checks passed;
- 1: contract or check failure, or a real module stopped the run (needs_input, no_candidates,
  error);
- 2: blocked by a missing prerequisite, an undefined handoff or a scaffold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("artifacts/integration")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-models", action="store_true")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--top-n", type=int, default=None, help="Student 2 top_n (its default)")
    args = parser.parse_args(argv)
    try:
        from tests import full_pipeline_harness as harness
    except ImportError as exc:
        print(f"Run from the repository root with `python -m`: {exc}", file=sys.stderr)
        return 1

    allow = args.allow_models or harness.models_allowed()
    try:
        run = harness.run_pipeline(
            harness.reviewed_payload(), allow_models=allow, beta=args.beta, top_n=args.top_n
        )
        if run.completed:
            dialogue = harness.confirm_and_preview(run)
            checks = harness.check_student5_result(run)
        else:
            dialogue, checks = None, None
        record = harness.build_record(
            run,
            input_description=harness.REVIEWED_PAYLOAD_SOURCE,
            dialogue=dialogue,
            checks=checks,
        )
    except harness.HandoffContractError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    name = "full_pipeline_run.json" if run.completed else "full_pipeline_run.pending.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / name).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"{record['status']}: wrote {name}")
    if not run.completed:
        print(run.stop_reason())
        return 2 if record["status"] in harness.PENDING_STATUSES.values() else 1
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
