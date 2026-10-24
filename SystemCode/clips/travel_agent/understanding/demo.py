"""Offline proposal demo, or an explicit local-Qwen/live-weather CLI."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from travel_agent.understanding.extraction import QwenExtractor
from travel_agent.understanding.service import run

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "understanding"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=EXAMPLES / "seoul_winter.json")
    parser.add_argument("--message")
    parser.add_argument("--state", type=Path, help="Read a previous serialized ConversationState")
    parser.add_argument("--save-state", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--live-weather",
        action="store_true",
        help="Route to forecasts or historical reference by coverage; removes fixtures",
    )
    parser.add_argument("--location", type=Path, help="Explicitly confirmed city coordinate JSON")
    parser.add_argument("--model-path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument(
        "--confirm-pending",
        action="store_true",
        help="Apply the accumulated pending request after human review; requires --state",
    )
    parser.add_argument(
        "--accept-extraction",
        action="store_true",
        help="Apply model facts only after reviewing this input's extraction",
    )
    args = parser.parse_args()
    if args.confirm_pending and (not args.state or args.message):
        parser.error("--confirm-pending requires --state and cannot be combined with --message")
    load_dotenv()
    payload = (
        {"accept_extraction": True}
        if args.confirm_pending
        else (
            {"message": args.message}
            if args.message
            else json.loads(args.input.read_text(encoding="utf-8"))
        )
    )
    if args.state:
        payload["state"] = json.loads(args.state.read_text())
    if args.accept_extraction:
        payload["accept_extraction"] = True
    if args.allow_draft:
        payload["allow_draft"] = True
    if args.live_weather:
        payload.pop("weather_fixture", None)
        payload["weather_provider"] = "open_meteo"
    if args.location:
        payload["confirmed_location"] = json.loads(args.location.read_text())
    extractor = (
        QwenExtractor(args.model_path, local_files_only=args.local_files_only)
        if args.message
        else None
    )
    result = run(payload, extractor=extractor)
    text = result.model_dump_json(indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.save_state:
        args.save_state.parent.mkdir(parents=True, exist_ok=True)
        args.save_state.write_text(
            json.dumps(result.data["state"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if result.status == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
