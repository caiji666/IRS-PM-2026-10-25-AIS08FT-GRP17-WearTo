"""Local natural-language chat for incremental understanding, not the full tool agent."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from travel_agent.understanding.conversation import Conversation
from travel_agent.understanding.extraction import QwenExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume", type=Path, help="Chat checkpoint or earlier understanding state"
    )
    parser.add_argument("--session-dir", type=Path)
    parser.add_argument("--allow-draft", action="store_true", help="Allow draft KG assertions")
    parser.add_argument(
        "--weather-provider", choices=["open_meteo", "climatology", "none"], default="climatology"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    os.environ.setdefault("HF_HOME", str(root / "artifacts" / "huggingface"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    folder = args.session_dir or root / "artifacts" / "chat" / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    folder.mkdir(parents=True, exist_ok=True)
    saved = json.loads(args.resume.read_text()) if args.resume else None
    settings = {"allow_draft": args.allow_draft, "weather_provider": args.weather_provider}
    extractor = QwenExtractor(local_files_only=True)
    print("Loading local Qwen3-0.6B for extraction and conversation...", flush=True)
    extractor._load()
    chat = Conversation(extractor=extractor, saved=saved, settings=settings)
    print("Ready. English is the primary evaluated language. Type one detail at a time.")
    print(
        "Say confirm after reviewing the summary. "
        "Commands: /state, /result, /suggest, /reset, /quit."
    )
    print("Optional choices: reply with numbers/names, skip for one step, or skip all for both.")
    print(f"KG draft mode: {args.allow_draft}; weather source: {args.weather_provider}.")
    print(f"Local session: {folder}\n", flush=True)
    (folder / "ready.json").write_text(
        json.dumps({"pid": os.getpid(), "model": extractor.model_path})
    )

    def display(output):
        print("\nAssistant > " + output["reply"] + "\n", flush=True)
        (folder / "session.json").write_text(
            json.dumps(chat.checkpoint(), indent=2, ensure_ascii=False) + "\n"
        )
        with (folder / "turns.jsonl").open("a") as log:
            log.write(json.dumps(output, ensure_ascii=False) + "\n")
        if args.debug:
            print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)

    display(chat.opening())
    while True:
        try:
            message = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession saved.")
            break
        if not message:
            continue
        if message == "/quit":
            break
        if message in ("/state", "/result"):
            print(
                json.dumps(
                    chat.checkpoint() if message == "/state" else chat.last_result,
                    indent=2,
                    ensure_ascii=False,
                )
            )
            continue
        if message == "/reset":
            chat = Conversation(extractor=extractor, settings=settings)
            display(chat.opening())
            continue
        try:
            display(chat.turn(message))
        except Exception as exc:
            # Keep the checkpoint available when local inference fails.
            (folder / "session.json").write_text(
                json.dumps(chat.checkpoint(), indent=2, ensure_ascii=False) + "\n"
            )
            print(
                f"Local dialogue error ({type(exc).__name__}): {exc}. Your state was saved.",
                flush=True,
            )


if __name__ == "__main__":
    main()
