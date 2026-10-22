"""Isolated model work for the local web app; never exposes raw artifacts to browsers."""

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlencode


def prepare(directory):
    from travel_agent.understanding.extraction import QwenExtractor
    from travel_agent.understanding.service import run
    from travel_agent.understanding.weather import get_json

    form = json.loads((directory / "form.json").read_text())
    message = (
        f"I am travelling to {form['city']}, {form['country_name']}, "
        f"from {form['start_date']} to {form['end_date']}. "
        f"My total clothing budget is {form['budget']} {form['currency']}. "
    )
    if form.get("activities"):
        message += f"My activities are {', '.join(form['activities'])}. "
    if form.get("style"):
        message += f"I prefer {form['style']} outfits. "
    if form.get("dress_code") and "dinner" in form["activities"]:
        message += f"The dress code for dinner is {form['dress_code']}. "
    if form.get("notes"):
        message += form["notes"]
    payload = {"message": message, "allow_draft": True, "weather_provider": "open_meteo"}
    (directory / "message_input.json").write_text(json.dumps(payload, indent=2) + "\n")
    result = run(payload, extractor=QwenExtractor(local_files_only=True))
    (directory / "extraction.json").write_text(result.model_dump_json(indent=2) + "\n")
    request = result.data.get("proposed_request")
    if not request:
        raise ValueError("Qwen could not extract a complete trip; edit the request and retry.")
    locations = get_json(
        "https://geocoding-api.open-meteo.com/v1/search?"
        + urlencode({"name": request["destination"], "count": 10, "language": "en"})
    ).get("results", [])
    locations = [x for x in locations if x.get("country_code") == form["country"]]
    if not locations:
        raise ValueError("No city found in the selected country; enter a city, not only a country.")
    draft = {
        "request": request,
        "locations": [
            {
                k: x.get(k)
                for k in ("name", "country", "latitude", "longitude", "timezone", "admin1")
            }
            for x in locations
        ],
    }
    (directory / "review.json").write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n")


def chat(directory):
    from .chat import advance_chat

    conversation = json.loads((directory / "conversation.json").read_text())
    conversation, audit, review = advance_chat(conversation)
    if review is not None:
        (directory / "review.json").write_text(
            json.dumps(review, ensure_ascii=False, indent=2) + "\n"
        )
    turns = directory / "turns"
    turns.mkdir(exist_ok=True)
    (turns / f"{conversation['turn']:03d}.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    temporary = directory / "conversation.tmp"
    temporary.write_text(json.dumps(conversation, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(directory / "conversation.json")


def main():
    root = Path.cwd().resolve()
    cache = root / ".cache/huggingface"
    if not (cache / "models--Qwen--Qwen3-0.6B").exists():
        cache = root / "artifacts/huggingface/hub"
    os.environ.setdefault("HF_HUB_CACHE", str(cache))
    os.environ["HF_HUB_OFFLINE"] = "1"
    if len(sys.argv) > 2 and sys.argv[2] == "chat":
        chat(Path(sys.argv[1]))
    elif len(sys.argv) > 2 and sys.argv[2] == "wardrobe":
        from .profile import ingest_upload

        ingest_upload(root, Path(sys.argv[1]))
    else:
        prepare(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
