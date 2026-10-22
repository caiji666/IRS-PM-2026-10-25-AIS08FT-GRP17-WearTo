"""Bind recommendation sessions and public image URLs to a single live run."""

import copy
import json
import re
from pathlib import Path

RUN_ID = re.compile(r"^\d{8}_\d{6}_\d{6}\+0800$")


def load_session(root, stamp, run):
    if not isinstance(stamp, str) or not RUN_ID.fullmatch(stamp):
        raise ValueError("Choose a completed travel result before loading outfits")
    folder = root / "runs/outfit_generation" / stamp
    source = folder / "student5_input.json"
    if not source.is_file():
        raise ValueError("This saved journey has no outfit output. Run a new journey first.")
    payload = json.loads(source.read_text())
    if payload.get("action") != "recommend" or not payload.get("student4", {}).get("outfits"):
        raise ValueError("The current journey has no compatible outfit combinations")
    result = run(payload)
    if result.status != "ok":
        raise ValueError("Recommendations could not be prepared for this journey")
    data = result.model_dump(mode="json")["data"]
    session = {
        "run_id": stamp,
        "request": payload["request"],
        "student4": payload["student4"],
        "state": data["state"],
        "recommendation_set": data["recommendation_set"],
        "filter_summary": data["filter_summary"],
        "confirmed_outfit": None,
        "preview": None,
    }
    target = root / "runs/recommendation" / stamp / "result.json"
    from travel_agent.outfits.output import save_artifact

    report = json.loads(
        (root / "runs/product_retrieval_diagnostics" / stamp / "run_report.json").read_text()
    )
    save_artifact(result.model_dump(mode="json"), target, report)
    return session


def public_session(root, session):
    from .student5_session import public

    result = copy.deepcopy(public(session))
    result["run_id"] = session.get("run_id")
    for outfit in result["top3"]:
        for item in outfit.get("purchase_items", []) + outfit.get("existing_items", []):
            reference = item.get("image_url")
            if not reference or reference.startswith(("https://", "http://")):
                continue
            path = Path(reference).resolve()
            product_scope = (root / "runs/product_retrieval").resolve()
            wardrobe_scope = (root / "runs/wardrobe/images").resolve()
            if path.is_relative_to(product_scope):
                relative = path.relative_to(product_scope)
                item["image_url"] = f"/media/{relative.as_posix()}"
            elif path.is_relative_to(wardrobe_scope) and re.fullmatch(
                r"[a-f0-9]{64}\.png", path.name
            ):
                item["image_url"] = f"/wardrobe-media/{path.name}"
            else:
                item["image_url"] = None
    return result
