"""Current run selection must never fall back to a fixed recommendation sample."""

import json
from types import SimpleNamespace

import pytest

from travel_agent.web.live_outfits import load_session, public_session

STAMP = "20261022_114026_063796+0800"


def test_current_run_payload_and_budget_are_forwarded(tmp_path, monkeypatch):
    payload = {
        "action": "recommend",
        "request": {"destination": "Tokyo", "budget": "650", "currency": "SGD"},
        "student4": {"outfits": [{"outfit_id": "live-123"}]},
    }
    folder = tmp_path / "runs/outfit_generation" / STAMP
    folder.mkdir(parents=True)
    (folder / "student5_input.json").write_text(json.dumps(payload))
    diagnostics = tmp_path / "runs/product_retrieval_diagnostics" / STAMP
    diagnostics.mkdir(parents=True)
    (diagnostics / "run_report.json").write_text('{"environment":"test"}')
    received = []
    data = {"state": {}, "recommendation_set": {"recommendations": []}, "filter_summary": {}}

    def run(value):
        received.append(value)
        return SimpleNamespace(status="ok", model_dump=lambda **_: {"data": data})

    session = load_session(tmp_path, STAMP, run)
    assert received == [payload]
    assert session["request"] == payload["request"]
    assert session["run_id"] == STAMP
    assert (tmp_path / "runs/recommendation" / STAMP / "result.json").is_file()
    with pytest.raises(ValueError, match="no outfit output"):
        load_session(tmp_path, "20261021_114026_063796+0800", run)
    assert len(received) == 1
    for invalid in (None, "../../secret", "/tmp/result.json"):
        with pytest.raises(ValueError):
            load_session(tmp_path, invalid, run)


def test_public_images_use_scoped_urls_without_mutating_preview_inputs(tmp_path):
    product = tmp_path / "runs/product_retrieval" / STAMP / "product_images/p.png"
    wardrobe = tmp_path / "runs/wardrobe/images" / ("a" * 64 + ".png")
    session = {
        "run_id": STAMP,
        "recommendation_set": {
            "recommendations": [
                {
                    "purchase_items": [{"image_url": str(product)}],
                    "existing_items": [{"image_url": str(wardrobe)}],
                }
            ]
        },
    }
    result = public_session(tmp_path, session)
    items = result["top3"][0]
    assert items["purchase_items"][0]["image_url"] == f"/media/{STAMP}/product_images/p.png"
    assert items["existing_items"][0]["image_url"].startswith("/wardrobe-media/")
    assert session["recommendation_set"]["recommendations"][0]["purchase_items"][0][
        "image_url"
    ] == str(product)
