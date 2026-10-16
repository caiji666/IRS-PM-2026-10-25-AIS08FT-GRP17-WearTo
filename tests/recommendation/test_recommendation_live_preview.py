"""Opt-in live Stage 14 test: one real, PAID image generation through the main-agent tool.

Skipped by default and in CI. It runs only when all of these hold:

    RUN_LIVE_PREVIEW_TESTS=1         test-only gate; never used by the production code
    PREVIEW_PROVIDER=openai
    PREVIEW_API_KEY and PREVIEW_MODEL set in the environment (never in a file in the repo)
    the preview extra installed:     uv sync --locked --extra preview

Each run generates exactly one small FLAT_LAY image (no portrait, no personal data) and costs
one image-generation request on the key's account.

Where the image goes: OpenAIImagePreviewProvider's default output folder is the relative path
`artifacts/previews/`. The test switches the working directory to pytest's `tmp_path` with
`monkeypatch.chdir` (undone after the test), so that folder resolves inside `tmp_path`. No
default, factory or global setting is changed, and the test asserts that no new file appears in
the repository's own `artifacts/previews/`.

A product-card fallback is a failure: the test requires `preview_source == "openai"`, status
SUCCESS, a generative delivered mode and a readable image file.

    RUN_LIVE_PREVIEW_TESTS=1 uv run pytest -m live_preview tests/recommendation
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from travel_agent.recommendation.student5.models import PreviewMode, PreviewStatus
from travel_agent.recommendation.tools import validate_and_recommend

ROOT = Path(__file__).resolve().parents[2]
REPO_PREVIEWS = ROOT / "artifacts" / "previews"

pytestmark = pytest.mark.live_preview


def _skip_reason() -> str | None:
    if os.environ.get("RUN_LIVE_PREVIEW_TESTS") != "1":
        return "set RUN_LIVE_PREVIEW_TESTS=1 to run the paid live preview test"
    if os.environ.get("PREVIEW_PROVIDER", "").strip() != "openai":
        return "PREVIEW_PROVIDER=openai is not set"
    for name in ("PREVIEW_API_KEY", "PREVIEW_MODEL"):
        if not os.environ.get(name, "").strip():
            return f"{name} is not set"
    for module in ("openai", "PIL", "requests"):
        if importlib.util.find_spec(module) is None:
            return "the preview extra is not installed (uv sync --locked --extra preview)"
    return None


REQUEST = {
    "destination": "Seoul",
    "start_date": "2026-12-20",
    "end_date": "2026-12-27",
    "activities": ["sightseeing"],
    "budget": "300",
    "currency": "SGD",
    "style": ["minimalist"],
}


def _item(pid, category, title, price_minor):
    return {
        "product_id": pid,
        "source": "mock_contract_fixture",
        "title": title,
        "category": category,
        "price_minor": price_minor,
        "currency": "SGD",
        "availability": True,
        "is_user_owned": False,
        "product_score": 0.6,
        # Required for purchasable products; only a link, never downloaded or sent as an image.
        "product_url": f"https://shop.example.com/{pid}",
    }


STUDENT4 = {
    "status": "ok",
    "model_version": "mock-compat-0.1",
    "source": "mock_contract_fixture",
    "outfits": [
        {
            "outfit_id": "live_a",
            "compatibility_score": 0.9,
            "items": [
                _item("live_top", "TOP", "Grey wool crew-neck sweater", 6000),
                _item("live_bottom", "BOTTOM", "Black straight-leg trousers", 5000),
                _item("live_shoes", "FOOTWEAR", "White leather sneakers", 5000),
            ],
        }
    ],
}


def test_live_flat_lay_preview_from_the_tool(tmp_path, monkeypatch):
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    from PIL import Image  # installed: checked by _skip_reason

    repo_files_before = _files_under(REPO_PREVIEWS)
    monkeypatch.chdir(tmp_path)  # relative artifacts/previews/ now resolves inside tmp_path

    first = validate_and_recommend.invoke(
        {"payload": {"action": "recommend", "request": REQUEST, "student4": STUDENT4}}
    )
    assert first["status"] == "ok", first
    confirmed = validate_and_recommend.invoke(
        {
            "payload": {
                "action": "confirm",
                "state": first["data"]["state"],
                "recommendation_set": first["data"]["recommendation_set"],
                "outfit_id": "live_a",
            }
        }
    )
    assert confirmed["status"] == "ok", confirmed

    result = validate_and_recommend.invoke(
        {
            "payload": {
                "action": "preview",
                "confirmed_outfit": confirmed["data"]["confirmed_outfit"],
                "preview_mode": "FLAT_LAY",
            }
        }
    )

    assert result["status"] == "ok", result
    preview = result["data"]["preview"]
    # A product-card fallback must not pass as a live success.
    assert result["data"]["preview_source"] == "openai", preview["warnings"]
    assert preview["status"] == PreviewStatus.SUCCESS.value, preview["warnings"]
    assert preview["delivered_mode"] in (PreviewMode.FLAT_LAY.value, PreviewMode.MANNEQUIN.value)
    assert preview["delivered_mode"] == preview["requested_mode"] == "FLAT_LAY"
    assert preview["provider_name"] == "openai"

    output = Path(preview["output_reference"])
    assert not output.is_absolute()
    image_path = (tmp_path / output).resolve()
    assert image_path.is_relative_to(tmp_path.resolve()) and image_path.is_file()
    with Image.open(image_path) as image:
        image.verify()  # raises if the file is not a valid image
    with Image.open(image_path) as image:
        assert image.width > 0 and image.height > 0

    assert _files_under(REPO_PREVIEWS) == repo_files_before
    assert os.environ["PREVIEW_API_KEY"] not in json.dumps(result)


def _files_under(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file()) if folder.is_dir() else []
