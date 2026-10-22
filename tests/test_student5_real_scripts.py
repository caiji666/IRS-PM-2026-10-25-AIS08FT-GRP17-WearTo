"""Offline tests for the Student 5 real-data scripts.

`scripts.export_student5_top3_assets` (image folders, contact sheet) and
`scripts.run_student5_real_flow` (recommend → confirm → preview). The Student 4 outfits are
MOCK contract fixtures in the category-keyed `products` shape (arbitrary scores), not the real
276 MB Student 4 output. Downloads use a fake fetcher; nothing touches the network, and the
PREVIEW_* variables are removed so no real image provider is selected.
"""

import io
import json
import os
from datetime import date
from decimal import Decimal

import pytest

from scripts import export_student5_top3_assets as export
from scripts import run_student5_real_flow as flow
from travel_agent.recommendation.currency_converter import ExchangeRateSnapshot
from travel_agent.recommendation.service import run as service_run
from travel_agent.recommendation.student5.preview_adapter import PREVIEW_ENV_KEYS

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _product(pid, category, amount):
    return {
        "product_id": pid,
        "title": pid,
        "category": category,
        "source_type": "ECOMMERCE_API",
        "price": {"amount": amount, "currency": "USD"},
        "sizes": ["M"],
        "availability": "IN_STOCK",
        "evidence": {"availability_verified": True, "sizes_verified": False},
        "product_url": f"https://shop.example.com/{pid}",
        "provenance": {
            "source": "fixture",
            "original": {"api_item": {"image": {"imageUrl": f"https://img.example.com/{pid}.jpg"}}},
        },
        "owned": False,
        "embedding": {"values": [0.1, 0.2]},
        "product_score": 0.4,
    }


def _outfit(oid, compatibility, amounts, outerwear=None):
    top, bottom, shoes = amounts
    products = {
        "TOP": _product(f"{oid}_top", "TOP", top),
        "BOTTOM": _product(f"{oid}_bottom", "BOTTOM", bottom),
        "FOOTWEAR": _product(f"{oid}_shoes", "FOOTWEAR", shoes),
    }
    if outerwear is not None:
        products["OUTERWEAR"] = _product(f"{oid}_coat", "OUTERWEAR", outerwear)
    return {"outfit_id": oid, "compatibility_score": compatibility, "products": products}


RANKED = [
    _outfit("O0001", 0.30, (50.0, 60.0, 40.0), outerwear=45.0),
    _outfit("O0002", 0.29, (40.0, 50.0, 30.0)),
    _outfit("O0003", 0.28, (30.0, 40.0, 20.0)),
    _outfit("O0004", 0.27, (20.0, 30.0, 10.0)),
    _outfit("O0005", 0.99, (400.0, 300.0, 200.0)),  # over the 500 USD test budget
]


@pytest.fixture(autouse=True)
def _no_preview_configuration(monkeypatch):
    for name in PREVIEW_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    from travel_agent.recommendation import service

    monkeypatch.setattr(
        service,
        "load_ecb_snapshot",
        lambda: ExchangeRateSnapshot(
            rate_date=date(2026, 10, 21),
            rates_per_eur={"EUR": Decimal("1"), "USD": Decimal("1"), "SGD": Decimal("1")},
            source="ECB_TEST_FIXTURE",
        ),
    )


@pytest.fixture
def top3():
    result = service_run(flow.build_payload(RANKED))
    assert result.status == "ok"
    data = result.model_dump(mode="json")["data"]
    return {"top3": data["recommendation_set"]["recommendations"]}


def _fetcher(failing=()):
    calls = []

    def fake(url):
        calls.append(url)
        if any(part in url for part in failing):
            raise TimeoutError("timed out")
        return "image/jpeg", JPEG

    fake.calls = calls
    return fake


# ---------------------------------------------------------------------------
# Image export
# ---------------------------------------------------------------------------


def test_folders_are_named_by_rank_and_outfit_id(tmp_path, top3):
    export.export_images(top3, tmp_path, _fetcher())
    folders = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert folders == ["01_O0001", "02_O0002", "03_O0003"]
    assert sorted(p.name for p in (tmp_path / "01_O0001").iterdir()) == [
        "BOTTOM.jpg",
        "FOOTWEAR.jpg",
        "OUTERWEAR.jpg",
        "TOP.jpg",
        "outfit.json",
    ]
    assert "OUTERWEAR.jpg" not in {p.name for p in (tmp_path / "02_O0002").iterdir()}


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "javascript:alert(1)"],
)
def test_unsafe_or_missing_urls_are_rejected_without_fetching(tmp_path, url):
    fetcher = _fetcher()
    with pytest.raises(export.ImageDownloadError, match="http/https"):
        export.download_image(url, tmp_path / "TOP", fetcher)
    assert fetcher.calls == []


def test_one_failed_image_does_not_stop_the_others(tmp_path, top3):
    summary = export.export_images(top3, tmp_path, _fetcher(failing=("O0001_bottom",)))
    assert summary["failed"] == 1
    assert summary["downloaded"] == 4 + 3 + 3 - 1
    first = summary["outfits"][0]["images"]
    assert first["BOTTOM"] == "download failed (TimeoutError)"
    assert first["TOP"] == "ok"
    outfit = json.loads((tmp_path / "01_O0001" / "outfit.json").read_text(encoding="utf-8"))
    bottom = next(p for p in outfit["products"] if p["category"] == "BOTTOM")
    assert (bottom["image_status"], bottom["image_file"]) == ("failed", None)
    saved = json.loads((tmp_path / "download_summary.json").read_text(encoding="utf-8"))
    assert saved["failed"] == 1


def test_contact_sheet_is_generated_with_placeholders(tmp_path, top3):
    image_module = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image_module.new("RGB", (60, 120), (120, 90, 60)).save(buffer, format="JPEG")
    fetched = buffer.getvalue()
    export.export_images(
        top3,
        tmp_path / "imgs",
        lambda url: (
            ("image/jpeg", fetched) if "bottom" not in url else (_ for _ in ()).throw(OSError())
        ),
    )
    output = export.contact_sheet(top3, tmp_path / "imgs", tmp_path / "sheet.png")
    with image_module.open(output) as sheet:
        expected_width = export.INFO_WIDTH + 4 * (export.TILE + export.PADDING) + export.PADDING
        assert sheet.size == (expected_width, 3 * export.ROW_HEIGHT + 60)
        # The missing BOTTOM image of row 1 is a placeholder with a red border.
        x = export.INFO_WIDTH + 1 * (export.TILE + export.PADDING) + 5
        y = 60 + export.PADDING + 5
        assert sheet.convert("RGB").getpixel((x, y)) == (200, 40, 40)


# ---------------------------------------------------------------------------
# recommend → confirm → preview
# ---------------------------------------------------------------------------


def _recording_run(calls, fail=None):
    def run(payload):
        action = payload["action"]
        calls.append(action)
        if action == fail:
            payload = {**payload, "outfit_id": "not-in-the-set"}
        return service_run(payload)

    return run


def test_order_is_recommend_confirm_preview():
    calls = []
    confirmed, preview = flow.run_flow(RANKED, "O0002", run=_recording_run(calls))
    assert calls == ["recommend", "confirm", "preview"]
    assert confirmed["confirm_status"] == "ok" and confirmed["session_status"] == "CONFIRMED"
    assert confirmed["confirmed_outfit"]["selected_outfit_id"] == "O0002"
    assert preview["selected_outfit_id"] == "O0002"


def test_preview_is_not_run_when_confirm_fails():
    calls = []
    confirmed, preview = flow.run_flow(RANKED, "O0001", run=_recording_run(calls, fail="confirm"))
    assert calls == ["recommend", "confirm"]
    assert confirmed["confirm_status"] == "needs_input"
    assert preview["preview_status"] is None and preview["image_generated"] is False


@pytest.mark.parametrize("select", ["O0004", "O9999"])
def test_selection_must_be_in_the_current_top3(select):
    calls = []
    with pytest.raises(flow.FlowError, match="Top-3"):
        flow.run_flow(RANKED, select, run=_recording_run(calls))
    assert calls == ["recommend"]


def test_server_provider_without_configuration_is_a_fallback_not_an_image():
    _, preview = flow.run_flow(RANKED, "O0001", provider="server")
    assert preview["preview_source"] == "product_card_fallback"
    assert preview["preview_result_status"] == "FALLBACK"
    assert preview["delivered_mode"] == "PRODUCT_CARDS"
    assert preview["image_generated"] is False
    assert "No AI image was generated" in preview["explanation"]
    assert preview["warnings"]


def test_mock_is_never_reported_as_a_generated_image():
    _, preview = flow.run_flow(RANKED, "O0003", "FLAT_LAY", provider="mock")
    assert preview["preview_source"] == "mock"
    assert preview["preview_result_status"] == "SUCCESS"
    assert preview["output_reference"] == "mock://preview/O0003/flat_lay"
    assert preview["image_generated"] is False
    assert "Mock provider" in preview["explanation"]


def test_output_json_structure(tmp_path, monkeypatch):
    ranked = tmp_path / "ranked.json"
    ranked.write_text(json.dumps(RANKED), encoding="utf-8")
    image = tmp_path / "real-preview.png"
    image.write_bytes(PNG)
    monkeypatch.setattr(flow, "load_real_preview_environment", lambda: None)
    real_run_flow = flow.run_flow

    def fake_formal_flow(*args, **kwargs):
        confirmed, preview = real_run_flow(RANKED, "O0001", provider="mock")
        preview.update(
            preview_source="openai",
            preview_result_status="SUCCESS",
            image_generated=True,
            output_reference=str(image),
        )
        return confirmed, preview

    monkeypatch.setattr(flow, "run_flow", fake_formal_flow)
    code = flow.main([str(ranked), "--select", "O0001", "--output-dir", str(tmp_path / "out")])
    assert code == 0
    confirmed = json.loads((tmp_path / "out" / "confirmed_outfit.json").read_text("utf-8"))
    preview = json.loads((tmp_path / "out" / "preview_result.json").read_text("utf-8"))
    assert {"selected_outfit_id", "confirm_status", "confirmed_outfit", "note"} <= set(confirmed)
    assert {
        "selected_outfit_id",
        "confirm_status",
        "preview_status",
        "requested_mode",
        "delivered_mode",
        "preview_source",
        "provider_name",
        "output_reference",
        "warnings",
        "image_generated",
    } <= set(preview)
    assert "model_name" not in preview
    saved = json.dumps(confirmed) + json.dumps(preview)
    for key in ('"embedding":', '"garment_embedding":', '"provenance":', '"evidence":'):
        assert key not in saved
    assert "sk-" not in saved


def test_formal_main_loads_repo_env_and_requires_a_real_preview(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PREVIEW_PROVIDER=openai\n"
        "PREVIEW_API_KEY=test-secret\n"
        "PREVIEW_MODEL=test-image-model\n"
        "PREVIEW_ALLOWED_IMAGE_HOSTS=i.ebayimg.com\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(flow, "REPO_ENV_FILE", env_file)
    for name in PREVIEW_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)

    flow.load_real_preview_environment()

    assert os.environ["PREVIEW_PROVIDER"] == "openai"
    assert os.environ["PREVIEW_MODEL"] == "test-image-model"
    assert os.environ["PREVIEW_ALLOWED_IMAGE_HOSTS"] == "i.ebayimg.com"


def test_real_preview_check_rejects_fallback_and_mock(tmp_path):
    fallback = {
        "preview_source": "product_card_fallback",
        "preview_result_status": "FALLBACK",
        "image_generated": False,
        "output_reference": None,
    }
    with pytest.raises(flow.FlowError, match="real OpenAI preview"):
        flow.require_real_preview(fallback)

    mock = {
        "preview_source": "mock",
        "preview_result_status": "SUCCESS",
        "image_generated": False,
        "output_reference": "mock://preview/O0001/flat_lay",
    }
    with pytest.raises(flow.FlowError, match="real OpenAI preview"):
        flow.require_real_preview(mock)

    image = tmp_path / "preview.png"
    image.write_bytes(PNG)
    real = {
        "preview_source": "openai",
        "preview_result_status": "SUCCESS",
        "image_generated": True,
        "output_reference": str(image),
    }
    assert flow.require_real_preview(real) == image
