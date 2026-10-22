"""Offline tests for `scripts.run_student5_real_refinement` (budget change → recommend again).

The Student 4 outfits are MOCK contract fixtures in the category-keyed `products` shape
(arbitrary scores), not the real 276 MB Student 4 output. Images use a fake fetcher; nothing
touches the network and no real preview provider is configured.
"""

import json
from datetime import date
from decimal import Decimal

import pytest

from scripts import run_student5_real_refinement as refinement
from travel_agent.recommendation.currency_converter import ExchangeRateSnapshot
from travel_agent.recommendation.service import run as service_run
from travel_agent.recommendation.student5.models import TripShoppingRequest
from travel_agent.recommendation.student5.preview_adapter import PREVIEW_ENV_KEYS

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


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


def _outfit(oid, compatibility, amounts):
    top, bottom, shoes = amounts
    return {
        "outfit_id": oid,
        "compatibility_score": compatibility,
        "products": {
            "TOP": _product(f"{oid}_top", "TOP", top),
            "BOTTOM": _product(f"{oid}_bottom", "BOTTOM", bottom),
            "FOOTWEAR": _product(f"{oid}_shoes", "FOOTWEAR", shoes),
        },
    }


# Totals (USD): 150, 120, 90, 60, 900. With 500 USD the Top-3 is O0001-O0003; with 100 USD only
# O0003 and O0004 remain.
RANKED = [
    _outfit("O0001", 0.30, (50.0, 60.0, 40.0)),
    _outfit("O0002", 0.29, (40.0, 50.0, 30.0)),
    _outfit("O0003", 0.28, (30.0, 40.0, 20.0)),
    _outfit("O0004", 0.27, (20.0, 30.0, 10.0)),
    _outfit("O0005", 0.99, (400.0, 300.0, 200.0)),
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


def _recording_run(calls):
    def run(payload):
        calls.append(payload)
        return service_run(payload)

    return run


def test_budget_is_modified_after_the_first_recommend_and_then_recommended_again():
    calls = []
    results = refinement.run_refinement(RANKED, "100", run=_recording_run(calls))
    assert [c["action"] for c in calls] == ["recommend", "modify_requirements", "recommend"]
    assert calls[1]["patch"] == {"intent": "MODIFY_REQUIREMENTS", "set": {"budget": "100"}}
    # modify_requirements uses the state returned by the first recommend (round 1, its Top-3).
    assert calls[1]["state"]["recommendation_round"] == 1
    assert calls[1]["state"]["last_recommended_outfit_ids"] == ["O0001", "O0002", "O0003"]
    mod = results["modification_result"]
    assert (mod["changed"], mod["changed_fields"]) == (True, ["budget"])
    assert {10, 11, 12} <= set(mod["rerun_plan"]["affected_stages"])
    assert mod["students_rerun"] == ["student5"]


def test_only_the_budget_changes_in_the_updated_request():
    results = refinement.run_refinement(RANKED, "100")
    original = TripShoppingRequest.model_validate(
        results["modification_result"]["original_request"]
    ).model_dump(mode="json")
    updated = results["modification_result"]["updated_request"]
    assert updated["budget"] == "100"
    assert {k: v for k, v in updated.items() if k != "budget"} == {
        k: v for k, v in original.items() if k != "budget"
    }


def test_second_recommend_uses_the_updated_request_and_state():
    calls = []
    results = refinement.run_refinement(RANKED, "100", run=_recording_run(calls))
    second = calls[2]
    assert second["request"] == results["modification_result"]["updated_request"]
    assert second["state"]["request_version"] == 2
    assert second["student4"] == calls[0]["student4"]  # same Student 4 candidates and scores
    assert results["student5_top3"]["recommendation_round"] == 2
    assert results["student5_top3"]["test_request"]["budget"] == "100"


def test_failed_modification_stops_before_recommending_again():
    calls = []
    with pytest.raises(refinement.RefinementError, match="changed"):
        refinement.run_refinement(RANKED, "500", run=_recording_run(calls))  # same value: no change
    assert [c["action"] for c in calls] == ["recommend", "modify_requirements"]


def test_comparison_lists_added_removed_and_unchanged_outfits():
    comparison = refinement.run_refinement(RANKED, "100")["comparison"]
    assert (comparison["original_budget"], comparison["new_budget"]) == ("500", "100")
    assert comparison["original_top3"] == ["O0001", "O0002", "O0003"]
    assert comparison["new_top3"] == ["O0003", "O0004"]
    assert comparison["added_outfits"] == ["O0004"]
    assert comparison["removed_outfits"] == ["O0001", "O0002"]
    assert comparison["unchanged_outfits"] == ["O0003"]
    assert (comparison["original"]["feasible_count"], comparison["original"]["rejected_count"]) == (
        4,
        1,
    )
    refined = comparison["refined"]
    assert (refined["feasible_count"], refined["rejected_count"], refined["budget_rejections"]) == (
        2,
        3,
        3,
    )
    assert [o["total_purchase_cost"] for o in refined["outfits"]] == ["90.00", "60.00"]
    assert comparison["budget_currency"] == "USD"
    assert comparison["display_currency"] == "USD"


def test_results_are_saved_in_their_own_folder_with_images_and_sheet(tmp_path):
    results = refinement.run_refinement(RANKED, "100")
    original = tmp_path / "student5_top3.json"
    original.write_text('{"untouched": true}', encoding="utf-8")
    out = tmp_path / "refinement_budget_100"
    pil = pytest.importorskip("PIL.Image")
    summary = refinement.save_results(results, out, fetcher=lambda url: ("image/jpeg", JPEG))
    assert json.loads(original.read_text(encoding="utf-8")) == {"untouched": True}
    for name in ("modification_result.json", "student5_top3.json", "comparison.json"):
        assert (out / name).is_file()
    assert sorted(p.name for p in (out / "top3_images").iterdir() if p.is_dir()) == [
        "01_O0003",
        "02_O0004",
    ]
    assert summary["downloaded"] == 6 and summary["failed"] == 0
    with pil.open(out / "top3_contact_sheet.png") as sheet:
        assert sheet.size[1] > 0
    saved = json.loads((out / "student5_top3.json").read_text(encoding="utf-8"))
    assert saved["test_request"]["budget"] == "100"
    text = json.dumps(saved)
    for key in ('"embedding":', '"provenance":', '"evidence":'):
        assert key not in text


def test_main_refuses_to_write_over_the_original_result_folder(tmp_path, monkeypatch):
    ranked = tmp_path / "ranked.json"
    ranked.write_text(json.dumps(RANKED), encoding="utf-8")
    code = refinement.main(
        [str(ranked), "--budget", "100", "--output-dir", str(refinement.DEFAULT_ROOT)]
    )
    assert code == 1


# ---------------------------------------------------------------------------
# After the refined Top-3: export cards, then confirm -> preview
# ---------------------------------------------------------------------------


def _fake_fetch(url):
    return "image/jpeg", JPEG


def _demo(tmp_path, monkeypatch, run=None, **kwargs):
    """run_demo with a recorded order of service actions and the card export step."""
    events = []
    real_export = refinement.export_images

    def recording_export(*args, **kw):
        events.append("export")
        return real_export(*args, **kw)

    def recording_run(payload):
        events.append(payload["action"])
        return (run or service_run)(payload)

    monkeypatch.setattr(refinement, "export_images", recording_export)
    results = refinement.run_demo(
        RANKED,
        "100",
        tmp_path / "refinement_budget_100",
        run=recording_run,
        fetcher=_fake_fetch,
        sheet=False,
        **kwargs,
    )
    return results, events


def test_full_order_and_confirm_uses_the_second_round_state(tmp_path, monkeypatch):
    seen = {}

    def run(payload):
        if payload["action"] == "confirm":
            seen["state"] = payload["state"]
            seen["ids"] = [r["outfit_id"] for r in payload["recommendation_set"]["recommendations"]]
        return service_run(payload)

    results, events = _demo(tmp_path, monkeypatch, run=run)
    assert events == [
        "recommend",
        "modify_requirements",
        "recommend",
        "export",
        "confirm",
        "preview",
    ]
    assert seen["state"]["recommendation_round"] == 2
    assert seen["state"]["current_request"]["budget"] == "100"
    assert seen["state"]["request_version"] == 2
    assert seen["ids"] == ["O0003", "O0004"]  # the refined Top-3, not the original one


def test_default_selection_confirms_and_falls_back_without_a_real_provider(tmp_path, monkeypatch):
    results, _ = _demo(tmp_path, monkeypatch)
    out = tmp_path / "refinement_budget_100"
    confirmed = json.loads((out / "confirmed_outfit.json").read_text(encoding="utf-8"))
    preview = json.loads((out / "preview_result.json").read_text(encoding="utf-8"))
    assert confirmed["selected_outfit_id"] == "O0003"  # first of the refined Top-3
    assert (confirmed["confirm_succeeded"], confirmed["session_status"]) == (True, "CONFIRMED")
    assert confirmed["recommendation_round"] == 2
    assert preview["preview_completed"] is True
    assert (preview["preview_result_status"], preview["delivered_mode"]) == (
        "FALLBACK",
        "PRODUCT_CARDS",
    )
    assert preview["image_generated"] is False
    assert "not configured" in preview["fallback_reason"]
    assert (out / "top3_images" / "download_summary.json").is_file()


def test_selection_outside_the_refined_top3_stops_before_confirm(tmp_path, monkeypatch):
    with pytest.raises(refinement.FlowError, match="Top-3"):
        _demo(tmp_path, monkeypatch, select="O0001")  # was in the 500 USD Top-3 only
    out = tmp_path / "refinement_budget_100"
    assert (out / "top3_images" / "download_summary.json").is_file()  # cards already exported
    assert not (out / "confirmed_outfit.json").exists()


def test_failed_confirm_does_not_run_preview(tmp_path, monkeypatch):
    def run(payload):
        if payload["action"] == "confirm":
            payload = {**payload, "outfit_id": "not-in-the-set"}
        return service_run(payload)

    results, events = _demo(tmp_path, monkeypatch, run=run)
    assert events[-1] == "confirm" and "preview" not in events
    assert results["confirmed_outfit"]["confirm_succeeded"] is False
    preview = results["preview_result"]
    assert (preview["preview_completed"], preview["image_generated"]) == (False, False)


def test_mock_preview_is_not_reported_as_a_generated_image(tmp_path, monkeypatch):
    results, _ = _demo(tmp_path, monkeypatch, select="O0004", provider="mock")
    preview = results["preview_result"]
    assert (preview["preview_source"], preview["preview_result_status"]) == ("mock", "SUCCESS")
    assert preview["image_generated"] is False
    assert preview["fallback_reason"] is None


def test_formal_refinement_rejects_a_non_real_preview(tmp_path, monkeypatch):
    results, _ = _demo(tmp_path, monkeypatch)
    with pytest.raises(refinement.FlowError, match="real OpenAI preview"):
        refinement.require_real_preview(results["preview_result"])
