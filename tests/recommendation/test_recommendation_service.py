"""Team entry point `travel_agent.recommendation.service.run` (Student 5, Stages 10-14).

The Student 4-shaped payloads below are MOCK contract fixtures in the `student4_adapter`
format: their compatibility scores are mock values, not Student 4 model output. They check
the service contract only; real Student 4 -> Student 5 integration is Module 11.
"""

import copy
import json
import os
import subprocess
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from travel_agent.recommendation.currency_converter import ExchangeRateSnapshot
from travel_agent.recommendation.service import ACTIONS, DEFAULT_BETA, run

ROOT = Path(__file__).resolve().parents[2]

REQUEST = {
    "destination": "Seoul",
    "start_date": "2026-12-20",
    "end_date": "2026-12-27",
    "activities": ["sightseeing"],
    "budget": "300",
    "currency": "SGD",
    "style": ["minimalist"],
}


def _item(pid, category, price_minor, score, *, owned=False, available=True, currency="SGD"):
    item = {
        "product_id": pid,
        "source": "mock_contract_fixture",
        "title": pid,
        "category": category,
        "price_minor": 0 if owned else price_minor,
        "currency": currency,
        "availability": available,
        "is_user_owned": owned,
        "product_score": score,
    }
    if not owned:
        item["product_url"] = f"https://shop.example.com/{pid}"
    return item


def _outfit(oid, compatibility, prices, score=0.5, **item_kwargs):
    top, bottom, shoes = prices
    return {
        "outfit_id": oid,
        "compatibility_score": compatibility,
        "items": [
            _item(f"{oid}_top", "TOP", top, score, **item_kwargs),
            _item(f"{oid}_bottom", "BOTTOM", bottom, score),
            _item(f"{oid}_shoes", "FOOTWEAR", shoes, score),
        ],
    }


STUDENT4 = {
    "status": "ok",
    "model_version": "mock-compat-0.1",
    "source": "mock_contract_fixture",
    "outfits": [
        _outfit("o_a", 0.90, (5000, 6000, 9000), 0.8),  # 200.00
        _outfit("o_b", 0.80, (4000, 5000, 8000), 0.7),  # 170.00
        _outfit("o_c", 0.70, (3000, 4000, 7000), 0.6),  # 140.00
        _outfit("o_d", 0.60, (2000, 3000, 6000), 0.5),  # 110.00
        _outfit("o_e", 0.95, (15000, 15000, 15000), 0.9),  # 450.00, over budget
        _outfit("o_f", 0.85, (4000, 4000, 4000), 0.8, available=False),  # TOP not available
    ],
}


def _recommend(**overrides):
    payload = {
        "action": "recommend",
        "request": copy.deepcopy(REQUEST),
        "student4": copy.deepcopy(STUDENT4),
    }
    payload.update(overrides)
    return run(payload)


@pytest.fixture
def first_round():
    result = _recommend()
    assert result.status == "ok", result.reason
    return result


def _selection(result, outfit_id, action="confirm"):
    return {
        "action": action,
        "state": result.data["state"],
        "recommendation_set": result.data["recommendation_set"],
        "outfit_id": outfit_id,
    }


# ---------------------------------------------------------------------------
# Action contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [{}, {"request": REQUEST, "student4": STUDENT4}],
)
def test_missing_action_is_an_error_not_an_implicit_recommend(payload):
    result = run(payload)
    assert result.status == "error"
    assert result.data["error"]["type"] == "InvalidAction" and "action" in result.reason


@pytest.mark.parametrize("action", ["buy", None])
def test_unknown_action_is_an_error(action):
    result = run({"action": action})
    assert result.status == "error"
    assert result.data["error"]["actions"] == list(ACTIONS)


def test_unknown_keys_need_input():
    result = _recommend(api_key="x")
    assert result.status == "needs_input" and "api_key" in result.data["error"]["message"]


# ---------------------------------------------------------------------------
# recommend (Stages 10 -> 12)
# ---------------------------------------------------------------------------


def test_recommend_returns_top3_state_and_filter_summary(first_round):
    data = first_round.data
    ids = [r["outfit_id"] for r in data["recommendation_set"]["recommendations"]]
    assert ids == ["o_a", "o_b", "o_c"]
    assert data["state"]["session_status"] == "AWAITING_CONFIRMATION"
    assert data["state"]["last_recommended_outfit_ids"] == ids
    assert data["filter_summary"] == {
        "total_candidates": 6,
        "feasible_count": 4,
        "rejected_count": 2,
        "conflict_summary": {"BUDGET": 1, "AVAILABILITY": 1, "SIZE": 0},
    }
    assert {r["outfit_id"] for r in data["rejected_outfits"]} == {"o_e", "o_f"}
    assert data["conflict"] is None
    assert data["beta"] == DEFAULT_BETA
    json.dumps(first_round.model_dump(mode="json"))


def test_compatibility_score_source_and_model_version_are_preserved(first_round):
    data = first_round.data
    assert (data["student4_source"], data["model_version"]) == (
        "mock_contract_fixture",
        "mock-compat-0.1",
    )
    raw = {o["outfit_id"]: o["compatibility_score"] for o in STUDENT4["outfits"]}
    for item in data["recommendation_set"]["recommendations"]:
        assert item["compatibility_score"] == raw[item["outfit_id"]]


def test_no_feasible_outfit_returns_structured_conflict():
    result = _recommend(request={**REQUEST, "budget": "50"})
    assert result.status == "no_candidates"
    conflict = result.data["conflict"]
    assert conflict["violated_constraints"] == ["BUDGET", "AVAILABILITY"]
    assert conflict["budget"] == "50" and conflict["currency"] == "SGD"
    assert result.data["recommendation_set"] is None
    assert result.reason == conflict["message"]


@pytest.mark.parametrize(
    "change, code",
    [
        (lambda s: s.update(status="not_implemented"), "status_not_ok"),
        (lambda s: s["outfits"][0]["items"][0].update(price_minor=19.9), "invalid_price"),
    ],
)
def test_student4_contract_violations_are_errors_with_code(change, code):
    student4 = copy.deepcopy(STUDENT4)
    change(student4)
    result = _recommend(student4=student4)
    assert result.status == "error" and result.data["error"]["code"] == code


# ---------------------------------------------------------------------------
# Stage 13: modify_requirements, next round, confirm, explain
# ---------------------------------------------------------------------------


def _modify(result, budget):
    return run(
        {
            "action": "modify_requirements",
            "state": result.data["state"],
            "patch": {"intent": "MODIFY_REQUIREMENTS", "set": {"budget": budget}},
        }
    )


def test_modify_requirements_returns_state_and_rerun_plan(first_round):
    change = _modify(first_round, "180")
    assert change.status == "ok"
    assert change.data["changed"] is True and change.data["changed_fields"] == ["budget"]
    assert change.data["rerun_plan"]["affected_stages"] == [10, 11, 12]
    assert change.data["state"]["last_recommended_outfit_ids"] == []
    assert change.data["state"]["request_version"] == 2
    assert change.data["updated_request"]["budget"] == "180"


def test_second_round_after_change_and_stale_set_is_rejected(first_round):
    change = _modify(first_round, "180")
    second = _recommend(request=change.data["updated_request"], state=change.data["state"])
    assert second.status == "ok", second.reason
    assert second.data["recommendation_set"]["recommendation_round"] == 2
    new_ids = [r["outfit_id"] for r in second.data["recommendation_set"]["recommendations"]]
    assert new_ids == ["o_b", "o_c", "o_d"]

    stale = run({**_selection(first_round, "o_a"), "state": second.data["state"]})
    assert stale.status == "needs_input"
    assert stale.data["error"]["type"] == "StaleRecommendationSetError"
    gone = run(_selection(second, "o_a"))
    assert gone.data["error"]["type"] == "InvalidOutfitSelectionError"

    confirmed = run(_selection(second, "o_b"))
    assert confirmed.status == "ok"
    assert confirmed.data["state"]["session_status"] == "CONFIRMED"
    assert confirmed.data["state"]["selected_outfit_id"] == "o_b"
    assert confirmed.data["confirmed_outfit"]["recommendation"]["outfit_id"] == "o_b"


@pytest.mark.parametrize("outfit_id", ["2", "o_e"])
def test_confirm_accepts_only_an_exact_outfit_id_from_the_latest_set(first_round, outfit_id):
    result = run(_selection(first_round, outfit_id))
    assert result.status == "needs_input"
    assert result.data["error"]["type"] == "InvalidOutfitSelectionError"


# ---------------------------------------------------------------------------
# Stage 14: preview
# ---------------------------------------------------------------------------


@pytest.fixture
def confirmed_outfit(first_round):
    return run(_selection(first_round, "o_a")).data["confirmed_outfit"]


@pytest.mark.parametrize("unconfirmed", ["missing", "recommendation_item"])
def test_preview_without_a_confirmed_outfit_is_rejected_and_calls_no_provider(
    first_round, monkeypatch, unconfirmed
):
    """Preview needs a ConfirmedOutfit from `confirm`; a recommended (unconfirmed) outfit or no
    outfit is rejected, and the server-configured provider is never created or called."""
    from travel_agent.recommendation import preview_provider

    monkeypatch.setenv("PREVIEW_PROVIDER", "openai")
    monkeypatch.setenv("PREVIEW_API_KEY", "sk-test-not-a-real-key-0123456789")
    monkeypatch.setenv("PREVIEW_MODEL", "fake-image-model")
    monkeypatch.setattr(
        preview_provider,
        "create_live_provider",
        lambda environ: pytest.fail("no provider may be created before confirmation"),
    )
    payload = {"action": "preview", "preview_mode": "FLAT_LAY"}
    if unconfirmed == "recommendation_item":
        payload["confirmed_outfit"] = first_round.data["recommendation_set"]["recommendations"][0]
    result = run(payload)
    assert result.status == "needs_input"
    assert "preview" not in result.data


def test_preview_with_mock_provider_is_marked_mock(confirmed_outfit):
    result = run(
        {
            "action": "preview",
            "confirmed_outfit": confirmed_outfit,
            "preview_mode": "FLAT_LAY",
            "preview_provider": "mock",
        }
    )
    assert result.status == "ok"
    assert result.data["preview_source"] == "mock"
    preview = result.data["preview"]
    assert (preview["status"], preview["provider_name"]) == ("SUCCESS", "mock")
    assert preview["output_reference"] == "mock://preview/o_a/flat_lay"


@pytest.mark.parametrize("mode", ["MANNEQUIN", "PRODUCT_CARDS"])
def test_removed_user_preview_modes_are_rejected(confirmed_outfit, mode):
    result = run(
        {
            "action": "preview",
            "confirmed_outfit": confirmed_outfit,
            "preview_mode": mode,
        }
    )
    assert result.status == "needs_input"


def test_portrait_ref_resolves_from_the_managed_upload_store(
    confirmed_outfit, tmp_path, monkeypatch
):
    store = tmp_path / "portraits"
    store.mkdir()
    (store / "portrait_abc123.jpg").write_bytes(b"test portrait placeholder")
    monkeypatch.setenv("PORTRAIT_UPLOAD_ROOT", str(store))

    result = run(
        {
            "action": "preview",
            "confirmed_outfit": confirmed_outfit,
            "preview_mode": "PORTRAIT",
            "preview_provider": "mock",
            "user_portrait_ref": "portrait_abc123",
            "portrait_consent": True,
        }
    )

    assert result.status == "ok"
    assert result.data["preview"]["requested_mode"] == "PORTRAIT"


@pytest.mark.parametrize(
    "portrait_ref",
    [
        "../portrait_abc123",
        "D:/uploads/portrait.jpg",
        "https://example.com/portrait.jpg",
        "missing_portrait",
    ],
)
def test_unsafe_or_missing_portrait_ref_is_rejected(
    confirmed_outfit, tmp_path, monkeypatch, portrait_ref
):
    store = tmp_path / "portraits"
    store.mkdir()
    monkeypatch.setenv("PORTRAIT_UPLOAD_ROOT", str(store))

    result = run(
        {
            "action": "preview",
            "confirmed_outfit": confirmed_outfit,
            "preview_mode": "PORTRAIT",
            "preview_provider": "mock",
            "user_portrait_ref": portrait_ref,
            "portrait_consent": True,
        }
    )

    assert result.status == "needs_input"


def test_raw_user_portrait_path_is_not_a_public_payload_field(confirmed_outfit):
    result = run(
        {
            "action": "preview",
            "confirmed_outfit": confirmed_outfit,
            "preview_mode": "PORTRAIT",
            "preview_provider": "mock",
            "user_portrait": "D:/uploads/portrait.jpg",
            "portrait_consent": True,
        }
    )
    assert result.status == "needs_input"


def test_preview_without_provider_is_marked_product_card_fallback(confirmed_outfit):
    result = run(
        {"action": "preview", "confirmed_outfit": confirmed_outfit, "preview_mode": "FLAT_LAY"}
    )
    assert result.data["preview_source"] == "product_card_fallback"
    preview = result.data["preview"]
    assert (preview["status"], preview["delivered_mode"]) == ("FALLBACK", "PRODUCT_CARDS")
    assert preview["output_reference"] is None


def test_preview_auto_uses_styled_layout_without_portrait(confirmed_outfit, tmp_path, monkeypatch):
    store = tmp_path / "portraits"
    store.mkdir()
    monkeypatch.setenv("PORTRAIT_UPLOAD_ROOT", str(store))
    result = run(
        {"action": "preview_auto", "confirmed_outfit": confirmed_outfit, "preview_provider": "mock"}
    )
    assert result.status == "ok"
    assert result.data["preview"]["requested_mode"] == "FLAT_LAY"


def test_preview_auto_uses_managed_portrait_when_present(confirmed_outfit, tmp_path, monkeypatch):
    store = tmp_path / "portraits"
    store.mkdir()
    (store / "current.png").write_bytes(b"portrait")
    monkeypatch.setenv("PORTRAIT_UPLOAD_ROOT", str(store))
    result = run(
        {"action": "preview_auto", "confirmed_outfit": confirmed_outfit, "preview_provider": "mock"}
    )
    assert result.status == "ok"
    assert result.data["preview"]["requested_mode"] == "PORTRAIT"


# ---------------------------------------------------------------------------
# Tool wrapper and import safety
# ---------------------------------------------------------------------------


IMPORT_PROBE = """
import builtins, io, socket, sys
sys.path.insert(0, sys.argv[1])
def _blocked(*a, **k):
    raise RuntimeError("network access attempted")
socket.socket.connect = _blocked
socket.create_connection = _blocked
opened = []
for owner, name in ((builtins, "open"), (io, "open")):
    real = getattr(owner, name)
    def recording(file, *a, _real=real, **k):
        opened.append(str(file))
        return _real(file, *a, **k)
    setattr(owner, name, recording)
heavy = ("torch", "transformers", "openai", "PIL")
before = {m for m in heavy if m in sys.modules}
import travel_agent.recommendation.service
after = {m for m in heavy if m in sys.modules}
assert after == before, sorted(after - before)
assert not [p for p in opened if p.replace("\\\\", "/").split("/")[-1].startswith(".env")], opened
print("ok", sorted(before))
"""


def test_importing_the_service_alone_loads_no_model_sdk_network_or_env():
    """Fresh, isolated interpreter (-I: no site customisation, no PYTHONPATH, no user site);
    only travel_agent.recommendation.service is imported, so earlier imports elsewhere in the
    test session (e.g. Student 2 loading torch) cannot affect the result."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PREVIEW_", "PYTHON"))}
    result = subprocess.run(
        [sys.executable, "-I", "-c", IMPORT_PROBE, str(ROOT)],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok []"


# ---------------------------------------------------------------------------
# Student 4 category-keyed `products` outfits through the service
# ---------------------------------------------------------------------------
#
# MOCK contract fixtures in the product-object shape that Student 4 copies from Student 3's
# Top-K output. Compatibility and product scores are arbitrary test values.


def _s4_product(pid, category, amount, *, availability="IN_STOCK", verified=True, score=0.5):
    product = {
        "product_id": pid,
        "title": pid,
        "category": category,
        "source_type": "ECOMMERCE_API",
        "price": {"amount": amount, "currency": "SGD", "is_synthetic": False},
        "sizes": ["M"],
        "availability": availability,
        "product_url": f"https://shop.example.com/{pid}",
        "image_path": f"product_images/{pid}.jpg",
        "evidence": {"sizes_verified": False, "availability_verified": verified},
        "provenance": {"source": "fixture", "original": {"api_item": {}}},
        "owned": False,
        "embedding": {"values": [0.1, 0.2]},
        "garment_embedding": {"values": [0.1]},
    }
    if score is not None:
        product["product_score"] = score
    return product


def _s4_outfit(oid, compatibility, amounts, **top_kwargs):
    top, bottom, shoes = amounts
    return {
        "outfit_id": oid,
        "compatibility_score": compatibility,
        "products": {
            "TOP": _s4_product(f"{oid}_top", "TOP", top, **top_kwargs),
            "BOTTOM": _s4_product(f"{oid}_bottom", "BOTTOM", bottom),
            "FOOTWEAR": _s4_product(f"{oid}_shoes", "FOOTWEAR", shoes),
        },
    }


S4_PRODUCTS = {
    "status": "ok",
    "model_version": "mock-compat-0.2",
    "source": "mock_contract_fixture",
    "outfits": [
        _s4_outfit("O0001", 0.32, (50.0, 60.0, 90.0)),
        _s4_outfit("O0002", 0.31, (40.0, 50.0, 80.0)),
        _s4_outfit("O0003", 0.30, (30.0, 40.0, 70.0)),
        _s4_outfit("O0004", 0.29, (20.0, 30.0, 60.0)),
    ],
}


def test_products_outfits_with_their_own_product_score_are_recommended():
    result = _recommend(student4=S4_PRODUCTS)
    assert result.status == "ok", result.reason
    ids = [r["outfit_id"] for r in result.data["recommendation_set"]["recommendations"]]
    assert ids == ["O0001", "O0002", "O0003"]
    top = result.data["recommendation_set"]["recommendations"][0]["purchase_items"][0]
    assert top["price"] == "50.0" and top["currency"] == "SGD"
    assert top["product_score"] == 0.5
    assert "embedding" not in top and "provenance" not in top and "evidence" not in top


def test_products_without_product_score_are_an_explicit_error():
    student4 = copy.deepcopy(S4_PRODUCTS)
    student4["outfits"][1]["products"]["TOP"] = _s4_product("O0002_top", "TOP", 40.0, score=None)
    result = _recommend(student4=student4)
    assert result.status == "error"
    error = result.data["error"]
    assert error["code"] == "missing_product_score"
    assert error["location"] == "outfits[1].products.TOP.product_score"


def test_unknown_availability_is_rejected_and_never_recommended():
    student4 = copy.deepcopy(S4_PRODUCTS)
    student4["outfits"][0]["products"]["TOP"]["availability"] = "UNKNOWN"
    result = _recommend(student4=student4)
    assert result.status == "error"
    assert result.data["error"]["code"] == "availability_unknown"
    assert result.data["error"]["location"] == "outfits[0].products.TOP.availability"
    assert "recommendation_set" not in result.data


def test_same_non_sgd_currency_does_not_load_exchange_rates(monkeypatch):
    """A USD request with only USD products is already directly comparable."""
    from travel_agent.recommendation import service

    monkeypatch.setattr(
        service,
        "load_ecb_snapshot",
        lambda: pytest.fail("same-currency recommendation must not fetch exchange rates"),
    )
    student4 = copy.deepcopy(STUDENT4)
    for outfit in student4["outfits"]:
        for item in outfit["items"]:
            item["currency"] = "USD"

    result = run(
        {
            "action": "recommend",
            "request": {**REQUEST, "budget": "300", "currency": "USD"},
            "student4": student4,
        }
    )

    assert result.status == "ok", result.reason
    assert result.data["state"]["current_request"]["currency"] == "USD"
    assert "currency_conversion" not in result.data
    recommendation = result.data["recommendation_set"]["recommendations"][0]
    assert recommendation["currency"] == "USD"
    assert recommendation["total_purchase_cost"] == "200.00"


def test_display_currency_follows_cny_request_even_when_products_are_usd(monkeypatch):
    from travel_agent.recommendation import service

    rates = ExchangeRateSnapshot(
        rate_date=date(2026, 10, 21),
        rates_per_eur={
            "EUR": Decimal("1"),
            "USD": Decimal("1.2"),
            "SGD": Decimal("1.5"),
            "CNY": Decimal("8"),
        },
        source="ECB",
    )
    monkeypatch.setattr(service, "load_ecb_snapshot", lambda: rates)
    student4 = copy.deepcopy(STUDENT4)
    for outfit in student4["outfits"]:
        for item in outfit["items"]:
            item["currency"] = "USD"

    result = run(
        {
            "action": "recommend",
            "request": {**REQUEST, "budget": "2000", "currency": "CNY"},
            "student4": student4,
        }
    )

    assert result.status == "ok", result.reason
    recommendation = result.data["recommendation_set"]["recommendations"][0]
    assert recommendation["currency"] == "CNY"
    assert recommendation["total_purchase_cost"] == "1333.33"
    assert [(p["price"], p["currency"]) for p in recommendation["purchase_items"]] == [
        ("333.33", "CNY"),
        ("400.00", "CNY"),
        ("600.00", "CNY"),
    ]


def test_all_sgd_input_does_not_load_exchange_rates(monkeypatch):
    from travel_agent.recommendation import service

    monkeypatch.setattr(
        service,
        "load_ecb_snapshot",
        lambda: pytest.fail("all-SGD recommendation must not fetch exchange rates"),
        raising=False,
    )
    assert _recommend().status == "ok"


def test_unconvertible_currency_is_an_error(monkeypatch):
    from travel_agent.recommendation import service

    rates = ExchangeRateSnapshot(
        rate_date=date(2026, 10, 21),
        rates_per_eur={"EUR": Decimal("1"), "SGD": Decimal("1.5")},
        source="ECB",
    )
    monkeypatch.setattr(service, "load_ecb_snapshot", lambda: rates)
    result = _recommend(request={**REQUEST, "currency": "XYZ"})
    assert result.status == "error"
    assert result.data["error"]["type"] == "CurrencyConversionError"
