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
from pathlib import Path

import pytest

from travel_agent.recommendation.service import ACTIONS, DEFAULT_BETA, run
from travel_agent.recommendation.student5.models import TripShoppingRequest
from travel_agent.recommendation.student5.orchestrator import Student5Orchestrator
from travel_agent.recommendation.student5.student4_adapter import adapt_student4_output
from travel_agent.recommendation.tools import validate_and_recommend

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
    [{}, {"request": REQUEST, "student4": STUDENT4}, {"intent": "MODIFY_REQUIREMENTS"}],
)
def test_missing_action_is_an_error_not_an_implicit_recommend(payload):
    result = run(payload)
    assert result.status == "error"
    assert result.data["error"]["type"] == "InvalidAction" and "action" in result.reason


@pytest.mark.parametrize("action", ["RECOMMEND", "CONFIRM_OUTFIT", "buy", None, ["recommend"]])
def test_unknown_action_is_an_error(action):
    result = run({"action": action})
    assert result.status == "error"
    assert result.data["error"]["actions"] == list(ACTIONS)


@pytest.mark.parametrize("payload", [None, [], "recommend"])
def test_non_object_payload_is_an_error(payload):
    assert run(payload).status == "error"


@pytest.mark.parametrize("missing", ["request", "student4"])
def test_recommend_without_request_or_student4_needs_input(missing):
    payload = {"action": "recommend", "request": REQUEST, "student4": STUDENT4}
    del payload[missing]
    result = run(payload)
    assert result.status == "needs_input" and missing in result.data["error"]["message"]


def test_unknown_keys_need_input():
    result = _recommend(api_key="x")
    assert result.status == "needs_input" and "api_key" in result.data["error"]["message"]


def test_request_keeps_student5_contract_and_rejects_unconsumed_fields():
    result = _recommend(request={**REQUEST, "dress_codes": {"dinner": "smart casual"}})
    assert result.status == "needs_input"
    assert result.data["error"]["type"] == "ValidationError"


@pytest.mark.parametrize("beta", [-0.1, 1.5, "high"])
def test_invalid_beta_needs_input(beta):
    assert _recommend(beta=beta).status == "needs_input"


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


def test_recommend_matches_the_student5_orchestrator_exactly(first_round):
    """The service is a wrapper: same Top-3 and scores as the orchestrator itself."""
    outfits = adapt_student4_output(STUDENT4).outfits
    direct = Student5Orchestrator(beta=DEFAULT_BETA, session_id="session").run(
        request=TripShoppingRequest.model_validate(REQUEST), candidate_outfits=outfits
    )
    assert first_round.data["recommendation_set"] == direct.recommendation_set.model_dump(
        mode="json"
    )


def test_recommend_uses_the_given_beta():
    by_relevance = _recommend(beta=0.0).data["recommendation_set"]["recommendations"]
    assert by_relevance[0]["final_outfit_score"] == by_relevance[0]["outfit_product_relevance"]


def test_no_feasible_outfit_returns_structured_conflict():
    result = _recommend(request={**REQUEST, "budget": "50"})
    assert result.status == "no_candidates"
    conflict = result.data["conflict"]
    assert conflict["violated_constraints"] == ["BUDGET", "AVAILABILITY"]
    assert conflict["budget"] == "50" and conflict["currency"] == "SGD"
    assert result.data["recommendation_set"] is None
    assert result.reason == conflict["message"]


@pytest.mark.parametrize(
    "currency, price_minor, expected",
    [("SGD", 3990, "39.90"), ("JPY", 5000, "5000"), ("KRW", 45000, "45000")],
)
def test_price_minor_is_converted_with_the_currency_minor_unit(currency, price_minor, expected):
    outfit = _outfit("o_x", 0.8, (price_minor, price_minor, price_minor), currency=currency)
    for item in outfit["items"]:
        item["currency"] = currency
    result = _recommend(
        request={**REQUEST, "currency": currency, "budget": "1000000"},
        student4={**STUDENT4, "outfits": [outfit]},
    )
    assert result.status == "ok", result.reason
    prices = {
        p["price"]
        for p in result.data["recommendation_set"]["recommendations"][0]["purchase_items"]
    }
    assert prices == {expected}


def test_missing_compatibility_score_is_an_explicit_error():
    student4 = copy.deepcopy(STUDENT4)
    del student4["outfits"][1]["compatibility_score"]
    result = _recommend(student4=student4)
    assert result.status == "error"
    error = result.data["error"]
    assert error["code"] == "invalid_compatibility_score"
    assert error["location"] == "outfits[1].compatibility_score"


@pytest.mark.parametrize("availability", ["unknown", None, "true", 1])
def test_availability_must_be_a_real_boolean_never_coerced_to_true(availability):
    student4 = copy.deepcopy(STUDENT4)
    student4["outfits"][0]["items"][0]["availability"] = availability
    result = _recommend(student4=student4)
    assert result.status == "error"
    assert result.data["error"]["code"] == "invalid_item"
    assert result.data["error"]["location"] == "outfits[0].items[0]"


def test_purchasable_item_without_product_url_is_rejected():
    student4 = copy.deepcopy(STUDENT4)
    del student4["outfits"][0]["items"][1]["product_url"]
    result = _recommend(student4=student4)
    assert result.status == "error"
    assert result.data["error"]["code"] == "invalid_item"
    assert "product_url" in result.data["error"]["message"]


@pytest.mark.parametrize(
    "change, code",
    [
        (lambda s: s.update(status="not_implemented"), "status_not_ok"),
        (lambda s: s.update(outfits=[]), "no_outfits"),
        (lambda s: s["outfits"][0]["items"][0].update(price_minor=19.9), "invalid_price"),
        (lambda s: s["outfits"][1].update(outfit_id="o_a"), "duplicate_outfit_id"),
    ],
)
def test_student4_contract_violations_are_errors_with_code(change, code):
    student4 = copy.deepcopy(STUDENT4)
    change(student4)
    result = _recommend(student4=student4)
    assert result.status == "error" and result.data["error"]["code"] == code


def test_currency_mismatch_is_an_error():
    result = _recommend(request={**REQUEST, "currency": "USD"})
    assert result.status == "error"
    assert result.data["error"]["type"] == "CurrencyMismatchError"


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


def test_patch_keeps_its_own_stage1_intent(first_round):
    patch = {"intent": "CONFIRM_OUTFIT", "set": {"budget": "180"}}
    result = run(
        {"action": "modify_requirements", "state": first_round.data["state"], "patch": patch}
    )
    assert result.status == "needs_input"


def test_same_value_patch_has_no_rerun_plan(first_round):
    change = _modify(first_round, "300")
    assert change.status == "ok" and change.data["changed"] is False
    assert change.data["rerun_plan"] is None


def test_invalid_patch_needs_input(first_round):
    assert _modify(first_round, "-5").status == "needs_input"


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


@pytest.mark.parametrize("outfit_id", ["o_z", "the second one", "2", "o_e"])
def test_confirm_accepts_only_an_exact_outfit_id_from_the_latest_set(first_round, outfit_id):
    result = run(_selection(first_round, outfit_id))
    assert result.status == "needs_input"
    assert result.data["error"]["type"] == "InvalidOutfitSelectionError"


@pytest.mark.parametrize("outfit_id", [2, None, ["o_a"]])
def test_confirm_requires_an_outfit_id_string(first_round, outfit_id):
    assert run(_selection(first_round, outfit_id)).status == "needs_input"


def test_explanation_is_data_only_and_does_not_change_the_state(first_round):
    state_before = copy.deepcopy(first_round.data["state"])
    result = run(_selection(first_round, "o_b", action="explain"))
    assert result.status == "ok"
    assert set(result.data) == {"explanation"}
    explanation = result.data["explanation"]
    assert explanation["outfit_id"] == "o_b" and explanation["display_position"] == 2
    assert first_round.data["state"] == state_before
    # The same, unchanged state can still confirm afterwards.
    assert run(_selection(first_round, "o_b")).status == "ok"


# ---------------------------------------------------------------------------
# Stage 14: preview
# ---------------------------------------------------------------------------


@pytest.fixture
def confirmed_outfit(first_round):
    return run(_selection(first_round, "o_a")).data["confirmed_outfit"]


def test_preview_with_mock_provider_is_marked_mock(confirmed_outfit):
    result = run(
        {
            "action": "preview",
            "confirmed_outfit": confirmed_outfit,
            "preview_mode": "MANNEQUIN",
            "preview_provider": "mock",
        }
    )
    assert result.status == "ok"
    assert result.data["preview_source"] == "mock"
    preview = result.data["preview"]
    assert (preview["status"], preview["provider_name"]) == ("SUCCESS", "mock")
    assert preview["output_reference"] == "mock://preview/o_a/mannequin"


def test_preview_without_provider_is_marked_product_card_fallback(confirmed_outfit):
    result = run(
        {"action": "preview", "confirmed_outfit": confirmed_outfit, "preview_mode": "MANNEQUIN"}
    )
    assert result.data["preview_source"] == "product_card_fallback"
    preview = result.data["preview"]
    assert (preview["status"], preview["delivered_mode"]) == ("FALLBACK", "PRODUCT_CARDS")
    assert preview["output_reference"] is None


@pytest.mark.parametrize(
    "extra",
    [{"preview_provider": "openai"}, {"preview_mode": "HOLOGRAM"}, {"user_portrait": "me.jpg"}],
)
def test_preview_rejects_real_provider_bad_mode_and_unconsented_portrait(confirmed_outfit, extra):
    payload = {
        "action": "preview",
        "confirmed_outfit": confirmed_outfit,
        "preview_mode": "FLAT_LAY",
    }
    payload.update(extra)
    assert run(payload).status == "needs_input"


# ---------------------------------------------------------------------------
# Tool wrapper and import safety
# ---------------------------------------------------------------------------


def test_tool_wrapper_returns_json_module_result():
    result = validate_and_recommend.invoke(
        {"payload": {"action": "recommend", "request": REQUEST, "student4": STUDENT4}}
    )
    assert result["status"] == "ok"
    assert result["data"]["recommendation_set"]["recommendations"][0]["outfit_id"] == "o_a"
    assert validate_and_recommend.invoke({"payload": {}})["status"] == "error"


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
