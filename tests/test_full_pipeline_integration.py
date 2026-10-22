"""Module 11: five-module full-pipeline integration test (Students 1-5).

Flow: fixed request -> Student 1 -> Student 2 -> Student 3 -> Student 4 -> Student 5
(`travel_agent.recommendation.service.run`, whose student4_adapter reads the Student 4 output)
-> Top-3 or structured no-feasible -> confirm by outfit_id -> MockPreviewProvider.

Only real public entry points are called (see tests/full_pipeline_harness.py). Nothing here
writes a module's output by hand. When a module is a scaffold, lacks a runtime prerequisite or
has no agreed handoff contract, the full-pipeline tests are SKIPPED with the exact stage and
reason; a skip is not an end-to-end pass. Tests that only check part of the chain are named
after that part (contract / boundary tests), never "full pipeline".

Default runs need no model, network, credentials, `.env` or money. Model/API-backed stages run
only with RUN_FULL_PIPELINE_TESTS=1 (marker `full_pipeline_live`). Student 5 is the in-repository
`travel_agent.recommendation` module. See docs/full_pipeline_integration.md.
"""

import copy
from pathlib import Path

import pytest

from tests.full_pipeline_harness import (
    ENTRY_POINTS,
    FIXED_MESSAGE,
    HARNESS_STATUSES,
    MODULES,
    RUN_ENV,
    PipelineRun,
    StageRecord,
    _summary,
    build_record,
    check_shared_request_fields,
    check_student1_output,
    check_student5_result,
    confirm_and_preview,
    find_unsafe_values,
    missing_inspiration_extras,
    missing_product_extras,
    models_allowed,
    rerun_from,
    reviewed_payload,
    run_pipeline,
    run_student4_stage,
    run_student5_stage,
    student2_payload,
    student3_payload,
    student5_payload,
    student5_request,
)
from travel_agent.recommendation.service import run as student5_run
from travel_agent.schemas import ModuleResult
from travel_agent.understanding.service import run as student1_run

ROOT = Path(__file__).resolve().parents[1]
live = pytest.mark.full_pipeline_live


def blocking_skip_reason(run):
    """Skip reason when the run stopped for a structural reason, else None."""
    stop = run.stopped
    if stop is not None and stop.status in ("not_implemented", *HARNESS_STATUSES):
        return f"Full pipeline not runnable: {run.stop_reason()}"
    return None


def _completed(payload, **kwargs):
    """Run the real chain; skip with the exact blocking stage unless every module ran."""
    run = run_pipeline(payload, **kwargs)
    if not run.completed:
        reason = blocking_skip_reason(run)
        if reason:
            pytest.skip(reason)
        pytest.fail(f"Real module stopped the pipeline: {run.stop_reason()}")
    return run


def _requires_inspiration_extra():
    missing = missing_inspiration_extras()
    if missing:
        pytest.skip(f"Student 2 inspiration extra not installed ({missing})")


def _requires_products_extra():
    missing = missing_product_extras()
    if missing:
        pytest.skip(f"Student 3 products extra not installed ({missing})")


@pytest.fixture
def s1():
    result = student1_run(reviewed_payload())
    assert result.status == "ok", result.reason
    return result


def _handoff_result(handoff) -> ModuleResult:
    """Student 2-shaped result for contract tests of the harness' handoff validation only."""
    return ModuleResult(status="ok", data={"product_handoff": handoff})


DOCUMENTED_HANDOFF = {  # shape from docs/inspiration-runbook.md section 9.3
    "upstream": {
        "pt": "inspiration-handoffs/contract/phase_2_output.pt",
        "text": "inspiration-handoffs/contract/expanded_fashion_context.txt",
        "context": "inspiration-handoffs/contract/context.json",
        "model_info": "inspiration-handoffs/contract/model_info.json",
    },
    "reference_mode": "imported",
}


@pytest.fixture(scope="module")
def normal_run():
    return _completed(reviewed_payload())


# ---------------------------------------------------------------------------
# Offline checks that run by default (partial; not end-to-end)
# ---------------------------------------------------------------------------


def test_student5_is_called_through_the_in_repository_recommendation_entry_point():
    assert ENTRY_POINTS["student5"] == "travel_agent.recommendation.service.run"
    result = student5_run({})
    # Implemented, not a scaffold: an empty payload has no `action`, which its contract rejects.
    assert result.status == "error" and result.data["error"]["type"] == "InvalidAction"
    harness = (ROOT / "tests" / "full_pipeline_harness.py").read_text(encoding="utf-8")
    assert "import student5" not in harness and "from student5" not in harness


def test_student1_handoff_for_fixed_reviewed_request():
    result = student1_run(reviewed_payload())
    assert result.status == "ok", result.reason
    assert check_student1_output(result) == []
    trip = result.data["trip_request"]
    assert (trip["destination"], trip["currency"], trip["activities"]) == (
        "Seoul",
        "SGD",
        ["sightseeing", "dinner"],
    )
    assert result.data["feasibility_status"] == "unverified"


@pytest.mark.parametrize("missing", ["budget"])
def test_flow_stops_and_keeps_student1_needs_input(missing):
    run = run_pipeline(reviewed_payload(**{missing: None}), allow_models=False)
    assert [s.module for s in run.stages] == ["student1"]
    assert run.stopped.status == "needs_input"
    assert missing in run.stopped.summary["questions"]
    assert not run.completed


def test_flow_stops_at_first_unavailable_module_without_calling_later_ones():
    run = run_pipeline(reviewed_payload(), allow_models=False)
    if run.completed:
        pytest.skip("Pipeline completed; no blocking module left")
    modules = [s.module for s in run.stages]
    assert modules == list(MODULES[: len(modules)])
    assert all(s.status == "ok" for s in run.stages[:-1])
    assert run.stopped.status in (
        "needs_input",
        "no_candidates",
        "error",
        "not_implemented",
        *HARNESS_STATUSES,
    )
    assert run.stopped.reason


# ---------------------------------------------------------------------------
# Student 1 -> Student 2 contract (offline)
# ---------------------------------------------------------------------------


def test_student2_payload_passes_student1_data_not_the_module_result(s1):
    payload = student2_payload(s1)
    assert set(payload) == {"understanding"}
    assert payload["understanding"] is s1.data
    assert "status" not in payload["understanding"] and "reason" not in payload["understanding"]
    assert student2_payload(s1, top_n=3)["top_n"] == 3


# ---------------------------------------------------------------------------
# Student 2 -> Student 3 contract (offline)
# ---------------------------------------------------------------------------


def test_student3_payload_is_the_product_handoff_unchanged():
    payload = student3_payload(_handoff_result(copy.deepcopy(DOCUMENTED_HANDOFF)))
    assert payload == DOCUMENTED_HANDOFF


def _boundary_phase2_output():
    """Unit vectors in Student 2's phase2_output layout (as in Student 2's own service test)."""
    import torch

    vectors = torch.eye(512)
    return {
        "context_embedding": vectors[0:1].clone(),
        "top_images": [
            {"image_id": "boundary_001", "image_url": "https://example.com/1.jpg", "rank": 1},
            {"image_id": "boundary_002", "image_url": "https://example.com/2.jpg", "rank": 2},
        ],
        "top_image_embeddings": vectors[1:3].clone(),
    }


def test_student2_product_handoff_is_accepted_by_student3_input_adapter(s1, tmp_path, monkeypatch):
    """Boundary/contract test, not end-to-end.

    The real Student 2 service runs with its retrieval pipeline isolated (fixed unit-vector
    output, no model, no Unsplash). Its real `save_inspiration_handoff` writes the files. The
    harness hands them to Student 3's real schema and upstream adapter (`prepare_input`).
    """
    _requires_inspiration_extra()
    _requires_products_extra()
    import travel_agent.inspiration.service as student2
    from travel_agent.products.retrieval.integration import (
        ProductToolInput,
        Settings,
        prepare_input,
    )

    monkeypatch.setenv("STUDENT3_DATA_ROOT", str(tmp_path))
    output = _boundary_phase2_output()
    monkeypatch.setattr(student2, "run_inspiration_pipeline", lambda **kwargs: (output, {}))
    s2 = student2.run(student2_payload(s1))
    assert s2.status == "ok", s2.reason

    payload = student3_payload(s2)
    assert payload is s2.data["product_handoff"]
    spec = ProductToolInput.model_validate(payload)
    request, metadata = prepare_input(spec, Settings(tmp_path, tmp_path / "work", tmp_path))
    assert metadata["status"] == "declared_compatible"
    assert [r.reference_id for r in request.top_social_references] == [
        "boundary_001",
        "boundary_002",
    ]
    assert request.context.trip.destination == s1.data["trip_request"]["destination"]

    summary = _summary("student2", s2)
    assert summary["product_handoff"] == {
        "reference_mode": "imported",
        "upstream_keys": ["context", "model_info", "pt", "text"],
    }
    run = PipelineRun(
        stages=[StageRecord("student2", ENTRY_POINTS["student2"], "ok", "boundary", None, summary)]
    )
    record = build_record(run, input_description="boundary")
    assert find_unsafe_values(record) == []
    assert not {"context_embedding", "top_image_embeddings", "top_images"} & set(summary)
    assert summary["has_context_embedding"] is True and summary["top_image_count"] == 2


# ---------------------------------------------------------------------------
# Student 4 boundary (offline)
# ---------------------------------------------------------------------------


def test_dynamic_student4_service_hands_student3_products_to_student5_contract(s1, monkeypatch):
    """The harness calls the real dynamic Student 4 boundary, not a fixed sample scaffold."""
    from datetime import date
    from decimal import Decimal

    from test_outfits_dynamic import MANIFEST, Model
    from test_outfits_dynamic import payload as student4_fixture

    from travel_agent.recommendation import service as recommendation_service
    from travel_agent.recommendation.currency_converter import ExchangeRateSnapshot

    monkeypatch.setattr(
        recommendation_service,
        "load_ecb_snapshot",
        lambda: ExchangeRateSnapshot(
            rate_date=date(2026, 10, 21),
            rates_per_eur={"EUR": Decimal("1"), "SGD": Decimal("1.5"), "USD": Decimal("1.2")},
            source="test-fixture",
        ),
    )

    fixture = student4_fixture()
    s3 = ModuleResult(status="ok", data=fixture["products"])
    run = PipelineRun()

    assert run_student4_stage(run, s1, s3, model=Model(), manifest=MANIFEST) is True
    result = run.outputs["student4"]
    assert result.status == "ok"
    assert result.data["source"] == "student4"
    assert result.data["outfits"]
    assert run_student5_stage(run, s1, beta=0.5, session_id="dynamic-offline") is True
    s5 = run.outputs["student5"]
    assert s5.status == "ok", s5.reason
    assert s5.data["student4_source"] == "student4"
    assert s5.data["recommendation_set"]["recommendations"]


def test_dynamic_student4_validation_error_stops_the_pipeline_with_its_reason(s1):
    run = PipelineRun()
    invalid_student3 = ModuleResult(status="ok", data={})

    assert run_student4_stage(run, s1, invalid_student3) is False
    assert (run.stopped.module, run.stopped.status) == ("student4", "needs_input")
    assert "bundle" in (run.stopped.reason or "")


def test_shared_request_field_check_detects_budget_unit_mismatch():
    result = student1_run(reviewed_payload())
    # Fault injection on a copy of real Student 1 output: shift budget_minor by one minor unit.
    broken = result.model_copy(deep=True)
    broken.data["hard_constraints"]["budget_minor"] += 1
    assert any(p.startswith("budget:") for p in check_shared_request_fields(broken))


def test_student5_stage_stops_with_named_missing_request_fields(s1):
    """Fault injection on a copy of real Student 1 output: no value is invented for Student 5."""
    broken = s1.model_copy(deep=True)
    broken.data["trip_request"]["destination"] = None
    broken.data["trip_request"]["currency"] = None
    run = PipelineRun()
    assert run_student5_stage(run, broken, beta=0.5, session_id="s") is False
    stop = run.stopped
    assert (stop.module, stop.status) == ("student5", "needs_input")
    assert stop.summary["missing_request_fields"] == ["destination", "currency"]
    assert blocking_skip_reason(run) is None  # a real stop: tests fail, they do not skip


# ---------------------------------------------------------------------------
# Mocked offline integration (NOT live end-to-end).
#
# Real Student 1 entry point and real Student 5 entry point; Students 2-4 are replaced by a
# Student 4 contract fixture in the real product-object shape (category-keyed `products`, each
# with `product_score` and verified stock). This checks the Student 1 -> Student 5 chain and
# the Student 4 -> Student 5 hand-off while the live chain is blocked (see the skipped
# full-pipeline tests below). The compatibility and product scores are fixture values.
# ---------------------------------------------------------------------------

MOCK_STUDENT4_SOURCE = "mocked-offline-student4-fixture"


def _mock_product(pid, category, amount):
    return {
        "product_id": pid,
        "title": pid,
        "category": category,
        "source_type": "ECOMMERCE_API",
        "price": {"amount": amount, "currency": "SGD"},
        "sizes": ["M"],
        "availability": "IN_STOCK",
        "evidence": {"availability_verified": True, "sizes_verified": False},
        "product_url": f"https://shop.example.com/{pid}",
        "provenance": {"source": "fixture", "original": {"api_item": {}}},
        "owned": False,
        "embedding": {"values": [0.1, 0.2]},
        "product_score": 0.4,
    }


def _mock_student4(*amounts_per_outfit):
    """Student 4 ModuleResult with one outfit per (top, bottom, shoes) price triple."""
    outfits = [
        {
            "outfit_id": f"O{n:04d}",
            "compatibility_score": round(0.4 - n * 0.01, 4),
            "products": {
                "TOP": _mock_product(f"O{n:04d}_top", "TOP", top),
                "BOTTOM": _mock_product(f"O{n:04d}_bottom", "BOTTOM", bottom),
                "FOOTWEAR": _mock_product(f"O{n:04d}_shoes", "FOOTWEAR", shoes),
            },
        }
        for n, (top, bottom, shoes) in enumerate(amounts_per_outfit, start=1)
    ]
    return ModuleResult(
        status="ok",
        data={"outfits": outfits, "model_version": "fixture", "source": MOCK_STUDENT4_SOURCE},
    )


AFFORDABLE = [(60.0, 70.0, 50.0), (50.0, 60.0, 40.0), (40.0, 50.0, 30.0), (30.0, 40.0, 20.0)]


def _mocked_run(s1, student4):
    """Student 5 on Student 1's real output and the mocked Student 4 result, using the
    harness payload builders. `run_student5_stage` is not used: its live hand-off check
    rightly refuses a mock Student 4 source."""
    assert check_shared_request_fields(s1) == []
    request = student5_request(s1)
    payload = student5_payload(request, student4, beta=0.5, session_id="offline")
    run = PipelineRun()
    s5 = student5_run(payload)
    run.outputs.update(
        student4=student4, student5=s5, student4_raw=payload["student4"], request5=request, beta=0.5
    )
    return run, s5.status in ("ok", "no_candidates")


class _FixtureExtractor:
    """Stands in for Student 1's Qwen extractor (offline): proposes the reviewed request."""

    def extract(self, message, previous):
        assert message == FIXED_MESSAGE
        return copy.deepcopy(reviewed_payload()["request"]), {"provider": "fixture-extractor"}


def test_mocked_offline_natural_language_request_goes_through_student1_review_to_student5():
    """Natural language enters Student 1's real entry point; the proposal needs review and is
    only used after explicit acceptance, then it reaches Student 5."""
    base = reviewed_payload()
    message = {"message": FIXED_MESSAGE, "weather_fixture": base["weather_fixture"]}
    proposal = student1_run({**message, "allow_draft": True}, extractor=_FixtureExtractor())
    assert proposal.status == "needs_input"
    assert proposal.data["questions"][0]["field"] == "extraction_review"
    accepted = student1_run(
        {**message, "allow_draft": True, "accept_extraction": True}, extractor=_FixtureExtractor()
    )
    assert accepted.status == "ok", accepted.reason
    assert check_student1_output(accepted) == []
    run, ok = _mocked_run(accepted, _mock_student4(*AFFORDABLE))
    assert ok and run.outputs["student5"].status == "ok"
    assert check_student5_result(run) == []


def test_mocked_offline_student4_scores_reach_a_student5_top3(s1):
    run, ok = _mocked_run(s1, _mock_student4(*AFFORDABLE))
    assert ok
    data = run.outputs["student5"].data
    recommendations = data["recommendation_set"]["recommendations"]
    assert [r["outfit_id"] for r in recommendations] == ["O0001", "O0002", "O0003"]
    assert data["student4_source"] == MOCK_STUDENT4_SOURCE
    raw = {o["outfit_id"]: o["compatibility_score"] for o in run.outputs["student4_raw"]["outfits"]}
    assert all(r["compatibility_score"] == raw[r["outfit_id"]] for r in recommendations)
    assert check_student5_result(run) == []


def test_mocked_offline_confirm_then_mock_preview(s1):
    run, _ = _mocked_run(s1, _mock_student4(*AFFORDABLE))
    outcome = confirm_and_preview(run)
    assert outcome["confirmed_outfit_id"] == "O0001"
    assert outcome["session_status"] == "CONFIRMED"
    assert outcome["preview"]["provider"] == "mock"


def test_mocked_offline_requirement_change_invalidates_the_old_outfit_ids(s1):
    run, _ = _mocked_run(s1, _mock_student4(*AFFORDABLE))
    first = run.outputs["student5"].data
    change = student5_run(
        {
            "action": "modify_requirements",
            "state": first["state"],
            "patch": {"intent": "MODIFY_REQUIREMENTS", "set": {"budget": 150}},
        }
    )
    assert change.status == "ok"
    # Budget affects Stages 10-12; Stage 11 (compatibility scoring) is Student 4's.
    assert change.data["rerun_plan"]["affected_stages"] == [10, 11, 12]
    assert rerun_from(change.data["rerun_plan"]) == "student4"
    stale = student5_run(
        {
            "action": "confirm",
            "state": change.data["state"],
            "recommendation_set": first["recommendation_set"],
            "outfit_id": first["recommendation_set"]["recommendations"][0]["outfit_id"],
        }
    )
    assert stale.status == "needs_input"
    second = student5_run(
        {
            "action": "recommend",
            "request": change.data["updated_request"],
            "student4": run.outputs["student4_raw"],
            "state": change.data["state"],
        }
    )
    assert second.status == "ok"
    assert second.data["state"]["recommendation_round"] == 2
    assert [r["outfit_id"] for r in second.data["recommendation_set"]["recommendations"]] == [
        "O0002",
        "O0003",
        "O0004",
    ]


# ---------------------------------------------------------------------------
# Full pipeline (all five real modules). Skipped with the blocking stage until runnable.
# Student 5 is always reached through travel_agent.recommendation.service.run.
# ---------------------------------------------------------------------------


def _s5(run):
    return run.outputs["student5"].data


def _selection(data, outfit_id, recommendation_set=None):
    return {
        "action": "confirm",
        "state": data["state"],
        "recommendation_set": recommendation_set or data["recommendation_set"],
        "outfit_id": outfit_id,
    }


@live
def test_full_pipeline_normal_request_returns_one_to_three_outfits(normal_run):
    assert normal_run.outputs["student5"].status == "ok"
    rs = _s5(normal_run)["recommendation_set"]
    assert rs is not None, "fixed normal request must yield recommendations"
    assert 1 <= len(rs["recommendations"]) <= 3
    assert check_student5_result(normal_run) == []
    ids = [r["outfit_id"] for r in rs["recommendations"]]
    assert len(set(ids)) == len(ids)
    assert _s5(normal_run)["state"]["last_recommended_outfit_ids"] == ids


@live
def test_full_pipeline_requirement_change_reruns_stages_and_rejects_old_outfit_id(normal_run):
    s1 = normal_run.outputs["student1"]
    old = _s5(normal_run)
    patch = {"intent": "MODIFY_REQUIREMENTS", "set": {"budget": 400}, "add": {}, "remove": {}}
    change = student5_run({"action": "modify_requirements", "state": old["state"], "patch": patch})
    assert change.status == "ok" and change.data["rerun_plan"] is not None
    payload = reviewed_payload()
    del payload["request"]
    payload.update(patch=patch, state=s1.data["state"])
    rerun = _completed(
        payload,
        previous=normal_run,
        resume_from=rerun_from(change.data["rerun_plan"]),
        state=change.data["state"],
    )
    new = _s5(rerun)
    assert change.data["updated_request"] == new["state"]["current_request"]
    assert [s.module for s in rerun.stages if s.status != "reused"][-1] == "student5"
    old_id = old["recommendation_set"]["recommendations"][0]["outfit_id"]
    stale = student5_run(_selection(new, old_id, recommendation_set=old["recommendation_set"]))
    assert stale.status == "needs_input"
    assert stale.data["error"]["type"] == "StaleRecommendationSetError"
    assert check_student5_result(rerun) == []


@live
def test_full_pipeline_natural_language_request_goes_through_student1_review(monkeypatch):
    """Student 1 Qwen extraction of FIXED_MESSAGE; continues only with the reviewed request."""
    if not models_allowed():
        pytest.skip(f"Qwen3-0.6B extraction needs local weights; set {RUN_ENV}=1")
    pytest.importorskip("transformers", reason="Student 1 qwen extra not installed")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    base = reviewed_payload()
    proposal = student1_run(
        {"message": FIXED_MESSAGE, "weather_fixture": base["weather_fixture"], "allow_draft": True}
    )
    if proposal.status == "error":
        pytest.skip(f"Student 1 Qwen extractor unavailable offline: {proposal.reason}")
    assert proposal.status == "needs_input"
    assert proposal.data["questions"][0]["field"] == "extraction_review"
    assert proposal.data["proposed_request"]["destination"]
    # Human review step: the reviewed structured request is submitted, never auto-accepted.
    run = _completed(base, student1_source="FIXED_MESSAGE -> Qwen proposal -> reviewed request")
    assert check_student5_result(run) == []
