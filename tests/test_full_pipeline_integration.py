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
import importlib
import json
from pathlib import Path

import pytest

from tests.full_pipeline_harness import (
    ENTRY_POINTS,
    FIXED_MESSAGE,
    HARNESS_STATUSES,
    MODULES,
    PENDING_STATUSES,
    RUN_ENV,
    SHARED_REQUEST_FIELDS,
    STUDENT1_EXTENSION_FIELDS,
    STUDENT2_REQUIRED_CONTEXT,
    UNSPLASH_ENV,
    HandoffContractError,
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
    student5_request,
)
from travel_agent.recommendation.service import run as student5_run
from travel_agent.schemas import ModuleResult
from travel_agent.schemas import TripShoppingRequest as SharedTripShoppingRequest
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


@pytest.mark.parametrize("module", MODULES)
def test_entry_points_of_students_1_to_5_are_importable(module):
    if module == "student2":
        # Student 2's service imports torch/transformers at import time.
        _requires_inspiration_extra()
    path, name = ENTRY_POINTS[module].rsplit(".", 1)
    assert callable(getattr(importlib.import_module(path), name)), module


def test_student2_entry_point_is_implemented_not_a_scaffold():
    _requires_inspiration_extra()
    from travel_agent.inspiration.service import run

    # Empty payload: the real service validates input instead of returning not_implemented.
    result = run({})
    assert result.status == "needs_input"


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


@pytest.mark.parametrize("missing", ["budget", "activities"])
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


def test_blocked_report_names_real_gaps_and_contains_no_unsafe_values():
    run = run_pipeline(reviewed_payload(), allow_models=False)
    if run.completed:
        pytest.skip("Pipeline completed; a blocked report is not produced")
    record = build_record(run, input_description="reviewed request")
    assert record["status"] == PENDING_STATUSES.get(run.stopped.status, "stopped_by_module_status")
    assert record["status"] != "completed"
    assert (record["stopped_module"], record["stopped_status"]) == (
        run.stopped.module,
        run.stopped.status,
    )
    assert record["final_result"] is None
    assert run.stopped.module in record["stopped_at"]
    assert record["gaps"] and record["not_reached"]
    assert not any("returns not_implemented" in g and "Student 2" in g for g in record["gaps"])
    assert find_unsafe_values(record) == []


def test_default_run_stops_at_student2_prerequisites_not_as_not_implemented(monkeypatch):
    monkeypatch.delenv(RUN_ENV, raising=False)
    monkeypatch.delenv(UNSPLASH_ENV, raising=False)
    run = run_pipeline(reviewed_payload())
    assert [s.module for s in run.stages] == ["student1", "student2"]
    stop = run.stopped
    assert stop.status == "prerequisite_missing" and stop.status != "not_implemented"
    assert stop.reason.startswith("Student 2 is implemented")
    assert RUN_ENV in stop.reason and UNSPLASH_ENV in stop.reason
    record = build_record(run, input_description="reviewed request")
    assert record["status"] == "blocked_by_missing_prerequisites"


# ---------------------------------------------------------------------------
# Student 1 -> Student 2 contract (offline)
# ---------------------------------------------------------------------------


def test_student2_payload_passes_student1_data_not_the_module_result(s1):
    payload = student2_payload(s1)
    assert set(payload) == {"understanding"}
    assert payload["understanding"] is s1.data
    assert "status" not in payload["understanding"] and "reason" not in payload["understanding"]
    assert student2_payload(s1, top_n=3)["top_n"] == 3


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(ready_for_downstream=False), "ready_for_downstream"),
        (lambda d: d.pop("trip_request"), "trip_request"),
        (lambda d: d["trip_request"].update(destination=""), "destination"),
        (lambda d: d.pop("season"), "season"),
        (lambda d: d.update(expanded_fashion_context="  "), "expanded_fashion_context"),
        (lambda d: d.update(weather_clothing_requirements=[]), "weather_clothing_requirements"),
        (lambda d: d.pop("activity_style_requirements"), "activity_style_requirements"),
    ],
)
def test_student2_payload_rejects_incomplete_student1_data(s1, mutate, message):
    broken = s1.model_copy(deep=True)
    mutate(broken.data)
    with pytest.raises(HandoffContractError, match=message):
        student2_payload(broken)


def test_student2_payload_rejects_non_ok_student1_result():
    with pytest.raises(HandoffContractError, match="not 'ok'"):
        student2_payload(ModuleResult(status="needs_input", data={}))


def test_student1_context_reaches_student2_service(s1, monkeypatch):
    """Boundary test: the real Student 2 service with its pipeline isolated (no model/API)."""
    _requires_inspiration_extra()
    import travel_agent.inspiration.service as student2

    seen = {}

    def isolated_pipeline(**kwargs):
        seen.update(kwargs)
        return None  # Student 2 maps this to no_candidates; no handoff is written.

    monkeypatch.setattr(student2, "run_inspiration_pipeline", isolated_pipeline)
    result = student2.run(student2_payload(s1, top_n=4))
    assert result.status == "no_candidates"
    data = s1.data
    assert seen["expanded_context"] == data["expanded_fashion_context"]
    assert seen["top_n"] == 4
    assert seen["structured_context"] == {
        "destination": data["trip_request"]["destination"],
        "season": data["season"],
        "weather_clothing_requirements": data["weather_clothing_requirements"],
        "activity_style_requirements": data["activity_style_requirements"],
        "user_style": data["trip_request"]["style"],
    }
    assert set(STUDENT2_REQUIRED_CONTEXT) <= set(data)


# ---------------------------------------------------------------------------
# Student 2 -> Student 3 contract (offline)
# ---------------------------------------------------------------------------


def test_student3_payload_is_the_product_handoff_unchanged():
    payload = student3_payload(_handoff_result(copy.deepcopy(DOCUMENTED_HANDOFF)))
    assert payload == DOCUMENTED_HANDOFF


def test_student3_payload_fails_without_product_handoff():
    with pytest.raises(HandoffContractError, match="product_handoff is missing"):
        student3_payload(ModuleResult(status="ok", data={"top_images": []}))


@pytest.mark.parametrize("handoff", [None, "inspiration-handoffs/x", ["upstream"]])
def test_student3_payload_fails_for_wrong_handoff_type(handoff):
    with pytest.raises(HandoffContractError, match="must be an object"):
        student3_payload(_handoff_result(handoff))


@pytest.mark.parametrize("missing", ["pt", "text", "context"])
def test_student3_payload_fails_when_upstream_lacks_a_required_file(missing):
    handoff = copy.deepcopy(DOCUMENTED_HANDOFF)
    del handoff["upstream"][missing]
    with pytest.raises(HandoffContractError, match=missing):
        student3_payload(_handoff_result(handoff))


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda h: h.pop("upstream"), "upstream is missing"),
        (lambda h: h.update(reference_mode="silently-trust"), "reference_mode"),
        (lambda h: h.update(invented_field=True), "invented_field"),
        (lambda h: h["upstream"].update(extra_file="x.bin"), "unknown keys"),
        (lambda h: h["upstream"].update(pt="/abs/phase_2_output.pt"), "relative"),
        (lambda h: h["upstream"].update(text="../outside.txt"), "relative"),
    ],
)
def test_student3_payload_validates_against_student3_schema(change, message):
    handoff = copy.deepcopy(DOCUMENTED_HANDOFF)
    change(handoff)
    with pytest.raises(HandoffContractError, match=message):
        student3_payload(_handoff_result(handoff))


def test_student3_payload_rejects_non_ok_student2_result():
    with pytest.raises(HandoffContractError, match="not 'ok'"):
        student3_payload(ModuleResult(status="error", data={}))


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


def test_student4_rejects_incompatible_test_vectors_without_claiming_model_success(s1, tmp_path):
    """Boundary test: a Student 3 result reaching Student 4 stops the chain at Student 4.

    The Student 3 result comes from Student 3's own offline test mode (prepared catalogue and
    an explicit encoder test double, as in tests/test_products.py), not from Student 2.
    """
    _requires_products_extra()
    from travel_agent.products.retrieval.integration import Settings
    from travel_agent.products.retrieval.m5 import vector
    from travel_agent.products.service import run as student3_run

    class Encoder:
        identity = "explicit-test-double"
        dimension = 512

        def encode_texts(self, texts):
            return [vector(self.identity, "text", [1] + [0] * 511) for _ in texts]

        def encode_images(self, images):
            return [vector(self.identity, "image", [1] + [0] * 511) for _ in images]

    request = json.loads((ROOT / "fixtures/phase1_request.json").read_text(encoding="utf-8"))
    request["wardrobe_uploads"] = []
    s3 = student3_run(
        {
            "source": "prepared",
            "request": request,
            "understanding": s1.model_dump(mode="json"),
            "k_per_category": 3,
        },
        settings=Settings(ROOT, tmp_path, tmp_path),
        encoder=Encoder(),
    )
    assert s3.status == "ok", s3.reason

    run = PipelineRun()
    assert run_student4_stage(run, s1, s3) is False
    assert (run.stopped.module, run.stopped.status) == ("student4", "needs_input")
    assert blocking_skip_reason(run) is None  # Input/model mismatch, not a scaffold.
    assert run.stopped.reason and "not_implemented" not in run.stopped.reason
    assert build_record(run, input_description="boundary")["status"] == ("stopped_by_module_status")


@pytest.mark.parametrize(
    "value",
    [
        {"api_key": "x"},
        {"note": "C:\\Users\\someone\\trip.json"},
        {"note": "/home/someone/trip.json"},
        {"image": "data:image/png;base64,AAAA"},
        {"user_portrait": "mock://portrait"},
        {"embedding": [0.1] * 32},
    ],
)
def test_run_record_safety_check_rejects_secrets_paths_images_and_vectors(value):
    assert find_unsafe_values(value)


def test_student1_and_student5_agree_on_shared_request_fields():
    """Shared fields only (names, types, budget, currency, dates). Student 1 extension fields
    are out of scope: Student 5 receives only the formal fields it consumes."""
    result = student1_run(reviewed_payload())
    assert result.status == "ok", result.reason
    assert check_shared_request_fields(result) == []


def test_shared_request_field_check_detects_budget_unit_mismatch():
    result = student1_run(reviewed_payload())
    # Fault injection on a copy of real Student 1 output: shift budget_minor by one minor unit.
    broken = result.model_copy(deep=True)
    broken.data["hard_constraints"]["budget_minor"] += 1
    assert any(p.startswith("budget:") for p in check_shared_request_fields(broken))


@pytest.mark.parametrize("omitted", ["style", "excluded_items"])
def test_shared_request_field_check_uses_schema_defaults_for_omitted_fields(omitted):
    result = student1_run(reviewed_payload())
    assert result.status == "ok", result.reason
    # Copy of real Student 1 output without one optional field; the schema default applies.
    trimmed = result.model_copy(deep=True)
    del trimmed.data["trip_request"][omitted]
    default = SharedTripShoppingRequest.model_fields[omitted].get_default(call_default_factory=True)
    assert getattr(
        SharedTripShoppingRequest.model_validate(trimmed.data["trip_request"]), omitted
    ) == (default)
    assert check_shared_request_fields(trimmed) == []


def test_request_projection_into_student5_uses_only_the_shared_fields():
    """Minimal projection of the real Student 1 request; no skip, no extension fields."""
    result = student1_run(reviewed_payload())
    assert result.status == "ok", result.reason
    assert result.data["trip_request"]["dress_codes"]  # Student 1 extension field is present
    request = student5_request(result)
    trip = SharedTripShoppingRequest.model_validate(result.data["trip_request"])
    for name in SHARED_REQUEST_FIELDS:
        assert getattr(request, name) == getattr(trip, name), name
    dumped = request.model_dump()
    assert not set(STUDENT1_EXTENSION_FIELDS) & set(dumped)
    assert request.user_size is None


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
def test_full_pipeline_student4_scores_reach_student5_through_adapter(normal_run):
    data = _s5(normal_run)
    assert data["student4_source"] and "mock" not in data["student4_source"]
    raw_scores = {
        o["outfit_id"]: o["compatibility_score"]
        for o in normal_run.outputs["student4_raw"]["outfits"]
    }
    for item in data["recommendation_set"]["recommendations"]:
        assert item["compatibility_score"] == raw_scores[item["outfit_id"]]


@live
def test_full_pipeline_missing_compatibility_score_fails_explicitly(normal_run):
    # Fault injection: drop one field from a copy of the real Student 4 output; nothing is added.
    broken = copy.deepcopy(normal_run.outputs["student4_raw"])
    del broken["outfits"][0]["compatibility_score"]
    result = student5_run(
        {
            "action": "recommend",
            "request": normal_run.outputs["request5"].model_dump(mode="json"),
            "student4": broken,
            "beta": normal_run.outputs["beta"],
        }
    )
    assert result.status == "error"
    assert result.data["error"]["code"] == "invalid_compatibility_score"


@live
def test_full_pipeline_infeasible_budget_returns_structured_conflict():
    run = _completed(reviewed_payload(budget=1))
    data = _s5(run)
    if data["recommendation_set"] is not None:
        pytest.fail("budget 1 SGD unexpectedly feasible; pick a lower budget for this data")
    assert run.outputs["student5"].status == "no_candidates"
    assert data["conflict"]["violated_constraints"] and data["conflict"]["currency"] == "SGD"
    assert check_student5_result(run) == []


@live
def test_full_pipeline_confirms_latest_outfit_and_previews_with_mock(normal_run):
    first = copy.copy(normal_run)
    first.outputs = dict(normal_run.outputs)
    summary = confirm_and_preview(first)
    confirmation = first.outputs["confirmation"].data
    assert confirmation["state"]["session_status"] == "CONFIRMED"
    assert summary["confirmed_outfit_id"] == confirmation["confirmed_outfit"]["selected_outfit_id"]
    preview = first.outputs["preview"].data["preview"]
    assert preview["status"] == "SUCCESS" and preview["provider_name"] == "mock"
    assert preview["output_reference"].startswith("mock://")
    assert find_unsafe_values(preview) == []


@live
def test_full_pipeline_rejects_invalid_outfit_id(normal_run):
    result = student5_run(_selection(_s5(normal_run), "not-a-recommended-outfit"))
    assert result.status == "needs_input"
    assert result.data["error"]["type"] == "InvalidOutfitSelectionError"


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
def test_full_pipeline_run_record_contains_no_unsafe_values(normal_run, tmp_path):
    first = copy.copy(normal_run)
    first.outputs = dict(normal_run.outputs)
    dialogue = confirm_and_preview(first)
    record = build_record(
        first,
        input_description="reviewed request",
        dialogue=dialogue,
        checks=check_student5_result(first),
    )
    assert record["status"] == "completed"
    assert find_unsafe_values(record) == []


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
