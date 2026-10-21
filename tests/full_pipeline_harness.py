"""Module 11 harness: drive the real Student 1-5 entry points in pipeline order.

Shared by tests/test_full_pipeline_integration.py and scripts/run_full_pipeline_integration.py.

Rules kept here:
- Only the public module entry points `travel_agent.<module>.service.run` are called. Student 5
  is `travel_agent.recommendation.service.run`; its internal `student4_adapter` converts the
  Student 4 output. No module output is written by hand and no constraint, scoring, ranking,
  state or preview logic is re-implemented (final checks reuse Student 5's own functions).
- A stage that returns a non-ok status (`needs_input`, `no_candidates`, `error`,
  `not_implemented`), lacks a prerequisite (`prerequisite_missing`) or has no agreed handoff
  contract (`handoff_undefined`) stops the run; its status is kept and no later module is called.
  `not_implemented` is only ever the status a scaffold entry point itself returns.
- A handoff whose data violates a contract raises HandoffContractError (a test failure).
- Prerequisites are checked before a module is called. The checks are offline: they inspect
  installed packages, the local Hugging Face cache and the process environment; they download
  nothing and never read `.env`.
- Model/API-backed stages run only when RUN_FULL_PIPELINE_TESTS=1:
  - Student 2: FashionCLIP and the Unsplash API, which needs network;
  - Student 3: FashionCLIP and its product data source.
  Note: Student 2's own retrieval calls `load_dotenv()` during a live run.

Handoffs:
- Student 1 -> Student 2: `{"understanding": <Student 1 ModuleResult.data>}`.
- Student 2 -> Student 3: `ModuleResult.data["product_handoff"]` from
  `save_inspiration_handoff`, passed unchanged as the Student 3 payload.
- Student 4 -> Student 5: `{"action": "recommend", "request": <projected request>,
  "student4": <Student 4 output>, "beta": ...}` to `recommendation.service.run`. The request
  is the minimal projection of Student 1's validated request onto eight shared fields
  (`travel_agent.recommendation.projection`). Student 1 extension fields are not forwarded,
  and `user_size` is not set (its source is still to be agreed).

PROVISIONAL (not a team interface): payload shapes marked TODO below, the stage-owner map, the
harness statuses `prerequisite_missing` / `handoff_undefined`, and the run-record layout.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from travel_agent.recommendation.projection import (
    SHARED_REQUEST_FIELDS,
    STUDENT1_EXTENSION_FIELDS,
)
from travel_agent.schemas import SUPPORTED_CURRENCIES, ModuleResult, ProductCandidate
from travel_agent.schemas import TripShoppingRequest as SharedTripShoppingRequest

ROOT = Path(__file__).resolve().parents[1]
RUN_ENV = "RUN_FULL_PIPELINE_TESTS"
STUDENT3_SOURCE_ENV = "FULL_PIPELINE_STUDENT3_SOURCE"
REVIEWED_PAYLOAD_PATH = "examples/understanding/seoul_winter.json"
REVIEWED_PAYLOAD_SOURCE = (
    f"Student 1 offline example {REVIEWED_PAYLOAD_PATH}: reviewed structured request, "
    "synthetic weather fixture (not a forecast), draft KG explicitly enabled"
)
# Fixed natural-language request; REVIEWED_PAYLOAD_PATH holds its human-reviewed structured form.
FIXED_MESSAGE = (
    "Seoul, 2026-12-20 to 2026-12-27. Sightseeing and dinner. Budget 500 SGD. "
    "I prefer minimalist style. Dinner requires smart casual dress code."
)
MODULES = ("student1", "student2", "student3", "student4", "student5")
ENTRY_POINTS = {
    "student1": "travel_agent.understanding.service.run",
    "student2": "travel_agent.inspiration.service.run",
    "student3": "travel_agent.products.service.run",
    "student4": "travel_agent.outfits.service.run",
    "student5": "travel_agent.recommendation.service.run",
}
# PROVISIONAL proposal stage -> owning module (README team responsibilities + stage names).
STAGE_OWNERS = {
    1: "student1",
    2: "student1",
    3: "student1",
    4: "student2",
    5: "student2",
    6: "student2",
    7: "student3",
    8: "student3",
    9: "student4",
    11: "student4",
    10: "student5",
    12: "student5",
    13: "student5",
    14: "student5",
}
PRODUCT_EXTRA_MODULES = ("torch", "transformers", "numpy", "pandas", "PIL")
INSPIRATION_EXTRA_MODULES = ("torch", "transformers", "PIL", "requests")
UNSPLASH_ENV = "UNSPLASH_ACCESS_KEY"
HARNESS_STATUSES = ("prerequisite_missing", "handoff_undefined")
STUDENT2_REQUIRED_CONTEXT = (
    "season",
    "expanded_fashion_context",
    "weather_clothing_requirements",
    "activity_style_requirements",
)
UPSTREAM_REQUIRED_FILES = ("pt", "text", "context")
UPSTREAM_OPTIONAL_FILES = ("model_info",)
# Student 5 statuses that complete a recommendation round: Top-3, or the structured no-feasible
# ConstraintConflict (constraints are never relaxed).
STUDENT5_ROUND_STATUSES = ("ok", "no_candidates")


class HandoffContractError(AssertionError):
    """Data crossing a module boundary violates the agreed contract (fails the test)."""


class HandoffUndefined(Exception):
    """No agreed contract exists yet for a handoff (stops the run with `handoff_undefined`).

    Kept for hand-offs added later; the current Student 1-5 hand-offs all have a defined or
    explicitly PROVISIONAL payload."""


@dataclass
class StageRecord:
    module: str
    entry_point: str
    status: str
    data_source: str
    reason: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineRun:
    stages: list[StageRecord] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)

    @property
    def last(self) -> StageRecord | None:
        return self.stages[-1] if self.stages else None

    @property
    def completed(self) -> bool:
        last = self.last
        return (
            last is not None
            and last.module == "student5"
            and last.status in STUDENT5_ROUND_STATUSES
        )

    @property
    def stopped(self) -> StageRecord | None:
        return None if self.completed else self.last

    def stop_reason(self) -> str:
        stop = self.stopped
        if stop is None:
            return "completed"
        return (
            f"{stop.module} ({stop.entry_point}) stopped the pipeline with status "
            f"'{stop.status}': {stop.reason}"
        )


def models_allowed() -> bool:
    return os.environ.get(RUN_ENV) == "1"


def missing_product_extras() -> list[str]:
    return [name for name in PRODUCT_EXTRA_MODULES if importlib.util.find_spec(name) is None]


def missing_inspiration_extras() -> list[str]:
    return [name for name in INSPIRATION_EXTRA_MODULES if importlib.util.find_spec(name) is None]


def fashionclip_cached(cache_dir: str | os.PathLike | None = None) -> bool:
    """True when the pinned FashionCLIP revision is in a local Hugging Face cache.

    Offline lookup only (`try_to_load_from_cache`); nothing is downloaded. Students 2 and 3 pin
    the same model and revision; the values are read from Student 3's encoder module.
    """
    from huggingface_hub import try_to_load_from_cache

    from travel_agent.products.retrieval.m5.encoder import MODEL_ID, MODEL_REVISION

    def cached(filename: str) -> bool:
        found = try_to_load_from_cache(
            MODEL_ID,
            filename,
            cache_dir=None if cache_dir is None else str(cache_dir),
            revision=MODEL_REVISION,
        )
        return isinstance(found, str)

    return cached("config.json") and (cached("model.safetensors") or cached("pytorch_model.bin"))


def student2_missing_prerequisites(allow_models: bool) -> list[str]:
    """Why a live Student 2 run cannot start here (empty when it can).

    Student 2 is implemented; these are runtime prerequisites, not missing code.
    """
    missing = []
    extras = missing_inspiration_extras()
    if extras:
        missing.append(f"inspiration extra not installed ({extras})")
    if not allow_models:
        missing.append(
            f"live run (FashionCLIP + Unsplash API over the network) not enabled: set {RUN_ENV}=1"
        )
    if not extras and not fashionclip_cached():
        missing.append(
            "FashionCLIP weights (pinned revision) not in the local Hugging Face cache; "
            "the harness does not download them"
        )
    if not os.environ.get(UNSPLASH_ENV):
        missing.append(
            f"{UNSPLASH_ENV} not set in the process environment (the harness does not read .env)"
        )
    return missing


def student3_missing_prerequisites(payload: dict[str, Any], settings, allow_models: bool):
    """Why a live Student 3 run cannot start here (empty when it can)."""
    missing = []
    extras = missing_product_extras()
    if extras:
        missing.append(f"products extra not installed ({extras})")
    if not allow_models:
        missing.append(f"FashionCLIP-backed product retrieval not enabled: set {RUN_ENV}=1")
    if not extras and not fashionclip_cached(settings.model_cache):
        missing.append("FashionCLIP weights not in STUDENT3_MODEL_CACHE")
    source = payload.get("source", "photos20")
    if source == "photos20" and not (settings.data_root / "data" / "photos20").is_dir():
        missing.append(
            "product source 'photos20' (Student 3 default; product_handoff sets no source) needs "
            "data/photos20 under STUDENT3_DATA_ROOT"
        )
    elif source == "prepared" and not (settings.data_root / "data/catalogue_phase1.json").is_file():
        missing.append("product source 'prepared' needs data/catalogue_phase1.json")
    elif source == "ebay" and settings.credentials is None:
        missing.append("product source 'ebay' needs STUDENT3_EBAY_CREDENTIALS and network")
    return missing


def reviewed_payload(**request_overrides: Any) -> dict[str, Any]:
    """Student 1 input: the reviewed request, optionally with controlled field changes."""
    payload = json.loads((ROOT / REVIEWED_PAYLOAD_PATH).read_text(encoding="utf-8"))
    for key, value in request_overrides.items():
        if value is None:
            payload["request"].pop(key, None)
        else:
            payload["request"][key] = value
    return payload


def _module_result(raw: Any, module: str) -> ModuleResult:
    if not isinstance(raw, ModuleResult):
        raise HandoffContractError(f"{module} must return ModuleResult, got {type(raw).__name__}")
    return raw


# ---------------------------------------------------------------------------
# Handoff checks: return problems; never repair or fill data.
# ---------------------------------------------------------------------------


def check_student1_output(result: ModuleResult) -> list[str]:
    """Student 1 -> downstream: status, state, complete request, contract fields."""
    data = result.data
    problems = []
    state = data.get("state") or {}
    if state.get("session_status") != "READY":
        problems.append(f"state.session_status is {state.get('session_status')!r}, not 'READY'")
    if not isinstance(state.get("request_version"), int) or state["request_version"] < 1:
        problems.append("state.request_version must be a positive integer")
    if data.get("ready_for_downstream") is not True:
        problems.append("ready_for_downstream is not true")
    if not data.get("contract_version"):
        problems.append("contract_version is missing")
    try:
        request = SharedTripShoppingRequest.model_validate(data.get("trip_request"))
    except ValueError as exc:
        return [*problems, f"trip_request violates TripShoppingRequest: {exc}"]
    for name in ("destination", "start_date", "end_date", "activities", "budget", "currency"):
        if not getattr(request, name):
            problems.append(f"trip_request.{name} is empty")
    if state.get("current_request") != data.get("trip_request"):
        problems.append("state.current_request differs from trip_request")
    hard = data.get("hard_constraints") or {}
    if hard.get("currency") != request.currency:
        problems.append("hard_constraints.currency differs from trip_request.currency")
    if hard.get("budget_minor") != request.budget_minor_units():
        problems.append("hard_constraints.budget_minor differs from the request budget")
    if not isinstance(data.get("expanded_fashion_context"), str):
        problems.append("expanded_fashion_context is missing")
    return problems


def check_student3_output(result: ModuleResult) -> list[str]:
    """Student 3 -> 4: complete, stable product candidates with price/currency/availability."""
    data = result.data
    problems = []
    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        return ["candidates must be a list"]
    ids = [c.get("id") for c in candidates]
    if len(set(ids)) != len(ids):
        problems.append("candidate ids are not unique")
    bundle_ids = {
        item["product_id"]
        for group in data.get("bundle", {}).get("top_k_products", {}).values()
        for item in group
    } | {item["product_id"] for item in data.get("bundle", {}).get("owned_items", [])}
    for n, raw in enumerate(candidates):
        try:
            ProductCandidate.model_validate(raw)
        except ValueError as exc:
            problems.append(f"candidates[{n}] violates ProductCandidate: {exc}")
            continue
        if raw["id"] not in bundle_ids:
            problems.append(f"candidates[{n}].id {raw['id']!r} is not traceable to the bundle")
        if raw["currency"] not in SUPPORTED_CURRENCIES:
            problems.append(f"candidates[{n}].currency {raw['currency']!r} is not supported")
    if data.get("feasibility_status") != "unverified":
        problems.append("Student 3 must not claim feasibility")
    return problems


def check_student4_to_5(raw_payload: dict[str, Any], s5: ModuleResult, currency: str) -> list[str]:
    """Student 4 -> 5 as seen in Student 5's result: the Student 4 source is kept, every
    recommended outfit carries Student 4's own CompatibilityScore, and money units and currency
    are intact."""
    from travel_agent.recommendation.student5.models import RecommendationSet

    problems = []
    data = s5.data
    source = data.get("student4_source") or ""
    if not source or "mock" in source:
        problems.append(f"Student 5 reports source {source!r}, which does not identify Student 4")
    if data.get("recommendation_set") is None:
        return problems
    raw_by_id = {o.get("outfit_id"): o for o in raw_payload.get("outfits", [])}
    for item in RecommendationSet.model_validate(data["recommendation_set"]).recommendations:
        raw = raw_by_id.get(item.outfit_id)
        if raw is None:
            problems.append(f"{item.outfit_id}: not present in Student 4 output")
            continue
        if item.compatibility_score != raw.get("compatibility_score"):
            problems.append(f"{item.outfit_id}: compatibility_score changed by the handoff")
        raw_items = {i.get("product_id"): i for i in raw.get("items", [])}
        for product in item.existing_items + item.purchase_items:
            raw_item = raw_items.get(product.product_id, {})
            if "price_minor" in raw_item:
                digits = SUPPORTED_CURRENCIES.get(raw_item.get("currency"))
                if digits is None:
                    problems.append(f"{product.product_id}: unsupported currency")
                elif product.price != Decimal(raw_item["price_minor"]).scaleb(-digits):
                    problems.append(f"{product.product_id}: minor/major unit conversion differs")
            if not product.is_user_owned and product.currency != currency:
                problems.append(f"{product.product_id}: currency {product.currency} != {currency}")
    return problems


# ---------------------------------------------------------------------------
# Handoff payloads. PROVISIONAL shapes are marked TODO(team).
# ---------------------------------------------------------------------------


def student2_payload(s1: ModuleResult, *, top_n: int | None = None) -> dict[str, Any]:
    """Student 1 -> Student 2: `understanding` is Student 1's ModuleResult.data (no envelope).

    Checks the fields `inspiration.service.prepare_inspiration_context` reads; nothing is
    filled in or repaired. `top_n` is passed only when given (Student 2's default otherwise).
    """
    if s1.status != "ok":
        raise HandoffContractError(f"Student 1 status is {s1.status!r}, not 'ok'")
    data = s1.data
    if not isinstance(data, dict):
        raise HandoffContractError("Student 1 ModuleResult.data must be a dict")
    if data.get("ready_for_downstream") is not True:
        raise HandoffContractError("Student 1 data.ready_for_downstream is not true")
    trip = data.get("trip_request")
    if not isinstance(trip, dict):
        raise HandoffContractError("Student 1 data.trip_request must be an object")
    if not isinstance(trip.get("destination"), str) or not trip["destination"].strip():
        raise HandoffContractError("Student 1 data.trip_request.destination is missing or empty")
    missing = [key for key in STUDENT2_REQUIRED_CONTEXT if key not in data]
    if missing:
        raise HandoffContractError(f"Student 1 data is missing Student 2 context fields: {missing}")
    text = data["expanded_fashion_context"]
    if not isinstance(text, str) or not text.strip():
        raise HandoffContractError("Student 1 data.expanded_fashion_context is empty")
    for key in ("weather_clothing_requirements", "activity_style_requirements"):
        if not isinstance(data[key], dict):
            raise HandoffContractError(f"Student 1 data.{key} must be an object")
    payload: dict[str, Any] = {"understanding": data}
    if top_n is not None:
        payload["top_n"] = top_n
    return payload


def student3_payload(s2: ModuleResult) -> dict[str, Any]:
    """Student 2 -> Student 3: Student 2's `product_handoff`, unchanged, as Student 3's payload.

    Validated against Student 3's public `ProductToolInput` schema plus the `upstream` file keys
    Student 3 requires (pt/text/context, optional model_info; relative paths only).
    """
    from pydantic import ValidationError

    from travel_agent.products.retrieval.integration import ProductToolInput

    if s2.status != "ok":
        raise HandoffContractError(f"Student 2 status is {s2.status!r}, not 'ok'")
    if not isinstance(s2.data, dict):
        raise HandoffContractError("Student 2 ModuleResult.data must be a dict")
    if "product_handoff" not in s2.data:
        raise HandoffContractError("Student 2 data.product_handoff is missing")
    handoff = s2.data["product_handoff"]
    if not isinstance(handoff, dict):
        raise HandoffContractError(
            f"Student 2 data.product_handoff must be an object, got {type(handoff).__name__}"
        )
    try:
        spec = ProductToolInput.model_validate(handoff)
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
        raise HandoffContractError(
            f"product_handoff violates Student 3 ProductToolInput at {fields}"
        ) from None
    upstream = spec.upstream
    if not upstream:
        raise HandoffContractError("product_handoff.upstream is missing or empty")
    missing = [key for key in UPSTREAM_REQUIRED_FILES if not upstream.get(key)]
    if missing:
        raise HandoffContractError(f"product_handoff.upstream is missing required files {missing}")
    extra = sorted(set(upstream) - set(UPSTREAM_REQUIRED_FILES) - set(UPSTREAM_OPTIONAL_FILES))
    if extra:
        raise HandoffContractError(f"product_handoff.upstream has unknown keys {extra}")
    for key, value in upstream.items():
        path = Path(value)
        if path.is_absolute() or value.startswith(("/", "\\")) or ".." in path.parts:
            raise HandoffContractError(
                f"product_handoff.upstream.{key} must be relative to STUDENT3_DATA_ROOT"
            )
    return handoff


def student4_payload(s1: ModuleResult, s3: ModuleResult) -> dict[str, Any]:
    # TODO(Student 4): outfits.service.run has no typed input contract yet. PROVISIONAL: the
    # Student 1 data and the Student 3 result data, so the real entry point is called and its
    # own status (currently `not_implemented`) is recorded.
    return {"understanding": s1.data, "products": s3.data}


def student4_adapter_payload(s4: ModuleResult) -> dict[str, Any]:
    # TODO(Student 4 + Student 5): PROVISIONAL. Assumes ModuleResult.data carries the adapter
    # payload (`outfits`, optional `model_version`) described in
    # travel_agent.recommendation.student5.student4_adapter.
    return {"status": s4.status, **s4.data}


def student5_payload(
    request: Any,
    s4: ModuleResult,
    *,
    beta: float,
    session_id: str,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Payload for `recommendation.service.run` with action "recommend": the projected request,
    the Student 4 output unchanged, and the Student 5 conversation state for a follow-up
    round."""
    payload = {
        "action": "recommend",
        "request": request.model_dump(mode="json"),
        "student4": student4_adapter_payload(s4),
        "beta": beta,
        "session_id": session_id,
    }
    if state is not None:
        payload["state"] = state
    return payload


def student5_request(s1: ModuleResult):
    """Request handed to Student 5: the minimal projection of Student 1's validated request
    onto the eight shared fields (`travel_agent.recommendation.projection`). Student 1
    extension fields are not forwarded; `user_size` stays None (source still to be agreed), so
    Student 5's size constraint is not enabled. Raises RequestProjectionError naming any
    missing required field."""
    from travel_agent.recommendation.projection import project_trip_request

    return project_trip_request(s1.data["trip_request"])


def check_shared_request_fields(s1: ModuleResult) -> list[str]:
    """Consistency of the projected request with Student 1's validated request: every shared
    field keeps its value (names, types, dates, currency) and the major-unit budget matches
    Student 1's `budget_minor`."""
    from pydantic import ValidationError

    from travel_agent.recommendation.projection import RequestProjectionError

    try:
        projected = student5_request(s1)
    except RequestProjectionError as exc:
        return [str(exc)]
    except ValidationError as exc:
        return [f"projected request violates a schema: {exc}"]
    source = SharedTripShoppingRequest.model_validate(s1.data["trip_request"])
    problems = [
        f"{name}: Student 1 and Student 5 interpret the value differently"
        for name in SHARED_REQUEST_FIELDS
        if getattr(source, name) != getattr(projected, name)
    ]
    digits = SUPPORTED_CURRENCIES.get(projected.currency)
    if digits is None:
        problems.append(f"currency {projected.currency!r} has no agreed minor unit")
    elif projected.budget.scaleb(digits) != s1.data["hard_constraints"]["budget_minor"]:
        problems.append("budget: major units differ from Student 1 budget_minor")
    return problems


def probe_student1_to_student5(s1: ModuleResult) -> dict[str, Any]:
    """Diagnostic for run records: request projection result and shared-field consistency."""
    problems = check_shared_request_fields(s1)
    return {
        "status": "projected" if not problems else "inconsistent",
        "shared_fields": list(SHARED_REQUEST_FIELDS),
        "excluded_student1_fields": list(STUDENT1_EXTENSION_FIELDS),
        "user_size": "not set (source still to be agreed; size constraint not enabled)",
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _summary(module: str, result: ModuleResult) -> dict[str, Any]:
    data = result.data
    if module == "student1":
        trip = data.get("trip_request") or {}
        return {
            "session_status": (data.get("state") or {}).get("session_status"),
            "request_version": (data.get("state") or {}).get("request_version"),
            "destination": trip.get("destination"),
            "activities": trip.get("activities"),
            "currency": trip.get("currency"),
            "context_status": data.get("context_status"),
            "contract_version": data.get("contract_version"),
            "questions": [q.get("field") for q in data.get("questions", [])],
        }
    if module == "student2":
        # Never record embeddings or image URLs: counts and handoff keys only.
        handoff = data.get("product_handoff")
        return {
            "top_image_count": len(data.get("top_images") or []),
            "has_context_embedding": "context_embedding" in data,
            "product_handoff": {
                "reference_mode": handoff.get("reference_mode"),
                "upstream_keys": sorted(handoff.get("upstream") or {}),
            }
            if isinstance(handoff, dict)
            else None,
        }
    if module == "student3":
        groups = (data.get("bundle") or {}).get("top_k_products") or {}
        return {
            "contract_version": data.get("contract_version"),
            "source": data.get("source"),
            "top_k_per_category": {k: len(v) for k, v in groups.items()},
            "candidate_count": len(data.get("candidates") or []),
            "handoff_blocker_count": len(data.get("handoff_blockers") or []),
            "model_identity": data.get("model_identity"),
        }
    if module == "student4":
        return {
            "outfit_count": len(data.get("outfits") or []),
            "model_version": data.get("model_version"),
        }
    if module == "student5":
        rs = data.get("recommendation_set")
        error = data.get("error") or {}
        return {
            "student4_source": data.get("student4_source"),
            "model_version": data.get("model_version"),
            "filter_summary": data.get("filter_summary"),
            "recommended_outfit_ids": [r["outfit_id"] for r in rs["recommendations"]] if rs else [],
            "no_feasible": data.get("conflict") is not None,
            "error_type": error.get("type"),
            "error_code": error.get("code"),
        }
    return {"data_keys": sorted(data)}


def _call(run: PipelineRun, module: str, func, payload: dict, source: str, **kwargs) -> bool:
    result = _module_result(func(payload, **kwargs), module)
    run.outputs[module] = result
    run.stages.append(
        StageRecord(
            module,
            ENTRY_POINTS[module],
            result.status,
            source,
            result.reason,
            _summary(module, result),
        )
    )
    return result.status == "ok"


def _stop(run: PipelineRun, module: str, status: str, reason: str, source: str) -> PipelineRun:
    run.stages.append(StageRecord(module, ENTRY_POINTS[module], status, source, reason))
    return run


def student3_settings():
    from travel_agent.products.retrieval.integration import Settings

    return Settings.from_environment()


def run_student4_stage(run: PipelineRun, s1: ModuleResult, s3: ModuleResult) -> bool:
    """Call the real Student 4 entry point; records its own status (e.g. not_implemented)."""
    from travel_agent.outfits.service import run as student4_run

    return _call(run, "student4", student4_run, student4_payload(s1, s3), "Student 3 result data")


def run_pipeline(
    student1_payload: dict[str, Any],
    *,
    allow_models: bool | None = None,
    beta: float = 0.5,
    session_id: str = "full-pipeline-session",
    student1_source: str = REVIEWED_PAYLOAD_SOURCE,
    previous: PipelineRun | None = None,
    resume_from: str = "student1",
    state=None,
    top_n: int | None = None,
) -> PipelineRun:
    """Run Students 1-5 in order (or resume after a requirement change, reusing upstream
    outputs of `previous`). Stops at the first stage that cannot continue."""
    from travel_agent.products.service import run as student3_run
    from travel_agent.understanding.service import run as student1_run

    allow_models = models_allowed() if allow_models is None else allow_models
    start = MODULES.index(resume_from)
    run = PipelineRun()
    if not _call(run, "student1", student1_run, student1_payload, student1_source):
        return run
    s1 = run.outputs["student1"]
    problems = check_student1_output(s1)
    if problems:
        raise HandoffContractError(f"Student 1 output: {problems}")
    if previous is not None:
        for module in MODULES[1:start]:
            run.outputs[module] = previous.outputs[module]
            run.stages.append(
                StageRecord(module, ENTRY_POINTS[module], "reused", "previous round output")
            )

    if start <= 1:
        payload = student2_payload(s1, top_n=top_n)
        source = "Student 1 ModuleResult.data"
        missing = student2_missing_prerequisites(allow_models)
        if missing:
            reason = "Student 2 is implemented; live-run prerequisites missing: " + "; ".join(
                missing
            )
            return _stop(run, "student2", "prerequisite_missing", reason, source)
        from travel_agent.inspiration.service import run as student2_run

        if not _call(run, "student2", student2_run, payload, source):
            return run
    if start <= 2:
        payload = student3_payload(run.outputs["student2"])
        source_name = payload.get("source", "photos20")
        source = f"Student 2 product_handoff (reference_mode={payload.get('reference_mode')}, "
        source += f"product source={source_name})"
        settings = student3_settings()
        missing = student3_missing_prerequisites(payload, settings, allow_models)
        if missing:
            return _stop(run, "student3", "prerequisite_missing", "; ".join(missing), source)
        if not _call(run, "student3", student3_run, payload, source, settings=settings):
            return run
        problems = check_student3_output(run.outputs["student3"])
        if problems:
            raise HandoffContractError(f"Student 3 output: {problems}")
        if not run.outputs["student3"].data["candidates"]:
            return _stop(
                run,
                "student4",
                "prerequisite_missing",
                "Student 3 projected no verified-price ProductCandidate "
                "(prepared/photos20 prices are synthetic); a verified-price source is "
                "required",
                source,
            )
    if start <= 3:
        if not run_student4_stage(run, s1, run.outputs["student3"]):
            return run

    run_student5_stage(run, s1, beta=beta, session_id=session_id, state=state)
    return run


def run_student5_stage(
    run: PipelineRun,
    s1: ModuleResult,
    *,
    beta: float,
    session_id: str,
    state: dict[str, Any] | None = None,
) -> bool:
    """Call Student 5 through `travel_agent.recommendation.service.run` (action "recommend").

    The request is the minimal projection of Student 1's request. If Student 1's request lacks
    a field Student 5 requires, the run stops with `needs_input` and the missing fields are
    recorded; nothing is invented. A projected value that disagrees with Student 1's request,
    or a Student 4 output that violates the hand-off contract, raises HandoffContractError.
    """
    from travel_agent.recommendation.projection import RequestProjectionError
    from travel_agent.recommendation.service import run as student5_run

    try:
        request = student5_request(s1)
    except RequestProjectionError as exc:
        run.stages.append(
            StageRecord(
                "student5",
                ENTRY_POINTS["student5"],
                "needs_input",
                "request projection (not called)",
                str(exc),
                {"missing_request_fields": exc.missing_fields},
            )
        )
        return False
    problems = check_shared_request_fields(s1)
    if problems:
        raise HandoffContractError(f"Student 1 / Student 5 shared request fields: {problems}")
    s4 = run.outputs["student4"]
    payload = student5_payload(request, s4, beta=beta, session_id=session_id, state=state)
    source = "Student 4 output (student4_adapter inside recommendation.service.run)"
    _call(run, "student5", student5_run, payload, source)
    s5 = run.outputs["student5"]
    if s5.status == "error" and (s5.data.get("error") or {}).get("type") == "Student4AdapterError":
        error = s5.data["error"]
        raise HandoffContractError(
            f"Student 4 -> Student 5 adapter: {error['code']} at {error['location']}: "
            f"{error['message'][:500]}"
        )
    if s5.status not in STUDENT5_ROUND_STATUSES:
        return False
    run.outputs.update(student4_raw=payload["student4"], request5=request, beta=beta)
    problems = check_student4_to_5(payload["student4"], s5, request.currency)
    if problems:
        raise HandoffContractError(f"Student 4 -> Student 5: {problems}")
    return True


def rerun_from(plan) -> str:
    """Earliest module owning an affected stage; it and every later module are re-run."""
    stages = plan["affected_stages"] if isinstance(plan, dict) else plan.affected_stages
    owners = {STAGE_OWNERS[stage] for stage in stages}
    return min(owners, key=MODULES.index)


def check_student5_result(run: PipelineRun) -> list[str]:
    """Final checks on Student 5's service result, using Student 5's own public functions
    (nothing re-implemented). The Student 4 output is re-read with the same student4_adapter to
    re-check each recommended outfit."""
    from travel_agent.recommendation.student5.constraint_engine import filter_candidate_outfits
    from travel_agent.recommendation.student5.models import RecommendationSet
    from travel_agent.recommendation.student5.ranking_engine import (
        calculate_final_outfit_score,
        calculate_outfit_product_relevance,
    )
    from travel_agent.recommendation.student5.student4_adapter import adapt_student4_output

    data = run.outputs["student5"].data
    request = run.outputs["request5"]
    adapted = {o.outfit_id: o for o in adapt_student4_output(run.outputs["student4_raw"]).outfits}
    beta = run.outputs["beta"]
    if data.get("recommendation_set") is None:
        conflict = data.get("conflict")
        if not conflict or not conflict.get("violated_constraints"):
            return ["no recommendation and no structured ConstraintConflict"]
        return []
    rs = RecommendationSet.model_validate(data["recommendation_set"])
    problems = []
    if not 1 <= len(rs.recommendations) <= 3:
        problems.append(f"{len(rs.recommendations)} recommendations, expected 1-3")
    for item in rs.recommendations:
        outfit = adapted.get(item.outfit_id)
        if outfit is None:
            problems.append(f"{item.outfit_id}: not a Student 4 outfit")
            continue
        if filter_candidate_outfits(request, [outfit]).feasible_count != 1:
            problems.append(f"{item.outfit_id}: violates a hard constraint")
        if item.compatibility_score != outfit.compatibility_score:
            problems.append(f"{item.outfit_id}: does not use the Student 4 CompatibilityScore")
        relevance = calculate_outfit_product_relevance(outfit)
        expected = calculate_final_outfit_score(outfit.compatibility_score, relevance, beta)
        if item.final_outfit_score != expected:
            problems.append(f"{item.outfit_id}: FinalOutfitScore differs from the formula")
    return problems


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------

_SECRET_KEY = re.compile(r"(api[_-]?key|token|secret|password|cookie|authori[sz]ation)", re.I)
_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/(?!/))|[A-Za-z]:\\")
_IMAGE_DATA = re.compile(r"data:image/|;base64,|base64", re.I)
MAX_LIST = 64


def find_unsafe_values(value: Any, path: str = "$") -> list[str]:
    """Paths of values a run record must not contain (secrets, local paths, images, vectors)."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if _SECRET_KEY.search(str(key)) or "portrait" in str(key).lower():
                found.append(f"{path}.{key}: sensitive key")
            found.extend(find_unsafe_values(item, f"{path}.{key}"))
    elif isinstance(value, list):
        if len(value) > MAX_LIST or (len(value) > 16 and all(isinstance(v, float) for v in value)):
            found.append(f"{path}: large list or embedding-like vector")
        for n, item in enumerate(value):
            found.extend(find_unsafe_values(item, f"{path}[{n}]"))
    elif isinstance(value, str):
        if _ABSOLUTE_PATH.search(value):
            found.append(f"{path}: absolute local path")
        if _IMAGE_DATA.search(value):
            found.append(f"{path}: image data")
        if len(value) > 2000:
            found.append(f"{path}: oversized text")
    return found


def _versions(run: PipelineRun) -> dict[str, Any]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    versions: dict[str, Any] = {
        "travel_agent": project["version"],
        "snapshot": "git metadata present (commit not read)"
        if (ROOT / ".git").exists()
        else "no git metadata (e.g. ZIP snapshot); commit not verifiable",
    }
    versions["student5"] = "travel_agent.recommendation (in this repository)"
    for stage in run.stages:
        for key in ("contract_version", "model_identity", "model_version"):
            if stage.summary.get(key):
                versions[f"{stage.module}.{key}"] = stage.summary[key]
    return versions


KNOWN_GAPS = (
    "Student 2 provides a real inspiration pipeline (inspiration.service.run); a live run needs "
    "the inspiration extra, the FashionCLIP weights, an Unsplash access key and network access",
    "Student 1 -> Student 2 uses Student 1 ModuleResult.data as `understanding`; Student 2 -> "
    "Student 3 uses Student 2's product_handoff (upstream files, reference_mode=imported)",
    "product_handoff sets no Student 3 `source`, so Student 3 uses its default 'photos20', which "
    "needs the private data/photos20 dataset under STUDENT3_DATA_ROOT",
    "Student 4 needs a trained normalized model and an explicitly compatible dynamic handoff; "
    "blocker; no real CandidateOutfit or CompatibilityScore is produced in this repository",
    "Student 3 -> Student 4 input contract is not defined (harness payload is PROVISIONAL)",
    "Real Student 4 -> Student 5 output still needs final hand-off verification",
    "Student 4 output shape for the Student 5 adapter "
    "(travel_agent.recommendation.student5.student4_adapter) is PROVISIONAL",
    "Student 5 is integrated as travel_agent.recommendation.service.run with an explicit action "
    "contract (recommend, modify_requirements, confirm, explain, preview); the Student 4 output "
    "inside `student4` stays PROVISIONAL until the real hand-off",
    "A minimal request projection into Student 5 is implemented (8 shared fields: destination, "
    "start_date, end_date, activities, budget, currency, style, excluded_items); Student 1 "
    "extension fields are not forwarded. The source of Student 5 user_size is still to be "
    "agreed; with user_size=None the size constraint is not enabled",
    "Student 3 availability is always 'unknown'; Student 5 Product.availability is a strict "
    "boolean, so verified availability must come from Student 3/4",
    "Student 5 Product requires product_url (and product_score) for purchasable items; Student 3 "
    "ProductCandidate carries neither, so Student 4 must pass them through from the bundle",
    "Student 3 prepared/photos20 prices are synthetic and never projected as ProductCandidate; "
    "verified prices need the eBay Sandbox (local credentials and network)",
    "Student 1 ConversationState and Student 5 Conversational Recommendation State both track "
    "the session; ownership of the transitions is not agreed",
)


# Pending-record status by the status of the stage that stopped the run. Any other module status
# (needs_input, no_candidates, error) maps to "stopped_by_module_status".
PENDING_STATUSES = {
    "not_implemented": "blocked_by_missing_modules",
    "prerequisite_missing": "blocked_by_missing_prerequisites",
    "handoff_undefined": "blocked_by_undefined_handoff",
}


def build_record(
    run: PipelineRun,
    *,
    input_description: str,
    dialogue: dict | None = None,
    checks: list[str] | None = None,
) -> dict[str, Any]:
    """Run record. A run that did not complete is only ever a pending report whose status says
    why it stopped (see PENDING_STATUSES)."""
    blocked = not run.completed
    if blocked:
        status = PENDING_STATUSES.get(run.stopped.status, "stopped_by_module_status")
    else:
        status = "failed" if checks else "completed"
    s1 = run.outputs.get("student1")
    record: dict[str, Any] = {
        "status": status,
        "run_time_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "input": {"message": FIXED_MESSAGE, "structured_request": input_description},
        "modules": [
            {
                "module": s.module,
                "entry_point": s.entry_point,
                "status": s.status,
                "data_source": s.data_source,
                "reason": s.reason,
                "summary": s.summary,
            }
            for s in run.stages
        ],
        "not_reached": [m for m in MODULES if m not in {s.module for s in run.stages}],
        "versions": _versions(run),
        "preview_provider": "MockPreviewProvider (no image API, no portrait)",
        "contract_probes": {
            "student1_request_to_student5": probe_student1_to_student5(s1)
            if s1 is not None and s1.status == "ok"
            else {"status": "not_probed"}
        },
        "gaps": list(KNOWN_GAPS),
    }
    if blocked:
        record["stopped_at"] = run.stop_reason()
        record["stopped_module"] = run.stopped.module
        record["stopped_status"] = run.stopped.status
        record["final_result"] = None
    else:
        stage5 = run.stages[-1].summary
        record["final_result"] = {
            "recommended_outfit_ids": stage5["recommended_outfit_ids"],
            "no_feasible": stage5["no_feasible"],
        }
        record["dialogue"] = dialogue or {}
        record["checks"] = {"problems": checks or [], "passed": not checks}
    unsafe = find_unsafe_values(record)
    if unsafe:
        raise HandoffContractError(f"run record contains unsafe values: {unsafe}")
    return record


def confirm_and_preview(run: PipelineRun) -> dict[str, Any]:
    """Confirm the first current outfit by exact outfit_id and preview it with
    MockPreviewProvider, both through `recommendation.service.run`."""
    from travel_agent.recommendation.service import run as student5_run

    data = run.outputs["student5"].data
    rs = data.get("recommendation_set")
    if rs is None:
        return {"confirmed_outfit_id": None, "preview": None}
    confirmation = student5_run(
        {
            "action": "confirm",
            "state": data["state"],
            "recommendation_set": rs,
            "outfit_id": rs["recommendations"][0]["outfit_id"],
        }
    )
    if confirmation.status != "ok":
        raise HandoffContractError(f"Student 5 confirmation failed: {confirmation.reason}")
    preview = student5_run(
        {
            "action": "preview",
            "confirmed_outfit": confirmation.data["confirmed_outfit"],
            "preview_mode": "FLAT_LAY",
            "preview_provider": "mock",
        }
    )
    if preview.status != "ok":
        raise HandoffContractError(f"Student 5 preview failed: {preview.reason}")
    run.outputs["confirmation"] = confirmation
    run.outputs["preview"] = preview
    result = preview.data["preview"]
    return {
        "confirmed_outfit_id": confirmation.data["state"]["selected_outfit_id"],
        "session_status": confirmation.data["state"]["session_status"],
        "preview": {
            "status": result["status"],
            "provider": result["provider_name"],
            "delivered_mode": result["delivered_mode"],
        },
    }
