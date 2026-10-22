"""Student 5: hard constraints, final ranking, user decision loop and outfit preview.

Proposal Stages 10, 12, 13 and 14. The business logic lives in
`travel_agent.recommendation.student5`; this module is the team entry point. It validates the
payload, calls the existing public functions (`student4_adapter`, `Student5Orchestrator`) and
returns a `ModuleResult`. It computes no CompatibilityScore (Student 4's Stage 11), never
relaxes a hard constraint and reads no `.env`. Only a `"preview"` call can reach a real image
API, and only when the server environment selects it (`PREVIEW_PROVIDER=openai`).

Payload contract. The top-level `action` selects the service operation and is always
required. It is not a Stage 1 user intent: the `intent` inside a `patch`
(`"MODIFY_REQUIREMENTS"`) keeps its proposal meaning.

- `"recommend"`: Stage 10 -> 12.
  - Required: `request` (Student 5 `TripShoppingRequest`; see `projection.py` for the minimal
    projection from Student 1) and `student4` (Student 4 output in the `student4_adapter`
    payload shape, PROVISIONAL until the Student 4 hand-off).
  - Optional: `beta` (default `DEFAULT_BETA`; a configurable hyperparameter, not a claimed
    optimum), `session_id`, and `state` for the next round after `"modify_requirements"`.
  - Returns `ok` with `state`, `recommendation_set`, `filter_summary`, `rejected_outfits`,
    `beta` and the Student 4 `student4_source` / `model_version`. When the request or products
    use another currency, an ephemeral copy is converted to SGD with one daily ECB rate
    snapshot before filtering. The returned state and RecommendationSet use the user's requested
    currency for display; `currency_conversion` records the internal currency, rate date and
    source.
  - Returns `no_candidates` when no outfit is feasible, with the structured `conflict`.
- `"modify_requirements"`. Required: `state` and `patch` (`TripShoppingRequestPatch`, whose
  own `intent` is `"MODIFY_REQUIREMENTS"`). Returns the new `state`, `changed`,
  `changed_fields`, `updated_request` and `rerun_plan`. The plan is only a plan; the caller
  re-runs the named stages.
- `"confirm"`. Required: `state`, `recommendation_set` (the latest one) and an exact
  `outfit_id`. Returns the new `state` and `confirmed_outfit`.
- `"explain"`. Same inputs as `"confirm"`. Returns a data-only `explanation`; the state is not
  changed and no state is returned.
- `"preview"`: Stage 14.
  - Required: `confirmed_outfit` and user-selectable `preview_mode` (`FLAT_LAY` or `PORTRAIT`).
  - Optional: `preview_provider` (only `"mock"`, for the offline `MockPreviewProvider`), and
    `user_portrait_ref` with `portrait_consent`. The reference is an opaque identifier for an
    image already stored under the server-managed portrait upload directory; raw paths and
    arbitrary URLs are not accepted.
  - Without `preview_provider`, the server environment decides
    (`preview_provider.server_preview_provider`): `PREVIEW_PROVIDER=openai` with
    `PREVIEW_API_KEY` and `PREVIEW_MODEL` uses the real `OpenAIImagePreviewProvider`. Unset or
    empty returns the product-card fallback; any other value, a missing setting, SDK or
    Pillow, or a known provider failure (`PreviewProviderError`) returns the fallback with a
    warning that contains no secret. Programming errors are not turned into a fallback.
  - Product cards are an internal fallback, never a requested preview mode. A portrait is only
    accepted with `portrait_consent=true` (otherwise `needs_input`, before any provider is
    created).
  - Returns `preview` and `preview_source`: `"mock"`, the real provider's name (`"openai"`,
    PROVISIONAL) when it delivered a generated image, or `"product_card_fallback"`.
- `"preview_auto"`: Stage 14 convenience action. Requires only `confirmed_outfit`; it uses the
  server-managed `current` portrait in PORTRAIT mode when present, otherwise FLAT_LAY mode with
  the styled body-shaped garment composition. The caller cannot supply a raw portrait path.
  - The real image provider is never selected or configured from a payload: `"openai"` is not
    an accepted `preview_provider`, and API key, model, allowed hosts and output directory are
    unknown payload keys (`needs_input`).

Status mapping:

- `error`: the payload is not an object, `action` is missing or unknown, or the Student 4
  output violates the hand-off contract (`Student4AdapterError`, with `code` and `location`).
  A missing CompatibilityScore, unavailable/stale exchange rates, unsupported currency, or a
  currency mismatch remaining after normalisation is also an `error`.
- `needs_input`: missing or unknown keys for the action, an invalid request, state or patch, or
  an `outfit_id` that is not in the latest recommendations (stale or unknown).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic import ValidationError

from travel_agent.recommendation.currency_converter import (
    CurrencyConversionError,
    load_ecb_snapshot,
    normalise_to_sgd,
    recommendation_in_currency,
)
from travel_agent.recommendation.portrait_store import (
    PortraitReferenceError,
    resolve_user_portrait_ref,
)
from travel_agent.recommendation.preview_provider import server_preview_provider
from travel_agent.recommendation.student5.confirmation_handler import OutfitAlreadyConfirmedError
from travel_agent.recommendation.student5.constraint_engine import CurrencyMismatchError
from travel_agent.recommendation.student5.dialogue_state import (
    InvalidOutfitSelectionError,
    StaleRecommendationSetError,
)
from travel_agent.recommendation.student5.models import (
    ConfirmedOutfit,
    ConversationalRecommendationState,
    PreviewDeliveryMode,
    PreviewMode,
    PreviewStatus,
    RecommendationSet,
    TripShoppingRequest,
    TripShoppingRequestPatch,
)
from travel_agent.recommendation.student5.orchestrator import Student5Orchestrator
from travel_agent.recommendation.student5.patch_processor import PatchApplicationError
from travel_agent.recommendation.student5.preview_adapter import MockPreviewProvider
from travel_agent.recommendation.student5.ranking_engine import MissingCompatibilityScoreError
from travel_agent.recommendation.student5.student4_adapter import (
    Student4AdapterError,
    adapt_student4_output,
)
from travel_agent.schemas import ModuleResult

DEFAULT_BETA = 0.5  # Default only; beta stays configurable per call.
DEFAULT_SESSION_ID = "session"
ACTIONS = ("recommend", "modify_requirements", "confirm", "explain", "preview", "preview_auto")
ALLOWED_KEYS = {
    "recommend": {"action", "request", "student4", "beta", "session_id", "state"},
    "modify_requirements": {"action", "state", "patch"},
    "confirm": {"action", "state", "recommendation_set", "outfit_id"},
    "explain": {"action", "state", "recommendation_set", "outfit_id"},
    "preview": {
        "action",
        "confirmed_outfit",
        "preview_mode",
        "preview_provider",
        "user_portrait_ref",
        "portrait_consent",
    },
    "preview_auto": {"action", "confirmed_outfit", "preview_provider"},
}
REQUIRED_KEYS = {
    "recommend": ("request", "student4"),
    "modify_requirements": ("state", "patch"),
    "confirm": ("state", "recommendation_set", "outfit_id"),
    "explain": ("state", "recommendation_set", "outfit_id"),
    "preview": ("confirmed_outfit", "preview_mode"),
    "preview_auto": ("confirmed_outfit",),
}
PREVIEW_PROVIDERS = {"mock": MockPreviewProvider}


class _PayloadError(ValueError):
    """The payload is malformed for its action (missing, unknown or mistyped keys)."""


def _dump(model: Any) -> Any:
    return None if model is None else model.model_dump(mode="json")


def _error(status: str, exc: Exception, **extra: Any) -> ModuleResult:
    detail: dict[str, Any] = {"type": type(exc).__name__, "message": str(exc)}
    detail.update(extra)
    return ModuleResult(
        status=status, reason=f"{type(exc).__name__}: {exc}", data={"error": detail}
    )


def _contract_error(message: str) -> ModuleResult:
    return ModuleResult(
        status="error",
        reason=message,
        data={"error": {"type": "InvalidAction", "message": message, "actions": list(ACTIONS)}},
    )


def _check_keys(action: str, payload: dict) -> None:
    unknown = sorted(set(payload) - ALLOWED_KEYS[action])
    if unknown:
        raise _PayloadError(f"unknown payload keys for action {action!r}: {unknown}")
    missing = [key for key in REQUIRED_KEYS[action] if key not in payload]
    if missing:
        raise _PayloadError(f"missing payload keys for action {action!r}: {missing}")


def _orchestrator(payload: dict, state: ConversationalRecommendationState | None = None):
    session_id = payload.get("session_id", state.session_id if state else DEFAULT_SESSION_ID)
    return Student5Orchestrator(beta=payload.get("beta", DEFAULT_BETA), session_id=session_id)


def _recommend(payload: dict) -> ModuleResult:
    request = TripShoppingRequest.model_validate(payload["request"])
    state = (
        ConversationalRecommendationState.model_validate(payload["state"])
        if payload.get("state") is not None
        else None
    )
    adapted = adapt_student4_output(payload["student4"])
    original_request = request
    conversion = None
    request_currency = request.currency.strip().upper()
    needs_conversion = any(
        item.currency.strip().upper() != request_currency
        for outfit in adapted.outfits
        for item in outfit.items
    )
    if needs_conversion:
        conversion = normalise_to_sgd(request, adapted.outfits, load_ecb_snapshot())
        request = conversion.request
        candidates = conversion.outfits
        if state is not None:
            state = ConversationalRecommendationState.model_validate(
                {**state.model_dump(), "current_request": request.model_dump()}
            )
    else:
        candidates = adapted.outfits
    orchestrator = _orchestrator(payload, state)
    outcome = orchestrator.run(request=request, candidate_outfits=candidates, state=state)
    if conversion is not None:
        restored_state = ConversationalRecommendationState.model_validate(
            {**outcome.state.model_dump(), "current_request": original_request.model_dump()}
        )
        displayed_recommendations = (
            recommendation_in_currency(
                outcome.recommendation_set,
                conversion.original_currency,
                conversion.snapshot,
            )
            if outcome.recommendation_set is not None
            else None
        )
        outcome = replace(
            outcome,
            state=restored_state,
            recommendation_set=displayed_recommendations,
        )
    result = outcome.filter_result
    data = {
        "state": _dump(outcome.state),
        "recommendation_set": _dump(outcome.recommendation_set),
        "filter_summary": {
            "total_candidates": result.total_candidates,
            "feasible_count": result.feasible_count,
            "rejected_count": result.rejected_count,
            "conflict_summary": {k.value: v for k, v in result.conflict_summary.items()},
        },
        "rejected_outfits": [_dump(r) for r in result.rejected_outfits],
        "conflict": _dump(outcome.conflict),
        "beta": orchestrator.beta,
        "student4_source": adapted.source,
        "model_version": adapted.model_version,
    }
    if conversion is not None:
        data["currency_conversion"] = {
            "normalised_currency": "SGD",
            "original_budget": str(conversion.original_budget),
            "original_currency": conversion.original_currency,
            "rate_date": conversion.snapshot.rate_date.isoformat(),
            "source": conversion.snapshot.source,
        }
    if outcome.recommendation_set is None:
        reason = (
            outcome.conflict.message
            if outcome.conflict is not None
            else "Student 4 supplied no candidate outfits"
        )
        return ModuleResult(status="no_candidates", data=data, reason=reason)
    return ModuleResult(status="ok", data=data)


def _modify(payload: dict) -> ModuleResult:
    state = ConversationalRecommendationState.model_validate(payload["state"])
    patch = TripShoppingRequestPatch.model_validate(payload["patch"])
    change = _orchestrator({}, state).modify_requirements(state, patch)
    return ModuleResult(
        status="ok",
        data={
            "state": _dump(change.state),
            "changed": change.update.changed,
            "changed_fields": change.update.changed_fields,
            "updated_request": _dump(change.update.updated_request),
            "rerun_plan": _dump(change.rerun_plan),
        },
    )


def _selection(payload: dict):
    state = ConversationalRecommendationState.model_validate(payload["state"])
    recommendation_set = RecommendationSet.model_validate(payload["recommendation_set"])
    outfit_id = payload["outfit_id"]
    if not isinstance(outfit_id, str):
        raise _PayloadError("outfit_id must be an exact outfit_id string")
    return state, recommendation_set, outfit_id


def _confirm(payload: dict) -> ModuleResult:
    state, recommendation_set, outfit_id = _selection(payload)
    outcome = _orchestrator({}, state).confirm(state, recommendation_set, outfit_id)
    return ModuleResult(
        status="ok",
        data={"state": _dump(outcome.state), "confirmed_outfit": _dump(outcome.confirmed_outfit)},
    )


def _explain(payload: dict) -> ModuleResult:
    state, recommendation_set, outfit_id = _selection(payload)
    explanation = _orchestrator({}, state).explain(state, recommendation_set, outfit_id)
    return ModuleResult(status="ok", data={"explanation": _dump(explanation)})


def _preview(payload: dict) -> ModuleResult:
    confirmed = ConfirmedOutfit.model_validate(payload["confirmed_outfit"])
    portrait_ref = payload.get("user_portrait_ref")
    portrait_path = (
        str(resolve_user_portrait_ref(portrait_ref)) if portrait_ref is not None else None
    )
    name = payload.get("preview_provider")
    if name is not None and name not in PREVIEW_PROVIDERS:
        raise _PayloadError(
            f"preview_provider must be one of {sorted(PREVIEW_PROVIDERS)} or absent"
        )
    # The real provider comes only from the server environment, read here at call time.
    provider = PREVIEW_PROVIDERS[name]() if name is not None else server_preview_provider()
    orchestrator = Student5Orchestrator(
        beta=DEFAULT_BETA, preview_provider=provider, session_id=confirmed.session_id
    )
    preview = orchestrator.generate_preview(
        confirmed,
        PreviewMode(payload["preview_mode"]),
        user_portrait=portrait_path,
        portrait_consent=payload.get("portrait_consent", False),
    )
    generated = (
        preview.status is PreviewStatus.SUCCESS
        and preview.delivered_mode is not PreviewDeliveryMode.PRODUCT_CARDS
    )
    if name == "mock":
        source = "mock"
    elif generated:
        source = preview.provider_name
    else:
        source = "product_card_fallback"
    return ModuleResult(status="ok", data={"preview": _dump(preview), "preview_source": source})


def _preview_auto(payload: dict) -> ModuleResult:
    try:
        resolve_user_portrait_ref("current")
        has_portrait = True
    except PortraitReferenceError:
        has_portrait = False
    forwarded = {
        "action": "preview",
        "confirmed_outfit": payload["confirmed_outfit"],
        "preview_mode": "PORTRAIT" if has_portrait else "FLAT_LAY",
    }
    if "preview_provider" in payload:
        forwarded["preview_provider"] = payload["preview_provider"]
    if has_portrait:
        forwarded.update(user_portrait_ref="current", portrait_consent=True)
    return _preview(forwarded)


HANDLERS = {
    "recommend": _recommend,
    "modify_requirements": _modify,
    "confirm": _confirm,
    "explain": _explain,
    "preview": _preview,
    "preview_auto": _preview_auto,
}


def run(payload: dict) -> ModuleResult:
    """Validate the payload, run the selected Student 5 action and return a ModuleResult."""
    if not isinstance(payload, dict):
        return _contract_error("Payload must be an object with an explicit `action`")
    if "action" not in payload:
        return _contract_error(f"Missing `action`; expected one of {list(ACTIONS)}")
    action = payload["action"]
    if not isinstance(action, str) or action not in HANDLERS:
        return _contract_error(f"Unknown action {action!r}; expected one of {list(ACTIONS)}")
    try:
        _check_keys(action, payload)
        return HANDLERS[action](payload)
    except Student4AdapterError as exc:
        return _error("error", exc, code=exc.code, location=exc.location)
    except (MissingCompatibilityScoreError, CurrencyMismatchError, CurrencyConversionError) as exc:
        return _error("error", exc)
    except (
        _PayloadError,
        ValidationError,
        PatchApplicationError,
        StaleRecommendationSetError,
        InvalidOutfitSelectionError,
        OutfitAlreadyConfirmedError,
    ) as exc:
        return _error("needs_input", exc)
    except ValueError as exc:
        # Other Student 5 validation errors (e.g. beta out of range, invalid preview mode).
        return _error("needs_input", exc)
