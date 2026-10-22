"""Session-safe bridge from the local web UI to the Student 5 service."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import ValidationError

from travel_agent.recommendation.service import run as run_recommendation
from travel_agent.schemas import ConversationState
from travel_agent.schemas import TripShoppingRequest as UnderstandingRequest


class SessionError(ValueError):
    """The requested action is invalid for the current recommendation round."""


def patch_from_message(message: str) -> dict:
    """Parse refinements that Student 5 can safely apply without rerunning Students 1-4."""
    text = message.strip()
    lower = text.casefold()
    upstream_terms = (
        "destination",
        "weather",
        "activity",
        "activities",
        "style",
        "currency",
        "travel date",
        "start date",
        "end date",
    )
    if any(word in lower for word in upstream_terms):
        return {"kind": "upstream", "message": text}
    budget = re.search(r"(?:budget\D{0,20}|(?:改|预算)\D{0,8})(\d+(?:\.\d+)?)", text, re.I)
    if budget:
        return {"intent": "MODIFY_REQUIREMENTS", "set": {"budget": budget.group(1)}}
    size = re.search(r"(?:size\s*(?:is|to)?|尺码(?:改成|是)?)\s*([A-Za-z0-9-]{1,8})\b", text, re.I)
    if size:
        return {"intent": "MODIFY_REQUIREMENTS", "set": {"user_size": size.group(1).upper()}}
    excluded = re.search(r"(?:exclude|remove|排除|不要)\s+(?:product\s+)?([^,，。\s]+)", text, re.I)
    if excluded:
        return {"intent": "MODIFY_REQUIREMENTS", "add": {"excluded_items": [excluded.group(1)]}}
    # Let Student 1 interpret other natural-language trip changes. The proposal is
    # still reviewed and validated before any paid/background pipeline work starts.
    return {"kind": "upstream", "message": text}


_UNDERSTANDING_FIELDS = tuple(UnderstandingRequest.model_fields)
_STUDENT5_FIELDS = (
    "destination",
    "start_date",
    "end_date",
    "activities",
    "budget",
    "currency",
    "style",
    "excluded_items",
)


def _understanding_request(request: dict) -> UnderstandingRequest:
    projected = {field: request.get(field) for field in _UNDERSTANDING_FIELDS if field in request}
    if isinstance(projected.get("budget"), str):
        try:
            projected["budget"] = Decimal(projected["budget"])
        except InvalidOperation as exc:
            raise SessionError("The current budget is not a valid number.") from exc
    return UnderstandingRequest.model_validate(projected)


def _result_data(result: Any) -> dict:
    dumped = result.model_dump(mode="json")
    return dumped.get("data") or {}


def _review_error(data: dict, reason: str | None) -> SessionError:
    questions = data.get("questions") or []
    if questions and questions[0].get("question"):
        return SessionError(questions[0]["question"])
    errors = data.get("validation_errors") or []
    if errors and errors[0].get("message"):
        return SessionError(errors[0]["message"])
    return SessionError(reason or "The requested change could not be extracted safely.")


def propose_refinement(session: dict, message: str, understand: Callable[[dict], Any]) -> dict:
    """Validate one natural-language upstream change without executing external stages."""
    if not isinstance(message, str) or not message.strip():
        raise SessionError("Describe the requirement you want to change.")

    current = dict(session["request"])
    student1_current = _understanding_request(current)
    state = ConversationState(
        session_id=(session.get("state") or {}).get("session_id", "web"),
        request_version=(session.get("state") or {}).get("request_version", 1),
        current_request=student1_current,
        recommendation_round=(session.get("state") or {}).get("recommendation_round", 0),
        last_recommended_outfit_ids=(session.get("state") or {}).get(
            "last_recommended_outfit_ids", []
        ),
        session_status=(session.get("state") or {}).get("session_status", "READY"),
    )
    from travel_agent.understanding.dates import normalize_dates

    date_message = re.sub(
        r"^.*?\b(?:travel|trip)\s+dates?\s+(?:to|from)\s+",
        "",
        message,
        flags=re.I,
    )
    _, date_facts, date_question = normalize_dates(
        date_message,
        current=student1_current.model_dump(mode="json"),
        today=date.today(),
    )
    if date_question:
        raise SessionError(date_question)
    if date_facts:
        proposed_raw = {**student1_current.model_dump(mode="json"), **date_facts}
        pending_state = state.model_copy(
            update={"pending_request": UnderstandingRequest.model_validate(proposed_raw)}
        )
        extracted_data = {"state": pending_state.model_dump(mode="json")}
    else:
        extracted = understand({"message": message.strip(), "state": state.model_dump(mode="json")})
        extracted_data = _result_data(extracted)
        proposed_raw = extracted_data.get("proposed_request")
        if proposed_raw is None:
            raise _review_error(extracted_data, getattr(extracted, "reason", None))

    try:
        proposed = UnderstandingRequest.model_validate(proposed_raw)
    except ValidationError as exc:
        messages = "; ".join(error["msg"] for error in exc.errors())
        raise SessionError(messages) from exc

    proposed_json = proposed.model_dump(mode="json")
    updates = {
        field: proposed_json[field]
        for field in _STUDENT5_FIELDS
        if proposed_json.get(field) != current.get(field)
    }
    if not updates:
        raise SessionError("The requested change did not change the current trip request.")

    patch = {"intent": "MODIFY_REQUIREMENTS", "set": updates}
    changed = _data(
        run_recommendation(
            {"action": "modify_requirements", "state": session["state"], "patch": patch}
        )
    )
    if not changed.get("changed") or not changed.get("rerun_plan"):
        raise SessionError("The requested change did not change the current trip request.")
    returned_state = extracted_data.get("state") or state.model_dump(mode="json")
    try:
        validated_state = ConversationState.model_validate(returned_state).model_dump(mode="json")
    except ValidationError as exc:
        raise SessionError("The extracted trip state is invalid.") from exc
    return {
        "patch": patch,
        "updated_request": changed["updated_request"],
        "changed_fields": changed["changed_fields"],
        "rerun_plan": changed["rerun_plan"],
        "student1_request": proposed_json,
        "student1_state": validated_state,
    }


def _data(result: Any) -> dict:
    if result.status != "ok":
        raise SessionError(result.reason or f"Student 5 returned {result.status}")
    return result.model_dump(mode="json")["data"] or {}


def public(session: dict) -> dict:
    recommendations = (session.get("recommendation_set") or {}).get("recommendations", [])
    confirmed = session.get("confirmed_outfit")
    return {
        "top3": recommendations,
        "count": len(recommendations),
        "recommendation_round": (session.get("state") or {}).get("recommendation_round"),
        "confirmed_outfit_id": (confirmed or {}).get("selected_outfit_id"),
        "preview": session.get("preview"),
    }


def refine(session: dict, patch: dict, run: Callable[[dict], Any]) -> dict:
    changed = _data(
        run({"action": "modify_requirements", "state": session["state"], "patch": patch})
    )
    data = _data(
        run(
            {
                "action": "recommend",
                "request": changed["updated_request"],
                "student4": session["student4"],
                "state": changed["state"],
            }
        )
    )
    return {
        **session,
        "request": changed["updated_request"],
        "state": data["state"],
        "recommendation_set": data["recommendation_set"],
        "filter_summary": data["filter_summary"],
        "confirmed_outfit": None,
        "preview": None,
    }


def confirm(session: dict, outfit_id: str, run: Callable[[dict], Any]) -> dict:
    current = {
        item["outfit_id"]
        for item in (session.get("recommendation_set") or {}).get("recommendations", [])
    }
    if outfit_id not in current:
        raise SessionError("Choose an outfit from the current Top-3.")
    data = _data(
        run(
            {
                "action": "confirm",
                "state": session["state"],
                "recommendation_set": session["recommendation_set"],
                "outfit_id": outfit_id,
            }
        )
    )
    return {
        **session,
        "state": data["state"],
        "confirmed_outfit": data["confirmed_outfit"],
        "preview": None,
    }


def preview(session: dict, run: Callable[[dict], Any]) -> dict:
    if not session.get("confirmed_outfit"):
        raise SessionError("Confirm an outfit before generating its preview.")
    data = _data(run({"action": "preview_auto", "confirmed_outfit": session["confirmed_outfit"]}))
    return {**session, "preview": data}
