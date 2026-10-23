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


_ITEM_TERMS = {
    "FOOTWEAR": r"shoes?|footwear|boots?|sneakers?|heels?",
    "TOP": r"tops?|shirts?|blouses?|sweaters?|knitwear|t-shirts?",
    "BOTTOM": r"bottoms?|pants|trousers|jeans|skirts?|shorts",
    "OUTERWEAR": r"outerwear|coats?|jackets?",
}


def _outfit_item_exclusion(text: str, session: dict) -> dict | None:
    """Resolve explicit dislike/removal against the currently displayed recommendation round."""
    if re.search(r"\b(?:don't|don’t|do not)\s+(?:remove|exclude|replace|swap)\b", text, re.I):
        raise SessionError("No item was removed. Tell me what you would like to change.")
    if not re.search(
        r"\b(?:(?:don't|don’t|do not)\s+(?:like|want)|exclude|remove|replace|swap|change)\b",
        text,
        re.I,
    ):
        return None
    categories = {
        category
        for category, terms in _ITEM_TERMS.items()
        if re.search(rf"\b(?:{terms})\b", text, re.I)
    }
    references = re.findall(
        r"\boutfit\s*#?\s*(\d+|one|two|three|first|second|third)\b"
        r"|\b(first|second|third)\s+outfit\b",
        text,
        re.I,
    )
    if not categories and not references:
        return None
    if not categories:
        raise SessionError("Which item would you like to replace: shoes, top, bottoms or coat?")
    if not references:
        raise SessionError(
            "Which outfit contains the item you dislike? For example: shoes from outfit 1."
        )
    numbers = {"one": 1, "two": 2, "three": 3, "first": 1, "second": 2, "third": 3}
    positions = set()
    for after, before in references:
        value = (after or before).casefold()
        positions.add(int(value) if value.isdigit() else numbers[value])
    if len(positions) != 1:
        raise SessionError("Please specify one current outfit at a time.")
    position = positions.pop()
    recommendations = (session.get("recommendation_set") or {}).get("recommendations", [])
    if not 1 <= position <= len(recommendations):
        raise SessionError("Choose an outfit number from the current Top-3.")
    outfit = recommendations[position - 1]
    items = outfit.get("purchase_items", []) + outfit.get("existing_items", [])
    matched = [item for item in items if item.get("category") in categories]
    if {item.get("category") for item in matched} != categories:
        raise SessionError(
            "That outfit does not contain every item you mentioned. Choose a displayed item."
        )
    return {
        "intent": "MODIFY_REQUIREMENTS",
        "add": {"excluded_items": list(dict.fromkeys(item["product_id"] for item in matched))},
    }


def patch_from_message(message: str, session: dict | None = None) -> dict:
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
    if session is not None:
        exclusion = _outfit_item_exclusion(text, session)
        if exclusion is not None:
            return exclusion
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
    # Keep the original candidate pool for later changes; each round filters all
    # cumulative exact product IDs, including owned items, before ranking.
    excluded = set(changed["updated_request"].get("excluded_items", []))
    student4 = session["student4"]
    if excluded:
        student4 = {
            **student4,
            "outfits": [
                outfit
                for outfit in student4.get("outfits", [])
                if not any(item["product_id"] in excluded for item in outfit.get("items", []))
            ],
        }
    data = _data(
        run(
            {
                "action": "recommend",
                "request": changed["updated_request"],
                "student4": student4,
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
