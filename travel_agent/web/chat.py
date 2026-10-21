"""Conversation helpers: provisional Qwen drafts stay uncommitted until trip review."""

import copy
import hashlib
import json
import re

from travel_agent.schemas import TripShoppingRequest, TripShoppingRequestPatch
from travel_agent.understanding.state import apply_patch

from .review import dress_code_options


def next_question(request, style_declined=False, today=None):
    """Compatibility helper for required slots; optional discovery uses Conversation."""
    if not request.get("destination"):
        return "destination", "Where would you like to go? Tell me the city and country."
    if needs_city(request["destination"]):
        return "destination", f"Which city in {request['destination']} will you be visiting?"
    if not request.get("start_date") or not request.get("end_date"):
        return "dates", "What are your departure and return dates? Please include the year."
    if not request.get("budget") or not request.get("currency"):
        return "budget", "What is your total clothing budget, and in which currency?"
    return None, None


def needs_city(destination):
    from travel_agent.understanding.weather_locations import country_for

    country = country_for(destination)
    return bool(country and not country.get("city_state")) or destination.casefold() in {
        "italy",
        "spain",
        "united states",
        "usa",
        "united kingdom",
        "uk",
        "korea",
    }


def contextual_answer(message, pending_field, request):
    # A bare explicit choice is bound only to the precise question just asked.
    # Never invent an activity or dress code from a generic acknowledgement.
    if pending_field == "destination" and re.fullmatch(r"[A-Za-zÀ-ž ,'-]{2,100}", message):
        return f"I am travelling to {message.strip()}."
    if pending_field == "activities":
        return f"My planned activities are {message.strip()}."
    if pending_field == "budget" and re.fullmatch(
        r"\s*(?:\d+(?:\.\d+)?\s+[A-Za-z]{3}|[A-Za-z]{3}\s+\d+(?:\.\d+)?)\.?\s*",
        message,
    ):
        return f"My total clothing budget is {message.strip().rstrip('.')}."
    if pending_field and pending_field.startswith("dress_codes."):
        activity = pending_field.split(".", 1)[1]
        choices = dress_code_options(request).get(activity, [])
        if message.strip().casefold() in choices:
            return f"{activity.capitalize()} dress code is {message.strip().casefold()}."
    if pending_field == "style" and message.strip().casefold() in {
        "no preference",
        "no style preference",
        "any style",
        "surprise me",
    }:
        return "Clear my style preferences."
    if pending_field == "style" and message.strip().casefold() in {
        "minimalist",
        "casual",
        "streetwear",
        "romantic",
        "sporty",
        "preppy",
        "retro",
    }:
        return f"I prefer {message.strip()} outfits."
    return message


def merge_candidate(extracted, previous):
    if previous:
        patch = TripShoppingRequestPatch.model_validate(extracted)
        return apply_patch(TripShoppingRequest.model_validate(previous), patch).model_dump(
            mode="json"
        )
    return TripShoppingRequest.model_validate(extracted).model_dump(mode="json")


LOCATION_FIELDS = (
    "name",
    "country",
    "country_code",
    "latitude",
    "longitude",
    "timezone",
    "admin1",
    "feature_code",
)


def public_offer(offered):
    """Only menu labels go to the browser; evidence and raw outputs stay private."""
    if not offered:
        return None
    key = hashlib.sha256(
        json.dumps(
            {
                "scope": offered["scope"],
                "field": offered["field"],
                "ids": [option["id"] for option in offered["options"]],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:20]
    return {
        "key": key,
        "field": offered["field"],
        "options": [
            {
                "number": i,
                "label": option["label"],
                "description": option.get("description", ""),
                "example": next(iter(option.get("examples", [])), {}).get("name", ""),
            }
            for i, option in enumerate(offered["options"], 1)
        ],
    }


def geocode_city(destination):
    from urllib.parse import urlencode

    from travel_agent.understanding.weather import get_json

    result = get_json(
        "https://geocoding-api.open-meteo.com/v1/search?"
        + urlencode(
            {
                "name": destination,
                "count": 10,
                "language": "en",
            }
        )
    )
    return [
        location
        for location in result.get("results", [])
        if location.get("feature_code", "PPL").startswith("PPL")
    ]


def advance_chat(conversation, *, extractor=None, responder=None, geocode=None):
    """Use the shared multi-turn controller, leaving external searches to UI approval."""
    from travel_agent.schemas import ConversationState
    from travel_agent.understanding.conversation import Conversation
    from travel_agent.understanding.service import required_questions

    conversation = copy.deepcopy(conversation)
    saved = conversation.get("dialogue")
    if not saved:
        previous = conversation.get("request")
        state = ConversationState(
            pending_request=(TripShoppingRequest.model_validate(previous) if previous else None)
        )
        saved = {
            "state": state.model_dump(mode="json"),
            "awaiting_field": conversation.get("pending_field", "destination"),
        }
    chat = Conversation(
        saved=saved,
        extractor=extractor,
        responder=responder,
        settings={"allow_draft": True, "weather_provider": "none", "structured_ui": True},
    )
    if not conversation.get("dialogue") and conversation.get("style_declined"):
        working = chat.state.get("pending_request") or chat.state["current_request"]
        chat.optional_decisions["style"] = {
            "status": "skipped",
            "scope": chat._scope("style", working),
        }
    message = conversation["messages"][-1]["content"]
    outcome = chat.turn(message)
    candidate = chat.state.get("pending_request") or chat.state["current_request"]
    result = outcome["result"]
    errors = result.get("data", {}).get("validation_errors")
    review = None
    reply = outcome["reply"]
    if chat.offered_options:
        reply = (
            f"What would you like to do in {candidate['destination']}?"
            if chat.offered_options["field"] == "activities"
            else "Do you have a preferred style?"
        )
    conversation["status"] = "collecting"
    if candidate.get("destination") and needs_city(candidate["destination"]):
        reply = f"Which city in {candidate['destination']} will you be visiting?"
        chat.awaiting_field = "destination"
        chat.offered_options = None
    elif (
        not errors
        and not result.get("data", {}).get("date_clarification")
        and result["status"] != "error"
        and not chat.offered_options
        and not required_questions(TripShoppingRequest.model_validate(candidate))
        and result.get("data", {}).get("dialogue_action") not in {"correct_review", "finish"}
    ):
        try:
            locations = (geocode or geocode_city)(candidate["destination"])
        except (OSError, RuntimeError, ValueError):
            locations = []
        if locations:
            review = {
                "request": candidate,
                "dress_code_options": dress_code_options(candidate),
                "locations": [
                    {key: location.get(key) for key in LOCATION_FIELDS} for location in locations
                ],
            }
            conversation["status"] = "review"
            reply = ""  # The review dialog already presents these exact facts.
        else:
            reply = (
                "I could not resolve that city. "
                "Please provide the city and country, or retry later."
            )
            chat.awaiting_field = "destination"
    if reply.strip():
        conversation["messages"].append({"role": "assistant", "content": reply})
    conversation["request"] = candidate
    conversation["pending_field"] = chat.awaiting_field
    conversation["dialogue"] = chat.checkpoint()
    audit = {
        "message": message,
        "effective_message": outcome.get("effective_message"),
        "understanding": result,
        "wording": outcome.get("wording"),
        "proposed_request": candidate,
    }
    return conversation, audit, review


def public_messages(messages):
    """Hide legacy UI narration while retaining the original messages in the audit."""
    visible = []
    legacy_menu = False
    for message in messages:
        content = message.get("content", "").strip()
        if message["role"] == "assistant":
            content = content.replace("Your other details are saved.", "").strip()
        if not content:
            continue
        is_menu = message["role"] == "assistant" and any(
            text in content
            for text in (
                "here are optional activity ideas:",
                "Here are optional aesthetic directions;",
                "Here are general activity ideas;",
            )
        )
        # Older chats did not tag option clicks. Only hide numeric replies
        # immediately following an actual menu, never ordinary numeric answers.
        legacy_choice = (
            legacy_menu
            and message["role"] == "user"
            and re.fullmatch(r"\d+(?:\s*(?:,|&|and)\s*\d+)*", content)
        )
        legacy_menu = is_menu
        if message.get("source") == "option_selection" or legacy_choice:
            continue
        if message["role"] == "assistant" and (
            "Here are your accumulated trip details:" in content
            or "Your trip details are confirmed:" in content
            or is_menu
        ):
            continue
        visible.append({"role": message["role"], "content": content})
    return visible


def accept_review(conversation, request):
    """Apply facts only after the existing review dialog has been confirmed."""
    from travel_agent.schemas import ConversationState
    from travel_agent.understanding.state import update_state

    conversation = copy.deepcopy(conversation)
    checkpoint = conversation.setdefault("dialogue", {})
    state = update_state(
        ConversationState.model_validate(checkpoint.get("state", {})),
        TripShoppingRequest.model_validate(request),
    )
    state.pending_request = None
    state.session_status = "READY"
    checkpoint.update(state=state.model_dump(mode="json"), offered_options=None)
    conversation["request"] = state.current_request.model_dump(mode="json")
    return conversation
