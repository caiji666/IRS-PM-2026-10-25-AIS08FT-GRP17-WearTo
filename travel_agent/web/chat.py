"""Conversation helpers: provisional Qwen drafts stay uncommitted until trip review."""

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from travel_agent.schemas import TripShoppingRequest, TripShoppingRequestPatch
from travel_agent.understanding.state import apply_patch

from .review import dress_code_options


def next_question(request, style_declined=False, today=None):
    today = today or datetime.now(ZoneInfo("Asia/Singapore")).date()
    destination = request.get("destination")
    if not destination:
        return "destination", "Where would you like to go? Tell me the city and country."
    if destination.casefold() in {
        "japan",
        "france",
        "italy",
        "china",
        "korea",
        "south korea",
        "united states",
        "usa",
        "uk",
        "united kingdom",
        "spain",
        "australia",
    }:
        return "destination", f"Which city in {destination} will you be visiting?"
    if not request.get("start_date") or not request.get("end_date"):
        return "dates", (
            "What are your departure and return dates? Please include the year "
            f"(for example, {today + timedelta(days=1)} to {today + timedelta(days=3)})."
        )
    first = datetime.strptime(request["start_date"], "%Y-%m-%d").date()
    last = datetime.strptime(request["end_date"], "%Y-%m-%d").date()
    if not today <= first <= last <= today + timedelta(days=15):
        return "dates", (
            "Live weather is available only within the next 15 days. "
            f"Please choose dates between {today} and {today + timedelta(days=15)}."
        )
    if not request.get("activities"):
        return (
            "activities",
            "What will you be doing there—sightseeing, dinner, museums, hiking, or something else?",
        )
    if not request.get("budget") or not request.get("currency"):
        return (
            "budget",
            "What is your total clothing budget, and in which currency? For example, 500 SGD.",
        )
    if not request.get("style") and not style_declined:
        return (
            "style",
            "Any style you love—minimalist, casual, streetwear? You can also say 'no preference'.",
        )
    for activity, options in dress_code_options(request).items():
        if request.get("dress_codes", {}).get(activity) not in options:
            return f"dress_codes.{activity}", (
                f"What is the dress code for {activity}? Please choose " + ", ".join(options) + "."
            )
    return None, None


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
