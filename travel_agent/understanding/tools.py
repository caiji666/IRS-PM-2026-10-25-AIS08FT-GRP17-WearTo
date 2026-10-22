from langchain.tools import tool

from travel_agent.understanding.service import run


@tool
def understand_and_reason(payload: dict) -> dict:
    """Parse a TripShoppingRequest, apply patches, clarify and derive fashion context.

    Supply exactly one of request, patch or message. Include returned state on
    follow-up turns, including previews: state.pending_request accumulates facts
    awaiting review. Only ask for missing destination, start/end dates and budget
    including currency. Activities/styles/dress codes are optional and may be empty.
    After user review, a structured patch applies to pending facts ({} accepts them
    unchanged); never submit it without user approval. optional_questions do not
    block downstream work. optional_suggestions offers KG activity/aesthetic candidates
    for missing optional fields; present choices, allow skip, and record only the user's
    explicit selections. weather_provider=open_meteo selects forecasts inside coverage
    and historical monthly references outside; never describe history as a forecast. Draft
    KG knowledge requires explicit allow_draft=true. Partial output and unknown
    weather never prove final outfit feasibility.
    """
    payload = dict(payload)
    # A tool-calling model must not approve its own extraction, including a
    # previously saved preview. The UI/CLI or a user-approved structured patch
    # applies reviewed facts on the following turn.
    payload["accept_extraction"] = False
    return run(payload).model_dump(mode="json")
