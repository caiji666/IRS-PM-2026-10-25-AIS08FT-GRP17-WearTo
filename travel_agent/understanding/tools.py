from langchain.tools import tool

from travel_agent.understanding.service import run


@tool
def understand_and_reason(payload: dict) -> dict:
    """Parse a TripShoppingRequest, apply patches, clarify and derive fashion context.

    Supply exactly one of request, patch or message. Include returned state on
    follow-up turns. Live weather requires weather_provider=open_meteo. Draft
    KG knowledge requires explicit allow_draft=true. Partial output and unknown
    weather never prove final outfit feasibility.
    """
    payload = dict(payload)
    if "message" in payload:
        # A tool-calling model must not approve its own extraction. The UI/CLI
        # control layer submits reviewed structured input on the following turn.
        payload["accept_extraction"] = False
    return run(payload).model_dump(mode="json")
