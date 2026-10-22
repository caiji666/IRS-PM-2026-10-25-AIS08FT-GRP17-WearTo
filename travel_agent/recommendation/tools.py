from langchain.tools import tool

from travel_agent.recommendation.service import run


@tool
def validate_and_recommend(payload: dict) -> dict:
    """Student 5: hard constraints, final Top-3 ranking, user decision and outfit preview.

    Always set `action`. "recommend": `request` (TripShoppingRequest) and `student4` (Student 4
    output with scored candidate outfits); returns the Top-3 RecommendationSet and state, or the
    structured no-feasible conflict (constraints are never relaxed). "modify_requirements":
    `state` and `patch` (its own intent stays MODIFY_REQUIREMENTS). "confirm" / "explain":
    `state`, the latest `recommendation_set` and an exact `outfit_id`. Include the returned state
    on follow-up turns.

    "preview_auto": `confirmed_outfit` (from "confirm"). It automatically uses the managed
    portrait when one is uploaded, otherwise it creates a styled no-person outfit composition.
    "preview" remains available for explicit low-level mode selection.
    Call it only after the user confirmed an outfit AND explicitly asked
    for a preview image, and only for that one confirmed outfit, never for every Top-3 outfit.
    A generated image may be created by a paid image API that the server configures; tell the
    user it may cost money. PORTRAIT also needs `user_portrait_ref` (the opaque ID returned by
    the system's portrait upload flow) and `portrait_consent: true`, set only after the user
    explicitly agreed to their portrait being sent to the image service. Never accept a raw
    local path or arbitrary URL from the user. Omit `preview_provider` (the server chooses the
    provider); "mock" is for offline tests. If generation is unavailable, the result falls
    back to product cards. Never ask for, accept or pass an API key, model name, host list or
    output path.
    """
    return run(payload).model_dump(mode="json")
