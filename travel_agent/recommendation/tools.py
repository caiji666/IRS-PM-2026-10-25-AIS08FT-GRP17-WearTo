from langchain.tools import tool

from travel_agent.recommendation.service import run


@tool
def validate_and_recommend(payload: dict) -> dict:
    """Student 5: hard constraints, final Top-3 ranking, user decision and outfit preview.

    Always set `action`. "recommend": `request` (TripShoppingRequest) and `student4` (Student 4
    output with scored candidate outfits); returns the Top-3 RecommendationSet and state, or the
    structured no-feasible conflict (constraints are never relaxed). "modify_requirements":
    `state` and `patch` (its own intent stays MODIFY_REQUIREMENTS). "confirm" / "explain":
    `state`, the latest `recommendation_set` and an exact `outfit_id`. "preview":
    `confirmed_outfit` and `preview_mode` (mock or product-card preview only). Include the
    returned state on follow-up turns. Credentials must never enter the payload.
    """
    return run(payload).model_dump(mode="json")
