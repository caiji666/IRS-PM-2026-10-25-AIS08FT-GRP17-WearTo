"""Adapters for the existing qualified Activity KG and optional explicit POIs."""

from travel_agent.understanding.activity_engine import KnowledgeGraph
from travel_agent.understanding.activity_engine import normalize as activity_graph_normalize
from travel_agent.understanding.place_engine import KG

# Vocabulary alignment only; this does not assert the user's chosen location
# supports an activity. PlaceKG checks actual place_activity relationships.
PLACE_ACTIVITIES = {
    "sightseeing": "city_walk",
    "观光": "city_walk",
    "city walk": "city_walk",
    "hiking": "outdoor_trek",
    "徒步": "outdoor_trek",
    "museum visit": "museum_indoor",
    "museum": "museum_indoor",
    "博物馆参观": "museum_indoor",
    "religious visit": "religious_visit",
    "temple visit": "religious_visit",
    "shopping": "shopping",
    "theme park": "theme_park",
    "beach": "beach",
}


def reason_activities(request, allow_draft=False, max_depth=4, graph=None):
    graph = graph or KnowledgeGraph.load()
    requirements, results, questions = {}, {}, []
    for activity in request.activities:
        dress_code = request.dress_codes.get(activity)
        context = {"dress_code": dress_code} if dress_code else {}
        result = graph.query(
            activity,
            context,
            allow_draft=allow_draft,
            max_depth=max_depth,
            preferred_styles=request.style,
        )
        results[activity] = result
        # Keep this legacy field limited to occasion/function directions.
        # Aesthetic alternatives are exposed separately as style_candidates.
        requirements[activity] = [
            v.replace("_", " ")
            for v in [
                *[c["style"] for c in result["style_candidates"] if c["kind"] != "aesthetic"],
                *result["attributes"],
            ]
        ]
        if result["status"] == "needs_clarification":
            questions.append(
                {
                    "field": f"dress_codes.{activity}",
                    "question": f"What dress code applies to {activity}?",
                    "reason": "The KG requires an explicit occasion dress code",
                }
            )
    return requirements, results, questions


def reason_places(request, allow_draft=False):
    if not request.places:
        return None
    graph = KG()
    # This integration uses POI rules only. Long-term climate/seasonal data stays
    # in the snapshot but must not become a second or conflicting forecast.
    graph.tables["climate"] = []
    graph.tables["season_events"] = []
    activities = []
    activity_graph = KnowledgeGraph.load()
    for activity in request.activities:
        canonical = activity_graph.aliases.get(activity_graph_normalize(activity))
        mapped = (
            activity_graph.nodes[canonical].get("place_activity_id") if canonical else None
        ) or PLACE_ACTIVITIES.get(activity.casefold(), activity)
        if mapped in graph.vocab["activity"]:
            activities.append(mapped)
    result = graph.query(
        request.places,
        month=None,
        activities=activities,
        enter_interior=request.enter_interior is True,
        gender=request.gender or "unknown",
        cold_sensitive=request.cold_sensitive is True,
        allow_draft=allow_draft,
    )
    result["missing_information"] = [
        x for x in result["missing_information"] if x["reason"] != "month_or_climate_missing"
    ]
    for place in result["places"]:
        place["source_name"] = place["name"]
        place["name"] = graph.places[place["place_id"]].get("name_en") or place["place_id"]
    for requirement in result["requirements"]:
        requirement["source_label"] = requirement["name"]
        requirement["name"] = requirement["req_id"].replace("_", " ")
    result["integration_scope"] = "Explicit POI rules only; weather uses the separate provider"
    return result
