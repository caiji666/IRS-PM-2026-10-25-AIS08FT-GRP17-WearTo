"""Optional KG discovery: candidates never become request facts until selected."""

from travel_agent.understanding.activity_engine import KnowledgeGraph
from travel_agent.understanding.place_engine import KG


def _usable(edge, allow_draft):
    return edge["review_status"] == "approved" or (
        allow_draft and edge["review_status"] == "pending"
    )


def activity_suggestions(request, *, allow_draft=False, limit=4, places=None, graph=None):
    graph = graph or KnowledgeGraph.load()
    places = places or KG()
    if request.activities:
        return []
    activity_nodes = {
        n["place_activity_id"]: n
        for n in graph.nodes.values()
        if n["type"] == "Activity" and n.get("place_activity_id")
    }
    scoped = {}
    try:
        destination = places.resolve(request.destination or "")
    except ValueError:
        destination = None
    for row in places.tables["place_activity"]:
        node = activity_nodes.get(row["activity_id"])
        if node is None or destination is None:
            continue
        chain = places.ancestors(row["place_id"])
        if destination not in chain:
            continue
        scoped_chain = chain[: chain.index(destination) + 1]
        if not all(places.usable(places.places[pid], allow_draft) for pid in scoped_chain):
            continue
        if not places.usable(row, allow_draft) or not places.usable(
            places.activity_defs[row["activity_id"]], allow_draft
        ):
            continue
        candidate = scoped.setdefault(
            node["id"],
            {
                "id": node["id"],
                "value": node["name"].replace("_", " "),
                "label": "Café hopping"
                if node["id"] == "activity:cafe_visit"
                else node["name"].replace("_", " "),
                "aliases": node["aliases"],
                "scope": "destination",
                "basis": "place_kg_relationship",
                "examples": [],
                "evidence": [],
            },
        )
        place = places.places[row["place_id"]]
        example = {
            "place_id": row["place_id"],
            "name": place["name_en"] or row["place_id"],
            "place_type": place["place_type"],
        }
        if example not in candidate["examples"]:
            candidate["examples"].append(example)
        candidate["evidence"].append(
            {
                "relationship": places.proof("place_activity", row),
                "hierarchy": [places.proof("places", places.places[pid]) for pid in scoped_chain],
                "activity_definition": places.proof(
                    "activities", places.activity_defs[row["activity_id"]]
                ),
            }
        )
    if scoped:
        options = sorted(scoped.values(), key=lambda c: (-len(c["examples"]), c["value"]))
        for option in options:
            preferred_type = {
                "museum visit": "poi_museum",
                "photo spot": "poi_viewpoint",
                "sightseeing": "poi_garden",
            }.get(option["value"])
            option["examples"].sort(key=lambda e: (e["place_type"] != preferred_type, e["name"]))
            statuses = [
                proof["record"]["review_status"]
                for evidence in option["evidence"]
                for proof in [
                    evidence["relationship"],
                    evidence["activity_definition"],
                    *evidence["hierarchy"],
                ]
            ]
            option["knowledge_status"] = "draft" if "pending" in statuses else "approved"
        return options[:limit]
    # Generic category suggestions are explicit about the absence of location evidence.
    options = []
    for node in graph.nodes.values():
        if node["type"] != "Activity" or node["requires_dress_code"]:
            continue
        edges = [
            e
            for e in graph.adjacency.get(node["id"], [])
            if _usable(e, allow_draft) and not e["conditions"]
        ]
        if not edges:
            continue
        options.append(
            {
                "id": node["id"],
                "value": node["name"].replace("_", " "),
                "label": node["name"].replace("_", " "),
                "aliases": node["aliases"],
                "scope": "general",
                "basis": "activity_kg_category",
                "examples": [],
                "knowledge_status": "draft"
                if any(e["review_status"] == "pending" for e in edges)
                else "approved",
                "evidence": [{"edge": e, "source": graph.sources[e["source_id"]]} for e in edges],
            }
        )
        if len(options) == limit:
            break
    return options


def style_suggestions(request, *, allow_draft=False, limit=3, graph=None):
    if request.style:
        return []
    graph = graph or KnowledgeGraph.load()
    options = {}
    for activity in request.activities:
        context = (
            {"dress_code": request.dress_codes[activity]} if activity in request.dress_codes else {}
        )
        result = graph.query(activity, context, allow_draft=allow_draft)
        for candidate in result["style_candidates"]:
            if candidate["kind"] != "aesthetic":
                continue
            node = graph.nodes[candidate["style_id"]]
            option = options.setdefault(
                node["id"],
                {
                    "id": node["id"],
                    "value": candidate["style"],
                    "label": candidate["style"],
                    "description": node["definition"],
                    "aliases": node["aliases"],
                    "scope": "selected_activities",
                    "activities": [],
                    "evidence": [],
                },
            )
            option["activities"].append(activity)
            option["evidence"].extend(candidate["evidence_paths"])
    if options:
        values = sorted(options.values(), key=lambda c: (-len(c["activities"]), c["value"]))
        for option in values:
            statuses = [e["review_status"] for p in option["evidence"] for e in p["evidence"]]
            option["knowledge_status"] = "draft" if "pending" in statuses else "approved"
        return values[:limit]
    # With no selected activity/usable mapping, offer aesthetic preferences only.
    for edge in graph.data["edges"]:
        if (
            edge["relation"] != "SUGGESTS_AESTHETIC"
            or edge["conditions"]
            or not _usable(edge, allow_draft)
        ):
            continue
        node = graph.nodes[edge["target"]]
        option = options.setdefault(
            node["id"],
            {
                "id": node["id"],
                "value": node["name"],
                "label": node["name"],
                "description": node["definition"],
                "aliases": node["aliases"],
                "scope": "general_preference",
                "activities": [],
                "evidence": [],
                "knowledge_status": "approved",
            },
        )
        option["evidence"].append({"edge": edge, "source": graph.sources[edge["source_id"]]})
        if edge["review_status"] == "pending":
            option["knowledge_status"] = "draft"
    return list(options.values())[:limit]


def optional_suggestions(request, *, allow_draft=False):
    graph = KnowledgeGraph.load()
    return {
        "activities": activity_suggestions(request, allow_draft=allow_draft, graph=graph),
        "style": style_suggestions(request, allow_draft=allow_draft, graph=graph),
        "backend": "local_kg_snapshot",
        "selection_required": True,
        "limitations": [
            "Options are suggestions, not selected user activities or preferences.",
            "General activity ideas do not establish destination or venue availability.",
            "Aesthetic options are alternatives and do not replace venue dress codes.",
            "No prices, opening hours, itinerary feasibility or availability are verified.",
        ],
    }
