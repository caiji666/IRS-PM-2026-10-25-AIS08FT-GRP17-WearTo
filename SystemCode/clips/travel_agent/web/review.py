"""Expose existing KG clarification requirements without inventing user dress codes."""

import copy

from travel_agent.understanding.activity_engine import KnowledgeGraph, normalize

LABELS = {
    "dinner": "dinner",
    "business meeting": "business meeting",
    "wedding guest": "wedding",
    "wedding": "wedding",
    "nightlife fine dining": "fine dining",
}


def dress_code_options(request):
    graph = KnowledgeGraph.load()
    options = {}
    for activity in request.get("activities", []):
        node_id = graph.aliases.get(normalize(activity))
        node = graph.nodes.get(node_id, {})
        if node.get("requires_dress_code"):
            options[activity] = node["supported_dress_codes"]
    return options


def confirmed_request(request, supplied):
    if not isinstance(supplied, dict):
        raise ValueError("Please confirm the activity dress codes")
    result = copy.deepcopy(request)
    codes = result.setdefault("dress_codes", {})
    options = dress_code_options(result)
    for activity, choices in options.items():
        code = supplied.get(activity, codes.get(activity))
        if code in (None, ""):
            if activity in supplied:
                codes.pop(activity, None)
            continue
        if code not in choices:
            label = LABELS.get(activity, activity)
            raise ValueError(f"Choose a supported dress code for {label}, or leave it unspecified.")
        codes[activity] = code
    return result
