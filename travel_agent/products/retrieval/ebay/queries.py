"""Deterministic, inspectable keyword planning; no inferred brand or gender."""

import json
import re

DEFAULTS = {"TOP": "shirt", "BOTTOM": "trousers", "FOOTWEAR": "shoes", "OUTERWEAR": "jacket"}
NOUNS = {
    "TOP": ("sweater", "blouse", "polo", "t-shirt", "shirt"),
    "BOTTOM": ("shorts", "jeans", "skirt", "trousers", "pants"),
    "FOOTWEAR": ("sneakers", "boots", "sandals", "shoes"),
    "OUTERWEAR": ("raincoat", "coat", "blazer", "jacket"),
}
MODIFIERS = (
    "lightweight",
    "linen",
    "cotton",
    "casual",
    "formal",
    "minimalist",
    "waterproof",
    "warm",
)


def plan_queries(request, upstream):
    # Explicit upstream vocabulary only. Query labels never establish product category.
    text = " ".join(
        [
            request.context.model_dump_json(),
            upstream.get("expanded_fashion_context", ""),
            json.dumps(upstream.get("original_context", {}), ensure_ascii=False),
            " ".join(r.description or "" for r in request.top_social_references),
        ]
    ).casefold()
    words = set(re.findall(r"[a-z]+(?:-[a-z]+)?", text))
    modifiers = [m for m in MODIFIERS if m in words][:2]
    categories = [c.value for c in request.categories] or list(DEFAULTS)
    plan = {}
    for category in categories:
        if category not in DEFAULTS:
            continue
        nouns = [n for n in NOUNS[category] if n in words]
        noun = nouns[0] if nouns else DEFAULTS[category]
        alternate = nouns[1] if len(nouns) > 1 else DEFAULTS[category]
        specific = " ".join([*modifiers, noun])
        second = " ".join([*modifiers[:1], alternate])
        plan[category] = {
            "primary": list(dict.fromkeys([specific, second])),
            "fallback": DEFAULTS[category],
        }
    return plan
