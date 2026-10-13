from travel_agent.inspiration.service import prepare_inspiration_context
from travel_agent.inspiration.query_generator import generate_search_queries


# Simulated Student 1 output
student1_data = {
    "trip_request": {
        "destination": "Seoul",
        "style": ["minimalist"]
    },
    "season": "winter",
    "expanded_fashion_context": (
        "Seoul winter winter coat puffer jacket sweater "
        "long trousers boots sightseeing casual minimalist"
    ),
    "weather_clothing_requirements": {
        "outerwear": ["winter coat", "puffer jacket"],
        "top": ["sweater"],
        "bottom": ["long trousers"],
        "footwear": ["boots"],
        "properties": ["warm"]
    },
    "activity_style_requirements": {
        "sightseeing": ["casual", "comfortable"],
        "dinner": ["smart casual", "polished"]
    }
}

# Test context adapter
expanded_context, structured_context = prepare_inspiration_context(student1_data)

assert expanded_context == student1_data["expanded_fashion_context"]
assert structured_context["destination"] == "Seoul"
assert structured_context["season"] == "winter"
assert structured_context["user_style"] == ["minimalist"]

print("Phase 1 (Understanding) -> Phase 2 (Inspiration) context mapping passed.")

# Test query generator compatibility
queries = generate_search_queries(structured_context)

assert queries, "No search queries generated."

print("\nGenerated Search Queries:")

for i, query in enumerate(queries, start=1):
    print(f"{i}. {query}")

print("\nContext adapter and query generation passed.")