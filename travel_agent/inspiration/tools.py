from langchain.tools import tool

from travel_agent.inspiration.service import run


@tool
def search_references(payload: dict) -> dict:
    """Retrieve and rank fashion inspiration references using Student 1's travel context."""
    return run(payload).model_dump(mode="json")
