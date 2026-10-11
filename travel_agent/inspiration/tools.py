from langchain.tools import tool

from travel_agent.inspiration.service import run


@tool
def search_references(payload: dict) -> dict:
    """Student 2: retrieve and rank fashion references.

    Current scaffold returns not_implemented.
    """
    return run(payload).model_dump()
