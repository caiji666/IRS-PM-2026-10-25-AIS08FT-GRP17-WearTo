from langchain.tools import tool

from travel_agent.products.service import run


@tool
def search_products(payload: dict) -> dict:
    """Student 3: retrieve and rank product candidates.

    Current scaffold returns not_implemented.
    """
    return run(payload).model_dump()
