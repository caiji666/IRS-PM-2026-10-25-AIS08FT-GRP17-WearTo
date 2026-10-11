from langchain.tools import tool

from travel_agent.understanding.service import run


@tool
def understand_and_reason(payload: dict) -> dict:
    """Student 1: parse user needs and derive weather/activity context.

    Current scaffold returns not_implemented.
    """
    return run(payload).model_dump()
