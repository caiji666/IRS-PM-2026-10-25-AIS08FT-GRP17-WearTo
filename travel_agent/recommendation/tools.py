from langchain.tools import tool

from travel_agent.recommendation.service import run


@tool
def validate_and_recommend(payload: dict) -> dict:
    """Student 5: validate hard constraints and rank final recommendations.

    Current scaffold returns not_implemented.
    """
    return run(payload).model_dump()
