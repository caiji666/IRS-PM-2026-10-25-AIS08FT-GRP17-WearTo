from langchain.tools import tool

from travel_agent.outfits.service import run


@tool
def build_and_score_outfits(payload: dict) -> dict:
    """Student 4: build outfits and score compatibility using offline-trained weights.

    Current scaffold returns not_implemented.
    """
    return run(payload).model_dump()
