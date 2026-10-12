"""Single shared agent entry point; live provider and real tools require configuration."""

from langchain.agents import create_agent

from travel_agent.inspiration.tools import search_references
from travel_agent.outfits.tools import build_and_score_outfits
from travel_agent.products.tools import search_products
from travel_agent.recommendation.tools import validate_and_recommend
from travel_agent.understanding.tools import understand_and_reason

TOOLS = [
    understand_and_reason,
    search_references,
    search_products,
    build_and_score_outfits,
    validate_and_recommend,
]


def build_agent(model):
    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=(
            "You assist with travel outfits. Select tools based on the user's goal and tool "
            "results. Missing information requires clarification. Never invent prices, stock, "
            "weather, or budgets. A not_implemented/error result is not success: explain the "
            "limitation and stop instead of retrying the same tool. Never relax hard constraints "
            "without the user's request. Only claim feasibility after deterministic validation. "
            "Use understanding with the original message for natural-language requests. When it "
            "asks for extraction_review, show the proposed facts and wait for the user's "
            "confirmation or correction. Never approve model extraction on the user's behalf."
        ),
    )
