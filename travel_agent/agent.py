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
            "returns state, pass that exact session state on every subsequent understanding "
            "call, including previews and errors. pending_request retains unreviewed facts; "
            "collect only the missing required fields, never ask the user to repeat all facts. "
            "Destination, start/end dates and budget including currency are required; activities, "
            "styles and dress codes are optional. Empty KG results are valid; optional_questions "
            "do not block proceeding. When optional_suggestions are available for missing "
            "activities or style, offer a few choices and allow skipping either or both. "
            "Keep suggestions separate from user facts; only record explicitly selected options. "
            "General suggestions do not establish destination availability. "
            "When it asks for extraction_review, show the accumulated "
            "proposed facts and wait for the user's confirmation or correction. After explicit "
            "user approval, submit a structured patch (empty if unchanged) with the returned "
            "state to apply pending facts. Never approve extraction on the user's behalf. "
            "Generate an outfit preview image only when the user explicitly asks for one, and "
            "only for the outfit the user confirmed; say that generation may incur API cost. "
            "Send a portrait only after the user explicitly consents to it. Never ask for or "
            "pass API keys or provider settings; the server configures the image provider."
        ),
    )
