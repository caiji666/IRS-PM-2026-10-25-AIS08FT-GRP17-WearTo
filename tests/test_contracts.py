import pytest
from pydantic import ValidationError

from travel_agent.agent import TOOLS
from travel_agent.schemas import ModuleResult, OutfitResult, ProductCandidate, TripRequest


def test_owned_items_cannot_add_purchase_cost():
    with pytest.raises(ValidationError):
        ProductCandidate(
            id="owned-shoe",
            role="footwear",
            origin="owned",
            purchase_cost_minor=100,
            currency="SGD",
        )


def test_end_date_cannot_precede_start_date():
    with pytest.raises(ValidationError):
        TripRequest(start_date="2026-12-20", end_date="2026-12-19")


@pytest.mark.parametrize("budget", [0, -1, 12.5, "200"])
def test_budget_requires_positive_integer_minor_units(budget):
    with pytest.raises(ValidationError):
        TripRequest(budget_minor=budget)


def test_recommendations_are_not_implicitly_verified():
    result = OutfitResult(
        outfit_id="fixture", item_ids=["a"], total_new_cost_minor=0, currency="SGD"
    )
    assert result.constraint_status == "unverified"


# Status each registered tool returns for an empty payload, by its current contract.
EMPTY_PAYLOAD_STATUS = {
    "understand_and_reason": "needs_input",  # Student 1: implemented
    "search_references": "needs_input",  # Student 2: implemented
    "search_products": "needs_input",  # Student 3: implemented
    "build_and_score_outfits": "not_implemented",  # Student 4: scaffold
    "validate_and_recommend": "error",  # Student 5: implemented; `action` is required
}


def test_each_tool_reports_its_current_implementation_status_for_an_empty_payload():
    """Implemented modules validate input (each by its own contract); only scaffolds return
    not_implemented."""
    assert {tool.name for tool in TOOLS} == set(EMPTY_PAYLOAD_STATUS)
    for tool in TOOLS:
        result = ModuleResult.model_validate(tool.invoke({"payload": {}}))
        assert result.status == EMPTY_PAYLOAD_STATUS[tool.name], tool.name
        assert result.reason
