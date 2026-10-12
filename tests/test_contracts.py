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


def test_tools_distinguish_implemented_understanding_from_other_scaffolds():
    assert len({tool.name for tool in TOOLS}) == 5
    for tool in TOOLS:
        result = ModuleResult.model_validate(tool.invoke({"payload": {}}))
        assert result.status == (
            "needs_input" if tool.name == "understand_and_reason" else "not_implemented"
        )
        assert result.reason
