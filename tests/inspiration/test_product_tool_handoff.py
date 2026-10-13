from unittest.mock import patch

from travel_agent.products.tools import search_products
from travel_agent.schemas import ModuleResult


def test_product_tool_handoff():
    handoff = {
        "upstream": {
            "pt": "inspiration-handoffs/test/phase_2_output.pt",
            "text": "inspiration-handoffs/test/expanded_fashion_context.txt",
            "context": "inspiration-handoffs/test/context.json",
            "model_info": "inspiration-handoffs/test/model_info.json",
        },
        "reference_mode": "imported",
    }

    mock_result = ModuleResult(
        status="ok",
        data={"candidates": []},
    )

    with patch(
        "travel_agent.products.tools.run",
        return_value=mock_result,
    ) as mock_service:
        result = search_products.invoke({"payload": handoff})

        assert result["status"] == "ok"

        mock_service.assert_called_once_with(handoff)

    print("Student 2 -> Student 3 LangChain tool handoff passed.")
