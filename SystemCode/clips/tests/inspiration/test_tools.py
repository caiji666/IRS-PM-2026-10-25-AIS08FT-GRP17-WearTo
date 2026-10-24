from unittest.mock import patch

from travel_agent.inspiration.tools import search_references
from travel_agent.schemas import ModuleResult


def test_search_references_tool():

    mock_handoff = {
        "upstream": {
            "pt": "test/phase_2_output.pt",
            "text": "test/expanded_fashion_context.txt",
            "context": "test/context.json",
            "model_info": "test/model_info.json",
        },
        "reference_mode": "imported",
    }

    mock_result = ModuleResult(
        status="ok",
        data={
            "top_images": [{"image_id": "test_image_001", "rank": 1}],
            "product_handoff": mock_handoff,
        },
    )

    with patch("travel_agent.inspiration.tools.run", return_value=mock_result) as mock_service:
        result = search_references.invoke(
            {"payload": {"understanding": {"ready_for_downstream": True}}}
        )

        assert result["status"] == "ok"
        assert result["data"]["product_handoff"] == mock_handoff

        mock_service.assert_called_once()

    print("LangChain inspiration tool handoff test passed.")
