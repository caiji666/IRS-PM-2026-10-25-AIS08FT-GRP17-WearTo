from unittest.mock import patch

import torch

from travel_agent.inspiration.service import run


# Simulated Student 1 output
student1_data = {
    "ready_for_downstream": True,
    "trip_request": {
        "destination": "Seoul",
        "style": ["minimalist"]
    },
    "season": "winter",
    "expanded_fashion_context": "Seoul winter minimalist warm coat trousers boots",
    "weather_clothing_requirements": {
        "outerwear": ["winter coat"],
        "top": ["sweater"],
        "bottom": ["trousers"],
        "footwear": ["boots"],
        "properties": ["warm"]
    },
    "activity_style_requirements": {
        "sightseeing": ["casual", "comfortable"]
    }
}


def test_inspiration_service():

    # Simulated output of the Inspiration Pipeline
    mock_phase2_output = {
        "context_embedding": torch.randn(1, 512),
        "top_images": [
            {
                "image_id": "test_image_001",
                "image_url": "https://example.com/image.jpg",
                "rank": 1,
                "similarity_score": 0.85
            }
        ],
        "top_image_embeddings": torch.randn(1, 512)
    }

    mock_evaluation_data = {
        "search_queries": ["Seoul winter minimalist outfit"],
        "retrieval_records": [],
        "candidate_records": []
    }

    mock_handoff_result = {
        "upstream": {
            "pt": "test/phase_2_output.pt",
            "text": "test/expanded_fashion_context.txt",
            "context": "test/context.json",
            "model_info": "test/model_info.json"
        },
        "reference_mode": "imported"
    }

    # Mock both the pipeline and file-based handoff
    with (
        patch(
            "travel_agent.inspiration.service.run_inspiration_pipeline",
            return_value=(mock_phase2_output, mock_evaluation_data)
        ) as mock_pipeline,

        patch(
            "travel_agent.inspiration.service.save_inspiration_handoff",
            return_value=mock_handoff_result
        ) as mock_handoff
    ):

        result = run({
            "understanding": student1_data,
            "top_n": 10
        })

        # Verify service output
        assert result.status == "ok"

        assert len(result.data["top_images"]) == 1
        assert len(result.data["context_embedding"][0]) == 512
        assert len(result.data["top_image_embeddings"][0]) == 512

        # Verify Student 3 handoff
        assert result.data["product_handoff"] == mock_handoff_result
        assert result.data["product_handoff"]["reference_mode"] == "imported"

        # Verify pipeline was called once
        mock_pipeline.assert_called_once()

        kwargs = mock_pipeline.call_args.kwargs

        assert kwargs["expanded_context"] == student1_data["expanded_fashion_context"]
        assert kwargs["structured_context"]["destination"] == "Seoul"
        assert kwargs["structured_context"]["season"] == "winter"
        assert kwargs["structured_context"]["user_style"] == ["minimalist"]
        assert kwargs["top_n"] == 10

        # Verify handoff was called once
        mock_handoff.assert_called_once()

        handoff_kwargs = mock_handoff.call_args.kwargs

        assert handoff_kwargs["phase2_output"] is mock_phase2_output
        assert handoff_kwargs["expanded_context"] == student1_data["expanded_fashion_context"]
        assert handoff_kwargs["structured_context"] == kwargs["structured_context"]

    print("Student 2 service and handoff integration test passed.")