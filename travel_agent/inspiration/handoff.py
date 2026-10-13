import torch
import json
import transformers

from uuid import uuid4

import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def save_inspiration_handoff(phase2_output, expanded_context, structured_context):
    """Save Student 2 outputs for Student 3 product retrieval."""

    # Use the same data root as Student 3
    data_root = Path(os.environ.get("STUDENT3_DATA_ROOT", "artifacts/product-retrieval")).resolve()

    handoff_root = data_root / "inspiration-handoffs"
    handoff_root.mkdir(parents=True, exist_ok=True)

    logger.info("Student 3 handoff directory: %s", handoff_root)

    # Create a unique directory for each pipeline execution
    request_id = uuid4().hex

    request_dir = handoff_root / request_id
    request_dir.mkdir(parents=True, exist_ok=False)

    logger.info("Created inspiration handoff: %s", request_id)

    # Save FashionCLIP embeddings and ranked image metadata
    pt_path = request_dir / "phase_2_output.pt"

    torch.save(phase2_output, pt_path)

    logger.info("Phase 2 output saved: %s", pt_path)

    # Save expanded fashion context
    text_path = request_dir / "expanded_fashion_context.txt"

    text_path.write_text(
        expanded_context,
        encoding="utf-8"
    )

    logger.info("Expanded fashion context saved: %s", text_path)

    # Save structured fashion context
    context_path = request_dir / "context.json"

    with open(context_path, "w", encoding="utf-8") as file:
        json.dump(structured_context, file, indent=4)

    logger.info("Structured fashion context saved: %s", context_path)

    # Save FashionCLIP model metadata
    model_profile = {
        "model_id": "patrickjohncyh/fashion-clip",
        "revision": "7e3ba62ce16b379a1ab479346b66f192e76f51b7",
        "preprocessing": "clip-rgb-exif-text77-v1",
        "transformers_version": transformers.__version__,
        "dimension": 512,
        "normalization": "L2"
    }

    model_info = {
        "context": model_profile.copy(),
        "social_images": model_profile.copy()
    }

    model_info_path = request_dir / "model_info.json"

    with open(model_info_path, "w", encoding="utf-8") as file:
        json.dump(model_info, file, indent=4)

    logger.info("Model metadata saved: %s", model_info_path)

    # Return Student 3-compatible relative paths
    upstream_paths = {
        "pt": pt_path.relative_to(data_root).as_posix(),
        "text": text_path.relative_to(data_root).as_posix(),
        "context": context_path.relative_to(data_root).as_posix(),
        "model_info": model_info_path.relative_to(data_root).as_posix()
    }

    logger.info("Student 3 handoff prepared successfully")

    return {
        "upstream": upstream_paths,
        "reference_mode": "imported"
    }
