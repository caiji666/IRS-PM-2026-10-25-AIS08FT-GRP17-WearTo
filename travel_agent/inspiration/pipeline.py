import json
import logging
from pathlib import Path

import torch
import transformers
from transformers import CLIPModel, CLIPProcessor

from .image_encoder import encode_images, load_images
from .image_ranking import rank_images, select_top_images
from .image_retrieval import deduplicate_images, retrieve_images
from .query_generator import generate_search_queries
from .text_encoder import encode_context

logger = logging.getLogger(__name__)


def run_inspiration_pipeline(
    expanded_context,
    structured_context,
    top_n=10,
    max_queries=None,
    per_page=30,
):

    # Beginning Check
    if type(top_n) is not int or top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    if max_queries is not None:
        if type(max_queries) is not int or max_queries <= 0:
            raise ValueError("max_queries must be a positive integer or None.")

    if type(per_page) is not int or not 1 <= per_page <= 30:
        raise ValueError("per_page must be between 1 and 30.")

    logger.info("Starting Student 2 Inspiration Pipeline")

    # Stage 2: FashionCLIP Context Embedding
    logger.info("Stage 2: Initializing FashionCLIP")

    MODEL_NAME = "patrickjohncyh/fashion-clip"
    MODEL_REVISION = "7e3ba62ce16b379a1ab479346b66f192e76f51b7"
    PREPROCESSING_VERSION = "clip-rgb-exif-text77-v1"

    model = CLIPModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, use_fast=False)

    model.eval()

    logger.info("FashionCLIP loaded successfully")
    logger.info(
        "\nMODEL IDENTITY:\nNAME: %s\nREVISION: %s\nPREPROCESSING_VERSION: %s",
        MODEL_NAME,
        MODEL_REVISION,
        PREPROCESSING_VERSION,
    )
    logger.info("Text hidden size: %d", model.config.text_config.hidden_size)
    logger.info("Vision hidden size: %d", model.config.vision_config.hidden_size)
    logger.info("Embedding dimension: %d", model.config.projection_dim)

    # Stage 3: Generating Context Embedding
    logger.info("Stage 3: Generating ContextEmbedding")
    context_embedding = encode_context(expanded_context, model, processor)

    assert context_embedding.shape == (1, model.config.projection_dim)
    logger.info("ContextEmbedding generated successfully: %s", tuple(context_embedding.shape))

    # Stage 5: Generate Search Queries
    logger.info("Stage 5: Generating Search Queries")
    search_queries = generate_search_queries(structured_context)

    if not search_queries:
        logger.error("No search queries were generated.")
        return

    logger.info("Successfully generated %d search queries", len(search_queries))

    for i, query in enumerate(search_queries, start=1):
        logger.info("Query %d: %s", i, query)

    # Stage 6: Unsplash Image Retrieval
    logger.info("Stage 6: Retrieving Fashion Images")
    executed_queries = search_queries[:max_queries] if max_queries is not None else search_queries
    all_retrieved_images = retrieve_images(executed_queries, per_page=per_page)

    if not all_retrieved_images:
        logger.error("No images retrieved from Unsplash.")
        return

    logger.info("Successfully retrieved %d image records", len(all_retrieved_images))

    # Stage 7: Image Deduplication
    logger.info("Stage 7: Deduplicating Retrieved Images")
    retrieved_images = deduplicate_images(all_retrieved_images)
    logger.info("Successfully retained %d unique images", len(retrieved_images))

    for original_rank, record in enumerate(retrieved_images, start=1):
        record["original_rank"] = original_rank

    # Stage 8: Load Images for FashionCLIP
    logger.info("Stage 8: Loading Images")
    images, valid_image_records = load_images(retrieved_images)

    if not images:
        logger.error("No images loaded successfully. Terminating pipeline.")
        return

    logger.info("Image loading completed: %d valid images", len(images))

    # Stage 9: Generate FashionCLIP Image Embeddings
    logger.info("Stage 9: Generating Image Embeddings")

    image_embeddings = encode_images(images, model, processor, batch_size=8)

    assert image_embeddings.shape == (len(valid_image_records), model.config.projection_dim)

    logger.info("ImageEmbeddings generated successfully: %s", tuple(image_embeddings.shape))

    assert torch.allclose(
        image_embeddings.norm(dim=-1),
        torch.ones(len(valid_image_records)),
        atol=1e-5,
    )

    logger.info("All image embeddings are L2-normalized.")

    # Stage 10: Cosine Similarity and Image Reranking
    logger.info("Stage 10: Calculating Cosine Similarity and Reranking")
    scores, ranked_indices = rank_images(context_embedding, image_embeddings)
    logger.info("Reranking completed for %d images", len(ranked_indices))

    # Stage 11: Select Top-N Fashion Images
    logger.info("Stage 11: Selecting Top-%d Images", top_n)

    top_images, top_image_embeddings = select_top_images(
        scores, ranked_indices, valid_image_records, image_embeddings, top_n=top_n
    )

    assert len(top_images) == top_image_embeddings.shape[0]

    logger.info("Top-N selection completed: %d images", len(top_images))

    # Stage 12: Prepare Phase 2 Output
    logger.info("Stage 12: Preparing Phase 2 Output")

    phase2_output = {
        "context_embedding": context_embedding.cpu(),
        "top_images": top_images,
        "top_image_embeddings": top_image_embeddings,
    }

    logger.info("Phase 2 output prepared successfully for Next Phase")

    # Evaluation Requirement
    all_scored_records = []

    for rank, idx in enumerate(ranked_indices, start=1):
        image_index = idx.item()

        record = valid_image_records[image_index].copy()

        record["fashionclip_rank"] = rank
        record["similarity_score"] = scores[image_index].item()

        all_scored_records.append(record)

    evaluation_data = {
        "search_queries": executed_queries,
        "retrieval_records": all_retrieved_images,
        "candidate_records": all_scored_records,
    }

    logger.info("Inspiration Pipeline completed successfully")

    return phase2_output, evaluation_data


def main():

    top_n = 10
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Suppress unnecessary external library INFO logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    # Load test inputs
    logger.info("Loading Inspiration Test Inputs")
    fixture_dir = Path("tests/fixtures/inspiration")

    try:
        with open(fixture_dir / "expanded_fashion_context.txt", encoding="utf-8") as file:
            expanded_context = file.read().strip()

        with open(fixture_dir / "context.json", encoding="utf-8") as file:
            structured_context = json.load(file)

    except FileNotFoundError as error:
        logger.error("Test input file not found: %s", error)
        return

    except json.JSONDecodeError as error:
        logger.error("Invalid structured context JSON: %s", error)
        return

    if not expanded_context:
        logger.error("ExpandedFashionContext is empty.")
        return

    if not isinstance(structured_context, dict) or not structured_context:
        logger.error("Structured Fashion Context must be a non-empty dictionary.")
        return

    logger.info("Both contexts loaded successfully")
    logger.info("ExpandedFashionContext word count: %d", len(expanded_context.split()))

    result = run_inspiration_pipeline(
        expanded_context=expanded_context,
        structured_context=structured_context,
        top_n=top_n,
        max_queries=1,
        per_page=5,
    )

    if result is None:
        logger.error("Inspiration Pipeline failed.")
        return

    phase2_output, evaluation_data = result

    # Save Outputs
    logger.info("Saving Inspiration Pipeline Outputs")

    output_dir = Path("tests/inspiration/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "phase_2_output.pt"
    evaluation_path = output_dir / "evaluation_data.pt"

    torch.save(phase2_output, output_path)
    torch.save(evaluation_data, evaluation_path)

    logger.info("Phase 2 output saved successfully: %s", output_path)
    logger.info("Evaluation data saved successfully: %s", evaluation_path)

    model_info = {
        "context": {
            "model_id": "patrickjohncyh/fashion-clip",
            "revision": "7e3ba62ce16b379a1ab479346b66f192e76f51b7",
            "preprocessing": "clip-rgb-exif-text77-v1",
            "transformers_version": transformers.__version__,
            "dimension": 512,
            "normalization": "L2",
        },
        "social_images": {
            "model_id": "patrickjohncyh/fashion-clip",
            "revision": "7e3ba62ce16b379a1ab479346b66f192e76f51b7",
            "preprocessing": "clip-rgb-exif-text77-v1",
            "transformers_version": transformers.__version__,
            "dimension": 512,
            "normalization": "L2",
        },
    }

    model_info_path = output_dir / "model_info.json"

    with open(model_info_path, "w", encoding="utf-8") as file:
        json.dump(model_info, file, indent=4)

    logger.info("Model metadata saved successfully: %s", model_info_path)

    # Optional: Display Top-5 FashionCLIP images
    # from .image_visualization import display_top_images
    # display_top_images(phase2_output["top_images"], top_n=5)


if __name__ == "__main__":
    main()
