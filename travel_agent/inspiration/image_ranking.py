import logging
import torch

logger = logging.getLogger(__name__)


def rank_images(context_embedding, image_embeddings):

    logger.info("Calculating cosine similarity scores")

    # Ensure embeddings are on the same device
    context_embedding = context_embedding.cpu()
    image_embeddings = image_embeddings.cpu()

    # Cosine similarity (embeddings are already L2-normalized)
    similarities = context_embedding @ image_embeddings.T

    scores = similarities.squeeze(0)

    # Rank images by descending similarity
    ranked_indices = torch.argsort(scores, descending=True)

    logger.info("Successfully ranked %d images", len(ranked_indices))

    return scores, ranked_indices

def select_top_images(
    scores,
    ranked_indices,
    valid_image_records,
    image_embeddings,
    top_n=10
):
    logger.info("Selecting Top-%d ranked images", top_n)

    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")

    # Handle fewer available images than requested
    top_n = min(top_n, len(ranked_indices))

    top_indices = ranked_indices[:top_n]
    top_images = []

    for rank, idx in enumerate(top_indices, start=1):

        image_index = idx.item()
        image_record = valid_image_records[image_index].copy()

        image_record["rank"] = rank
        image_record["similarity_score"] = scores[image_index].item()

        top_images.append(image_record)

    # Select matching image embeddings in ranked order
    top_image_embeddings = image_embeddings[top_indices].cpu()

    logger.info("Successfully selected %d images", len(top_images))

    return top_images, top_image_embeddings