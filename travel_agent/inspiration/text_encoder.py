import logging
import torch

logger = logging.getLogger(__name__)


def encode_context(expanded_context, model, processor):

    logger.info("Generating FashionCLIP ContextEmbedding")
    # Validate input
    if not isinstance(expanded_context, str) or not expanded_context.strip():
        raise ValueError("ExpandedFashionContext must be non-empty text.")

    # Tokenize the expanded context
    text_inputs = processor(
        text=expanded_context,
        return_tensors="pt",
        truncation=True,
        max_length=model.config.text_config.max_position_embeddings
    )
    logger.info("Token Shape: %s", tuple(text_inputs["input_ids"].shape))
    
    # Generate text embedding
    with torch.inference_mode():
        context_embedding = model.get_text_features(**text_inputs)

    logger.info("Context Embedding Created, Starting Embedding Normalization")
    # L2 normalization
    context_embedding = context_embedding / context_embedding.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    logger.info("ContextEmbedding generated. Shape: %s", tuple(context_embedding.shape))
    logger.info("ContextEmbedding L2 norm: %s",context_embedding.norm(dim=-1).tolist())

    return context_embedding