import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

from travel_agent.inspiration.image_encoder import encode_images
from travel_agent.inspiration.image_ranking import rank_images, select_top_images
from travel_agent.inspiration.query_generator import generate_search_queries
from travel_agent.inspiration.text_encoder import encode_context

fixture_dir = Path("tests/fixtures/inspiration")

structured_context = json.loads((fixture_dir / "context.json").read_text(encoding="utf-8"))

expanded_context = (
    (fixture_dir / "expanded_fashion_context.txt").read_text(encoding="utf-8").strip()
)

MODEL_NAME = "patrickjohncyh/fashion-clip"
MODEL_REVISION = "7e3ba62ce16b379a1ab479346b66f192e76f51b7"

model = CLIPModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
processor = CLIPProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, use_fast=False)

model.eval()

context_embedding = encode_context(expanded_context, model, processor)

search_queries = generate_search_queries(structured_context)

print("ContextEmbedding:", context_embedding.shape)
print("Generated queries:", len(search_queries))

image_paths = sorted(Path("tests/fixtures/inspiration/test_images").glob("*.jpg"))

assert image_paths, "No local test images found."

images = [ImageOps.exif_transpose(Image.open(path)).convert("RGB") for path in image_paths]

image_embeddings = encode_images(images, model, processor, batch_size=8)

assert context_embedding.shape == (1, 512)
assert image_embeddings.shape == (len(images), 512)

assert torch.allclose(context_embedding.norm(dim=-1), torch.ones(1), atol=1e-5)

assert torch.allclose(image_embeddings.norm(dim=-1), torch.ones(len(images)), atol=1e-5)
print("ImageEmbeddings:", image_embeddings.shape)
print("FashionCLIP text and image embeddings validated successfully.")


# Test cosine similarity and image reranking
scores, ranked_indices = rank_images(context_embedding, image_embeddings)

records = [{"image_id": path.stem, "image_path": str(path)} for path in image_paths]

top_images, top_image_embeddings = select_top_images(
    scores, ranked_indices, records, image_embeddings, top_n=min(5, len(images))
)

assert len(top_images) == top_image_embeddings.shape[0]

print("\nTop-5 FashionCLIP Ranking:")

for image in top_images:
    print(
        f"Rank {image['rank']} | "
        f"Image: {image['image_id']} | "
        f"Similarity: {image['similarity_score']:.4f}"
    )

print("Top image embeddings:", top_image_embeddings.shape)
print("FashionCLIP ranking test passed.")
