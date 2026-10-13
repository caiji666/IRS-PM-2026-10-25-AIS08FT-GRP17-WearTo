import torch
from transformers import CLIPModel, CLIPProcessor

from travel_agent.inspiration.text_encoder import encode_context

MODEL_NAME = "patrickjohncyh/fashion-clip"
MODEL_REVISION = "7e3ba62ce16b379a1ab479346b66f192e76f51b7"

model = CLIPModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
processor = CLIPProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)

model.eval()

expanded_context = "Seoul winter minimalist outfit warm coat trousers boots"

context_embedding = encode_context(
    expanded_context,
    model,
    processor
)

print("Embedding shape:", context_embedding.shape)
print("L2 norm:", context_embedding.norm(dim=-1))


from PIL import Image
from travel_agent.inspiration.image_encoder import encode_images

# Create a synthetic RGB image
test_image = Image.new("RGB", (224, 224), color="blue")

# Generate image embedding
image_embeddings = encode_images(
    images=[test_image],
    model=model,
    processor=processor,
    batch_size=1
)

print("Image embedding shape:", image_embeddings.shape)
print("Image embedding L2 norm:", image_embeddings.norm(dim=-1))