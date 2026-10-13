import logging

from travel_agent.inspiration.image_encoder import load_images
from travel_agent.inspiration.image_retrieval import deduplicate_images, retrieve_images

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Test with one search query
search_queries = ["Seoul winter minimalist outfit"]

# Step 1: Retrieve image metadata from Unsplash
all_retrieved_images = retrieve_images(search_queries, per_page=5)

if not all_retrieved_images:
    raise SystemExit("No images retrieved from Unsplash.")

print("\nRetrieved image records:", len(all_retrieved_images))

# Step 2: Deduplicate images
retrieved_images = deduplicate_images(all_retrieved_images)

print("Unique images:", len(retrieved_images))

# Step 3: Download images from their URLs
images, valid_image_records = load_images(retrieved_images)

if not images:
    raise SystemExit("No images loaded successfully.")

print("Successfully loaded images:", len(images))

# Verify metadata alignment
assert len(images) == len(valid_image_records)

for record in valid_image_records:
    print("Image ID:", record["image_id"], "| URL:", record["image_url"])

print("\nOnline retrieval test passed.")
