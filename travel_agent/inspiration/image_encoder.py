import logging
from io import BytesIO

import requests
import torch
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def load_image_from_url(image_url):

    response = requests.get(image_url, timeout=20)
    response.raise_for_status()
    # image = Image.open(BytesIO(response.content)).convert("RGB")
    image = Image.open(BytesIO(response.content))
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image


def load_images(image_records):

    images = []
    valid_image_records = []

    logger.info("Loading %d images for FashionCLIP", len(image_records))

    for record in image_records:
        try:
            image = load_image_from_url(record["image_url"])
            images.append(image)
            valid_image_records.append(record)

        except (requests.RequestException, OSError) as error:
            logger.warning("Failed to load image %s: %s", record["image_id"], error)

    logger.info("Successfully loaded %d/%d images", len(images), len(image_records))

    return images, valid_image_records


def encode_images(images, model, processor, batch_size=8):

    logger.info("Generating FashionCLIP Image Embeddings")

    all_embeddings = []

    for start in range(0, len(images), batch_size):
        batch_images = images[start : (start + batch_size)]

        # Image preprocessing
        image_inputs = processor(images=batch_images, return_tensors="pt")

        # Moves input tensors to the device where FashionCLIP runs
        image_inputs = image_inputs.to(model.device)

        # Image embedding generation
        with torch.inference_mode():
            batch_embeddings = model.get_image_features(**image_inputs)

        # L2 normalization
        batch_embeddings = batch_embeddings / (
            batch_embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        )
        all_embeddings.append(batch_embeddings.cpu())
        logger.info("Encoded %d/%d images", min(start + batch_size, len(images)), len(images))

    image_embeddings = torch.cat(all_embeddings, dim=0)
    logger.info("ImageEmbeddings generated. Shape: %s", tuple(image_embeddings.shape))

    return image_embeddings
