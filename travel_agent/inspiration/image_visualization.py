import logging
from io import BytesIO

import matplotlib.pyplot as plt
import requests
from PIL import Image

logger = logging.getLogger(__name__)


def display_top_images(top_images, top_n=5):

    logger.info("Displaying Top-%d FashionCLIP images", top_n)

    selected_images = top_images[:top_n]

    fig, axes = plt.subplots(
        1, len(selected_images), figsize=(4 * len(selected_images), 5), squeeze=False
    )

    for ax, record in zip(axes[0], selected_images):
        try:
            response = requests.get(record["image_url"], timeout=20)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content)).convert("RGB")

            ax.imshow(image)

            ax.set_title(f"Rank {record['rank']}\nSimilarity: {record['similarity_score']:.4f}")

        except (requests.RequestException, OSError) as error:
            logger.warning("Failed to display image %s: %s", record["image_id"], error)

            ax.text(0.5, 0.5, "Image unavailable", ha="center", va="center", transform=ax.transAxes)

        ax.axis("off")

    plt.tight_layout()
    plt.show()
