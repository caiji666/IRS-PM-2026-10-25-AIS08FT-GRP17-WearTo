import logging
import os

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def retrieve_images(search_queries, per_page=30):

    load_dotenv()

    access_key = os.getenv("UNSPLASH_ACCESS_KEY")

    if not access_key:
        logger.error("Unsplash API key not found. Image retrieval cannot proceed.")
        return None

    url = "https://api.unsplash.com/search/photos"

    headers = {"Authorization": f"Client-ID {access_key}", "Accept-Version": "v1"}

    all_retrieved_images = []

    for query_number, query in enumerate(search_queries, start=1):
        logger.info("Retrieving images for Query %d: %s", query_number, query)
        params = {"query": query, "per_page": per_page, "page": 1}
        try:
            response = requests.get(url, headers=headers, params=params, timeout=20)

            response.raise_for_status()
            logger.info(
                "Unsplash API Requests: %s/%s remaining",
                response.headers.get("X-Ratelimit-Remaining"),
                response.headers.get("X-Ratelimit-Limit"),
            )

            data = response.json()

        except requests.RequestException as error:
            logger.error("Image retrieval failed for Query %d: %s", query_number, error)
            continue

        for photo in data.get("results", []):
            image_record = {
                "image_id": photo["id"],
                "image_url": photo["urls"]["small"],
                "source_url": photo["links"]["html"],
                "description": (photo.get("description") or photo.get("alt_description")),
                "photographer": photo["user"]["name"],
                "photographer_url": photo["user"]["links"]["html"],
                "query_number": query_number,
            }

            all_retrieved_images.append(image_record)

        logger.info("Query %d returned %d images", query_number, len(data.get("results", [])))

    logger.info("Total retrieved image records: %d", len(all_retrieved_images))

    return all_retrieved_images


def deduplicate_images(image_records):

    logger.info("Starting image deduplication")
    unique_images = {}
    for image in image_records:
        image_id = image["image_id"]

        if image_id not in unique_images:
            unique_images[image_id] = image

    deduplicated_images = list(unique_images.values())

    logger.info(
        "Deduplication completed: %d → %d images",
        len(image_records),
        len(deduplicated_images),
    )

    return deduplicated_images
