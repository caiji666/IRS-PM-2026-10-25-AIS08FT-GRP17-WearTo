import json
from itertools import product
from pathlib import Path

import joblib
import numpy as np

CATEGORIES = ("OUTERWEAR", "TOP", "BOTTOM", "FOOTWEAR")
EMBEDDING_DIM = 512

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outfits_compatibility_model.pkl"
JSON_PATH = BASE_DIR / "result.json"


def get_outfit_embedding(outfit):
    """Concatenate category-specific embeddings; zero-pad absent categories."""
    embeddings = []
    for category in CATEGORIES:
        if category in outfit:
            embedding = np.asarray(outfit[category]["embedding"]["values"], dtype=np.float32)
            if embedding.shape != (EMBEDDING_DIM,):
                raise ValueError(
                    f"Expected {EMBEDDING_DIM}-dimensional embedding for {category}; "
                    f"got shape {embedding.shape}"
                )
        else:
            embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        embeddings.append(embedding)
    return np.concatenate(embeddings)


def run():
    """Rank all valid outfits formed from bundle['top_k_products'].

    Combination rules: TOP, BOTTOM, and FOOTWEAR are mandatory;
    OUTERWEAR is optional; at most one product per category.

    Returns a JSON string sorted by descending probability of
    compatible (class 1). Outfit IDs reflect the resulting rank.
    """
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        result = json.load(f)

    top_k_products = result["bundle"]["top_k_products"]
    ranked_products = result["bundle"]["product_ranking"]["ranked_products"]

    # Map each product ID to its product score
    product_scores = {
        entry["item"]["product"]["product_id"]: entry["score"]
        for category_products in ranked_products.values()
        for entry in category_products
    }

    # Add product_score to each product
    top_k_products = {
        category: [
            {
                **item,
                "product_score": product_scores[item["product_id"]]
            }
            for item in items
        ]
        for category, items in top_k_products.items()
    }

    options = [
        [None] + top_k_products["OUTERWEAR"],
        top_k_products["TOP"],
        top_k_products["BOTTOM"],
        top_k_products["FOOTWEAR"],
    ]

    outfits = []
    for combination in product(*options):
        outfit = {
            category: item
            for category, item in zip(CATEGORIES, combination)
            if item is not None
        }
        outfits.append(outfit)

    if not outfits:
        return json.dumps([])

    X_inference = np.stack([get_outfit_embedding(outfit) for outfit in outfits])

    # The saved pipeline includes the training-time scaler.
    model = joblib.load(MODEL_PATH)
    class_index = list(model.classes_).index(1)
    scores = model.predict_proba(X_inference)[:, class_index]

    ranked_indices = np.argsort(-scores, kind="stable")

    results = [
        {
            "outfit_id": f"O{rank:04d}",
            "compatibility_score": float(scores[index]),
            "products": outfits[index],
        }
        for rank, index in enumerate(ranked_indices, start=1)
    ]

    return json.dumps(results, ensure_ascii=False, indent=2)
