"""Dynamic outfit enumeration, trusted model scoring and explicit Student 5 projection."""

import hashlib
import itertools
import json
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .train import CATEGORIES, DIMENSION


@lru_cache(maxsize=2)
def _load_model(directory, manifest_digest):
    import joblib
    import sklearn

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest() != manifest_digest:
        raise ValueError("Model manifest changed while loading")
    if manifest.get("sklearn_version") != sklearn.__version__:
        raise ValueError("Install the scikit-learn version recorded in the model manifest")
    path = directory / "model.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["model_sha256"]:
        raise ValueError("Model checksum mismatch")
    if manifest.get("categories") != list(CATEGORIES) or manifest.get("feature_dimension") != 2048:
        raise ValueError("Model category order or input dimension mismatch")
    # This is an operator-configured, locally trained artifact, never supplied in an API payload.
    model = joblib.load(path)
    if model.n_features_in_ != 2048 or 1 not in model.classes_:
        raise ValueError("Model must accept 2048 features and contain compatible class 1")
    return model, manifest


def load_model():
    directory = Path(os.environ.get("OUTFITS_MODEL_DIR", "artifacts/outfits/normalized-v1"))
    contents = (directory / "manifest.json").read_bytes()
    return _load_model(str(directory.resolve()), hashlib.sha256(contents).hexdigest())


def unit_values(value, identity):
    if not isinstance(value, dict) or value.get("model") != identity:
        raise ValueError("Embedding model/revision/preprocessing identity mismatch")
    array = np.asarray(value.get("values"), dtype=np.float64)
    if array.shape != (DIMENSION,) or not np.isfinite(array).all():
        raise ValueError("Embedding must have 512 finite values")
    if not math.isclose(float(np.linalg.norm(array)), 1, abs_tol=1e-5):
        raise ValueError("Embedding must have unit L2 norm; do not rescale with an average factor")
    return array


def owned_score(item, request, scoring, identity):
    if not isinstance(request, dict):
        raise ValueError("Provide ranking_request with context/social vectors to score owned items")
    context = request.get("context_embedding")
    refs = [
        ref["embedding"] for ref in request.get("top_social_references", []) if ref.get("embedding")
    ]
    if not context and not refs:
        raise ValueError("Owned relevance needs a context or social reference embedding")
    vector = unit_values(item.get("garment_embedding") or item["embedding"], identity)
    cw = float(scoring.get("context_weight", 0.5)) if context else 0
    sw = float(scoring.get("social_weight", 0.5)) if refs else 0
    if cw < 0 or sw < 0 or not math.isfinite(cw + sw) or cw + sw <= 0:
        raise ValueError("Invalid relevance weights")
    context_score = float(np.clip(vector @ unit_values(context, identity), -1, 1)) if context else 0
    social_score = (
        max(float(np.clip(vector @ unit_values(ref, identity), -1, 1)) for ref in refs)
        if refs
        else 0
    )
    return (cw * context_score + sw * social_score) / (cw + sw)


def project_item(item, score, base_dir, currency):
    """No FX, no invented stock/size reliability; raw fields remain in product_catalog."""
    owned = item["owned"]
    if owned:
        if not currency:
            raise ValueError("Owned item projection needs a currency for zero purchase cost")
        amount, code = 0, currency
        # Ignored for owned items by Stage 10; never asserts merchant inventory.
        available = False
    else:
        price = item.get("price")
        if not isinstance(price, dict) or price.get("amount") is None or not price.get("currency"):
            raise ValueError("Purchase price/currency is unknown")
        amount, code = price["amount"], price["currency"]
        evidence = item.get("evidence", {})
        if not evidence.get("price_verified"):
            raise ValueError("Purchase price is not verified")
        if not evidence.get("availability_verified") or item.get("availability") not in {
            "IN_STOCK",
            "OUT_OF_STOCK",
        }:
            raise ValueError("Stock is unknown/unverified; cannot project to a stock boolean")
        available = item["availability"] == "IN_STOCK"
    image_path = item.get("image_path")
    image = None
    if image_path:
        path = Path(image_path)
        path = path if path.is_absolute() else base_dir / path
        if path.is_file():
            image = str(path.resolve())
    data = {
        "product_id": item["product_id"],
        "source": item["source_type"],
        "title": item["title"],
        "description": item.get("description"),
        "product_url": item.get("product_url"),
        "image_url": image,
        "price": amount,
        "currency": code,
        "availability": available,
        "category": item["category"],
        "product_score": score,
        "available_sizes": item.get("sizes", []),
        "size_data_reliable": bool(item.get("evidence", {}).get("sizes_verified")),
        "colour": ", ".join(item.get("colors", [])) or None,
        "is_user_owned": owned,
    }
    from travel_agent.recommendation.student5.models import Product

    return Product.model_validate(data).model_dump(mode="json")


def score_outfits(
    report,
    understanding,
    ranking_request,
    base_dir,
    *,
    model,
    manifest,
    batch_size=256,
    max_combinations=200000,
):
    if manifest.get("categories") != list(CATEGORIES) or manifest.get("feature_dimension") != 2048:
        raise ValueError("Model manifest has incompatible category order/dimension")
    identity = manifest["encoder_identity"]
    bundle = report["bundle"]
    pools = {category: [] for category in CATEGORIES}
    catalog, vectors, scores, exclusions, projection_issues = {}, {}, {}, [], []
    ranked = bundle["product_ranking"]["ranked_products"]
    product_scores = {
        r["item"]["product"]["product_id"]: r["score"] for rows in ranked.values() for r in rows
    }
    # User budget remains in Student 1; zero incremental wardrobe costs use USD.
    currency = "USD"
    signals = ranking_request or report.get("request_used")
    projected = {}
    products = [(item, False) for rows in bundle["top_k_products"].values() for item in rows]
    products.extend((item, True) for item in bundle.get("owned_items", []))
    for original, is_owned in products:
        item = dict(original)
        pid = item["product_id"]
        if pid in catalog:
            raise ValueError("Duplicate product ID in input pools")
        if item.get("owned") is not is_owned:
            raise ValueError("Input pool ownership disagrees with owned field")
        catalog[pid] = item
        if item["category"] not in pools:
            exclusions.append(
                {
                    "product_id": pid,
                    "reason": "UNKNOWN/unsupported category retained, not assigned an outfit role",
                }
            )
            continue
        vectors[pid] = unit_values(item.get("garment_embedding") or item["embedding"], identity)
        if item.get("embedding"):
            if not np.allclose(vectors[pid], unit_values(item["embedding"], identity), atol=1e-7):
                raise ValueError("Duplicate garment/image embedding fields disagree")
        score = (
            owned_score(item, signals, report.get("scoring", {}), identity)
            if is_owned
            else (float(product_scores[pid]))
        )
        if not math.isfinite(score) or not -1 <= score <= 1:
            raise ValueError("Product relevance score must be finite and in [-1,1]")
        scores[pid] = score
        item["product_score"] = score
        pools[item["category"]].append(pid)
        try:
            projected[pid] = project_item(item, score, base_dir, currency)
        except ValueError as exc:
            projection_issues.append({"product_id": pid, "reason": str(exc)})
    counts = {k: len(v) for k, v in pools.items()}
    count = (counts["OUTERWEAR"] + 1) * counts["TOP"] * counts["BOTTOM"] * counts["FOOTWEAR"]
    if count > max_combinations:
        raise ValueError(
            f"{count} combinations exceeds the safety limit; narrow the pool. "
            "No truncation was performed."
        )
    all_outfits, outfits = [], []
    options = [[None] + sorted(pools["OUTERWEAR"])] + [sorted(pools[k]) for k in CATEGORIES[1:]]
    combinations = itertools.product(*options)
    compatible_column = list(model.classes_).index(1)
    while batch := list(itertools.islice(combinations, batch_size)):
        features = np.stack(
            [
                np.concatenate(
                    [vectors[pid] if pid else np.zeros(DIMENSION) for pid in combination]
                )
                for combination in batch
            ]
        )
        probabilities = model.predict_proba(features)[:, compatible_column]
        if not np.isfinite(probabilities).all() or np.any(
            (probabilities < 0) | (probabilities > 1)
        ):
            raise ValueError("Model returned invalid compatibility probabilities")
        for combination, probability in zip(batch, probabilities):
            roles = {k: v for k, v in zip(CATEGORIES, combination) if v is not None}
            identifier = (
                "outfit_"
                + hashlib.sha256(json.dumps(roles, sort_keys=True).encode()).hexdigest()[:20]
            )
            common = {"outfit_id": identifier, "compatibility_score": float(probability)}
            all_outfits.append({**common, "product_ids": roles})
            if all(pid in projected for pid in roles.values()):
                outfits.append({**common, "items": [projected[pid] for pid in roles.values()]})
    all_outfits.sort(key=lambda row: (-row["compatibility_score"], row["outfit_id"]))
    outfits.sort(key=lambda row: (-row["compatibility_score"], row["outfit_id"]))
    return {
        "status": "ok" if outfits else "no_candidates",
        "source": "student4",
        "schema_version": "outfits.1",
        "model_version": manifest["model_version"],
        "model_metadata": {
            k: manifest.get(k)
            for k in (
                "encoder_identity",
                "model_sha256",
                "normalization",
                "sklearn_version",
                "evaluation",
            )
        },
        "outfits": outfits,
        "all_scored_combinations": all_outfits,
        "product_catalog": catalog,
        "excluded_items": exclusions,
        "projection_issues": projection_issues,
        "summary": {
            "pool_per_category": counts,
            "input_owned_count": len(bundle.get("owned_items", [])),
            "eligible_owned_count": sum(
                i["owned"] for i in catalog.values() if i["category"] in pools
            ),
            "all_combination_count": len(all_outfits),
            "student5_ready_count": len(outfits),
        },
        "projection_policy": {
            "owned_price": "Zero added purchase cost; original unknown price retained in catalog",
            "owned_availability": "Ignored for owned items; raw UNKNOWN retained in catalog",
            "unverified_purchases": "Scored but not Student5-ready; see explicit projection issues",
            "currency": "USD prices and zero owned purchase costs; no FX conversion",
        },
        "warnings": manifest.get("evaluation", {}).get("limitations", []),
    }
