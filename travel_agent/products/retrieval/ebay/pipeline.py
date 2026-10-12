import copy
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from ..m3 import process_records
from ..m3.models import Provenance
from ..m5 import build_index
from ..m5.images import load_image
from ..m6 import RankedProductProvider
from ..phase2.service import combine, load_store
from ..schemas import Embedding, ProductRetrievalRequest
from ..upstream import adapt_files
from .client import EbayError, download_photo

ALIASES = {
    "TOP": {"shirts", "t-shirts", "t shirts", "tops", "sweaters", "blouses", "polos"},
    "BOTTOM": {"jeans", "pants", "trousers", "shorts", "skirts"},
    "FOOTWEAR": {
        "shoes",
        "boots",
        "sandals",
        "sneakers",
        "athletic shoes",
        "heels",
        "flats",
        "casual shoes",
        "dress shoes",
    },
    "OUTERWEAR": {
        "coats, jackets & vests",
        "coats & jackets",
        "jackets",
        "coats",
        "blazers & suits",
        "blazers",
    },
}


def category_names(row):
    names = [r.get("categoryName", "") for r in row.get("categories", []) if isinstance(r, dict)]
    names += str(row.get("categoryPath", "")).split("|")
    # Select most specific known leaf, never derive category solely from search query.
    for name in reversed(names):
        for cat, aliases in ALIASES.items():
            if name.strip().casefold() in aliases:
                return cat
    return "UNKNOWN"


def map_product(row, environment, marketplace, fetched_at, image_path=None):
    item_id = row.get("itemId")
    title = row.get("title")
    if not isinstance(item_id, str) or not item_id or not isinstance(title, str) or not title:
        raise ValueError("item requires itemId/title")
    price = row.get("price")
    mapped_price = None
    if price is not None:
        if not isinstance(price, dict) or isinstance(price.get("value"), bool):
            raise ValueError("invalid API price")
        mapped_price = {
            "amount": float(price["value"]),
            "currency": price["currency"],
            "is_synthetic": environment != "production",
        }
    aspects = {
        a.get("name", "").casefold(): a.get("value", "")
        for a in row.get("localizedAspects", [])
        if isinstance(a, dict)
    }
    data = {
        "product_id": f"ebay:{environment}:{marketplace}:{item_id}",
        "title": title,
        "raw_category": category_names(row),
        "description": row.get("shortDescription") or "",
        "source_type": "ECOMMERCE_API",
        "price": mapped_price,
        "availability": "UNKNOWN",
        "sizes": [aspects["size"]] if aspects.get("size") else [],
        "colors": [aspects["color"]] if aspects.get("color") else [],
        "image_path": str(image_path) if image_path else None,
        "product_url": row.get("itemWebUrl"),
        "evidence": {
            "price_verified": environment == "production" and price is not None,
            "image_verified": image_path is not None,
            "notes": [
                f"eBay {environment} API snapshot at {fetched_at}; stock not verified; price excludes shipping/taxes.",
                "Raster downloaded and decoded; image_verified means local image readiness, not semantic authenticity.",
            ],
        },
    }
    report = process_records([data], source=f"ebay:{environment}:{marketplace}")
    report.warnings = ["API snapshot, not guaranteed live stock; no currency conversion."]
    if not report.products:
        raise ValueError("API product validation failed")
    product = report.products[0]
    product.provenance = Provenance(
        source=f"ebay:{environment}:{marketplace}",
        record_index=1,
        original={
            "api_item": row,
            "fetched_at": fetched_at,
            "marketplace": marketplace,
            "environment": environment,
        },
    )
    return product


def run(
    client,
    encoder,
    pt,
    text,
    context,
    output_dir,
    model_info=None,
    reference_mode="recompute",
    max_references=3,
    limit=20,
    max_products=40,
    k=5,
    wardrobe_store="runs/wardrobe/items.json",
    category_ids=None,
    fetch_photo=download_photo,
    request_override=None,
):
    if reference_mode not in {"recompute", "imported"}:
        raise ValueError("invalid reference mode")
    if not 1 <= max_references <= 10 or not 1 <= max_products <= 200:
        raise ValueError("invalid reference/product limit")
    req, fallback, upstream = adapt_files(pt, text, context, model_info)
    if request_override is not None:
        override = ProductRetrievalRequest.model_validate(request_override)
        req.context = override.context
        req.query = override.query
        req.categories = override.categories
        req.size = override.size
        req.max_price = override.max_price
        req.currency = override.currency
        req.limit = override.limit
        req.context_embedding = None
        upstream["expanded_fashion_context"] = (
            override.context.query or override.context.trip.destination
        )
        upstream["request_override"] = override.model_dump(mode="json")
    if reference_mode == "imported" and upstream["status"] != "declared_compatible":
        raise ValueError(
            "confirmed compatible upstream model metadata required for imported vectors"
        )
    if reference_mode == "recompute" or request_override is not None:
        req.context_embedding = Embedding(
            model=encoder.identity,
            values=encoder.encode_texts([upstream["expanded_fashion_context"]])[0].values,
        )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    issues = []
    raw_search = []
    references = []
    candidates = {}
    fetched = datetime.now(timezone.utc).isoformat()
    client.authenticate()
    for i, (reference, meta) in enumerate(zip(req.top_social_references, upstream["top_images"])):
        if i >= max_references:
            break
        try:
            url = meta.get("image_url")
            path = output_dir / "reference_images" / f"{i:03d}.jpg"
            image_bytes = fetch_photo(url, path, {"images.unsplash.com"})
            ref = reference.model_copy(deep=True)
            ref.image_path = str(path)
            if reference_mode == "recompute":
                vec = encoder.encode_images([load_image(path, output_dir)])[0]
                ref.embedding = Embedding(model=vec.model, values=vec.values)
            references.append(ref)
            response = client.search_image(image_bytes, limit, category_ids)
            raw_search.append(
                {"reference_id": ref.reference_id, "fetched_at": fetched, "response": response}
            )
            for row in response.get("itemSummaries", []):
                if isinstance(row, dict) and row.get("itemId"):
                    key = row["itemId"]
                    if key not in candidates and len(candidates) < max_products:
                        candidates[key] = {"row": row, "references": [ref.reference_id]}
                    elif key in candidates:
                        candidates[key]["references"].append(ref.reference_id)
        except (ValueError, OSError, EbayError) as exc:
            issues.append(
                {
                    "stage": "reference_download_or_search",
                    "reference_id": reference.reference_id,
                    "reason": str(exc),
                }
            )
    products = []
    for item_id, entry in candidates.items():
        try:
            row = copy.deepcopy(entry["row"])
            try:
                row.update(client.item(item_id))
            except EbayError as exc:
                issues.append({"stage": "item_details", "item_id": item_id, "reason": str(exc)})
            url = (row.get("image") or {}).get("imageUrl")
            if not url:
                raise ValueError("item has no primary photo; excluded from image ranking")
            path = (
                output_dir
                / "product_images"
                / (hashlib.sha256(item_id.encode()).hexdigest() + ".jpg")
            )
            fetch_photo(url, path, {"i.ebayimg.com"})
            product = map_product(row, client.environment, client.marketplace, fetched, path)
            product.provenance.original["matched_reference_ids"] = entry["references"]
            product.provenance.original["downloaded_image_sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            products.append(product)
        except (ValueError, OSError, EbayError, KeyError) as exc:
            issues.append({"stage": "product", "item_id": item_id, "reason": str(exc)})
    # Stock remains UNKNOWN; failed images never masquerade as successful visual matches.
    index = build_index(products, encoder, output_dir)
    req.top_social_references = references
    ranking = RankedProductProvider(index, encoder, output_dir).retrieve_grouped(req, k)
    warnings = [
        f"eBay {client.environment} snapshots; no verified stock, FX conversion or shipping/tax totals.",
        "Sandbox listings are test data, not real purchasing opportunities.",
        "Local research reranking; production display/use must meet applicable eBay terms.",
    ]
    if client.environment == "production":
        warnings.remove("Sandbox listings are test data, not real purchasing opportunities.")
    ranking.warnings = [w for w in ranking.warnings if "Prepared records" not in w] + warnings
    bundle = combine(load_store(wardrobe_store), ranking, encoder.identity, encoder.dimension)
    bundle.warnings = warnings + [
        "Owned items come from the saved user wardrobe; social reference photos are not imported as owned clothes."
    ]
    return {
        "schema_version": "ebay.pipeline.1",
        "environment": client.environment,
        "marketplace": client.marketplace,
        "status": "completed" if products else "no_usable_api_products",
        "fetched_at": fetched,
        "reference_mode": reference_mode,
        "upstream_metadata": upstream,
        "raw_search_responses": raw_search,
        "request_used": req.model_dump(mode="json"),
        "index": index.model_dump(mode="json"),
        "bundle": bundle.model_dump(mode="json"),
        "issues": issues,
        "summary": {
            "references_used": len(references),
            "unique_candidates": len(candidates),
            "products_indexed": len(index.items),
            "image_vectors": sum(i.vector.modality == "image" for i in index.items),
            "returned_per_category": {k.value: len(v) for k, v in ranking.ranked_products.items()},
        },
    }
