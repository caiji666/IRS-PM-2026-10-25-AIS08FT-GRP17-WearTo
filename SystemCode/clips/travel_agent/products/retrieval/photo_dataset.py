"""Opt-in real-photo prepared dataset smoke. No live ecommerce or old M5 changes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .m3 import process_catalogue
from .m5 import CachedEncoder, FashionCLIPEncoder
from .m5.images import load_image
from .m5.models import IndexedProduct, ProductIndex
from .m6 import RankedProductProvider
from .phase2.service import combine, ingest
from .schemas import ProductRetrievalRequest, SourceType


def photo_index(dataset, encoder):
    dataset = Path(dataset).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text())
    entries = {r["product_id"]: r for r in manifest["records"]}
    if len(entries) != len(manifest["records"]):
        raise ValueError("duplicate manifest ID")
    normalized = process_catalogue(dataset / "catalogue.json")
    if {p.product_id for p in normalized.products} != set(entries):
        raise ValueError("catalogue/manifest identities disagree")
    items = []
    paths = []
    for p in normalized.products:
        if p.source_type != SourceType.PREPARED_CATALOGUE:
            raise ValueError("photo dataset requires prepared source")
        r = entries[p.product_id]
        path = (dataset / r["image_file"]).resolve()
        try:
            path.relative_to(dataset / "images")
        except ValueError:
            raise ValueError("manifest image outside dataset images")
        if hashlib.sha256(path.read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError("image checksum mismatch: " + p.product_id)
        image = load_image(path, dataset)
        v = encoder.encode_images([image])[0]
        # Extend source provenance, never manufacture ecommerce/stock verification.
        data = p.model_dump(mode="json")
        data["image_path"] = str(path)
        data["provenance"]["original"]["photo_dataset_source"] = r
        from .m3.models import NormalizedProduct

        p = NormalizedProduct.model_validate(data)
        items.append(
            IndexedProduct(
                product=p,
                vector=v,
                encoding_reason="explicit offline real-photo dataset adapter; decoded raster, provenance retained; no merchant verification",
            )
        )
        paths.append(path)
    return (
        normalized,
        ProductIndex(model=encoder.identity, dimension=encoder.dimension, items=items),
        paths,
    )


def main():
    parser = argparse.ArgumentParser(description="20 real retail photos: local pipeline smoke")
    parser.add_argument("--dataset", type=Path, default=Path("data/photos20"))
    parser.add_argument("--model-cache", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--embedding-cache", type=Path, default=Path(".cache/embeddings"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--request", type=Path, default=Path("fixtures/phase1_request.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/photos20_demo.json"))
    parser.add_argument("--wardrobe-dir", type=Path, default=Path("runs/photos20_wardrobe"))
    args = parser.parse_args()
    try:
        encoder = CachedEncoder(
            FashionCLIPEncoder(args.model_cache, offline=args.offline), args.embedding_cache
        )
        normalized, index, paths = photo_index(args.dataset, encoder)
        request = ProductRetrievalRequest.model_validate_json(args.request.read_text())
        if request.wardrobe_uploads or request.wardrobe_garments:
            raise ValueError(
                "this smoke imports dataset photos as simulated wardrobe; use an empty wardrobe request"
            )
        ranking = RankedProductProvider(index, encoder, Path(".")).retrieve_grouped(request, 5)
        store, issues = ingest(
            paths, encoder, args.wardrobe_dir / "items.json", args.wardrobe_dir / "images"
        )
        bundle = combine(store, ranking, encoder.identity, encoder.dimension, issues)
        bundle.warnings.append(
            "All dataset photos are reused as SIMULATED owned uploads to test transport only; no real user ownership or independent retrieval evaluation."
        )
        from collections import Counter

        counts = Counter(i.category.value for i in store.items)
        result = {
            "schema_version": "photos20.smoke.1",
            "normalization": normalized.model_dump(mode="json"),
            "index": index.model_dump(mode="json"),
            "bundle": bundle.model_dump(mode="json"),
            "summary": {
                "products": len(index.items),
                "image_vectors": sum(i.vector.modality == "image" for i in index.items),
                "owned_items": len(store.items),
                "predicted_wardrobe_categories": dict(counts),
                "returned_per_category": {k: len(v) for k, v in bundle.top_k_products.items()},
                "wardrobe_issues": len(issues),
            },
            "warnings": [
                "20 examples with five per class test execution only; Top-5 returns all eligible class members, not ranking quality.",
                "Historical retail dataset photos; no live price, inventory or purchase URLs.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(result["summary"], ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
