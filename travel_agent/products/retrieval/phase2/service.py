import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ..m4.adapter import wardrobe_index
from ..m4.wardrobe import WardrobeAnalyzer, WardrobeIssue, predict
from ..m5.images import load_image
from ..schemas import Category, Embedding, WardrobeUpload
from .models import Phase2Bundle, UnifiedItem, WardrobeStore

SUBCATEGORIES = {
    Category.TOP: ["t-shirt", "shirt", "blouse", "sweater", "hoodie", "tank top"],
    Category.BOTTOM: ["trousers", "jeans", "shorts", "skirt"],
    Category.FOOTWEAR: ["sneakers", "boots", "sandals", "dress shoes"],
    Category.OUTERWEAR: ["jacket", "coat", "blazer", "cardigan"],
}
STYLES = ["casual", "formal", "sporty", "minimalist", "streetwear", "other"]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".wardrobe-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(value.model_dump_json(indent=2) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_store(path):
    path = Path(path)
    return (
        WardrobeStore.model_validate_json(path.read_text(encoding="utf-8"))
        if path.exists()
        else WardrobeStore()
    )


def classify(vector, labels, encoder, minimum, margin):
    prompts = [f"a product photo of {label} clothing" for label in labels]
    return predict(vector, labels, encoder.encode_texts(prompts), minimum, margin)


def ingest(paths, encoder, store_path, image_dir, minimum=0.20, margin=0.03):
    """Local single-writer store. Validate existing store before importing any files."""
    store = load_store(store_path)
    if any(
        i.garment_embedding.model != encoder.identity
        or len(i.garment_embedding.values) != encoder.dimension
        for i in store.items
    ):
        raise ValueError(
            "stored wardrobe embeddings incompatible; use a new store or re-ingest with original model"
        )
    image_dir = Path(image_dir).resolve()
    image_dir.mkdir(parents=True, exist_ok=True)
    uploads = []
    origins = {}
    issues = []
    seen = set()
    for raw in paths:
        try:
            path = Path(raw).expanduser().resolve()
            image = load_image(path, path.parent)
            digest = hashlib.sha256(
                f"{image.width}x{image.height}:RGB:".encode() + image.tobytes()
            ).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            managed = image_dir / (digest + ".png")
            # Replace from decoded pixels so EXIF, orientation and color normalization are stable.
            fd, tmp = tempfile.mkstemp(dir=image_dir, suffix=".png")
            os.close(fd)
            try:
                image.save(tmp, format="PNG")
                os.replace(tmp, managed)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            uploads.append(WardrobeUpload(upload_id=digest, image_path=managed.name))
            origins[digest] = {
                "original_path": str(path),
                "sha256_rgb": digest,
                "width": image.width,
                "height": image.height,
                "imported_at": datetime.now(timezone.utc).isoformat(),
                "minimum_similarity": minimum,
                "minimum_margin": margin,
                "preprocessing": "EXIF transpose, RGB, metadata-free PNG; CLIPProcessor resize/crop/normalize during encoding",
            }
        except (OSError, ValueError) as exc:
            issues.append(WardrobeIssue(upload_id=str(raw), reason=str(exc)))
    report = WardrobeAnalyzer(encoder, image_dir, minimum, margin).analyze(uploads)
    index = wardrobe_index(report, encoder)
    normalized = {i.product.product_id: i.product for i in index.items}
    updated = {i.product_id: i for i in store.items}
    for result in report.products:
        p = normalized[result.garment.product_id]
        predictions = {"category": result.category_prediction, "color": result.color_prediction}
        subcategory = "UNKNOWN"
        styles = []
        labels = SUBCATEGORIES.get(p.category)
        if labels:
            sub = classify(result.vector, labels + ["other"], encoder, minimum, margin)
            predictions["subcategory"] = sub
            if sub.accepted and sub.label != "other":
                subcategory = sub.label
        style = classify(result.vector, STYLES, encoder, minimum, margin)
        predictions["style"] = style
        if p.category != Category.UNKNOWN and style.accepted and style.label != "other":
            styles = [style.label]
        data = p.model_dump(mode="json")
        data["image_path"] = str(image_dir / result.upload.image_path)
        data["provenance"]["original"]["phase2_upload"] = origins[result.upload.upload_id]
        updated[p.product_id] = UnifiedItem(
            **data,
            subcategory=subcategory,
            styles=styles,
            style_status="PREDICTED" if styles else "UNKNOWN",
            owned=True,
            ownership_source="user_upload",
            additional_purchase_required=False,
            garment_embedding=result.vector,
            attribute_predictions=predictions,
        )
    issues.extend(report.issues)
    issues.extend(
        WardrobeIssue(upload_id=i.product_id or "normalization", reason=i.reason)
        for i in index.issues
    )
    store = WardrobeStore(items=list(updated.values()))
    atomic_json(store_path, store)
    return store, issues


def combine(store, ranking, model, dimension, issues=None):
    for item in store.items:
        if item.garment_embedding.model != model or len(item.garment_embedding.values) != dimension:
            raise ValueError("owned/product embedding compatibility mismatch")
    groups = {}
    for category, results in ranking.ranked_products.items():
        groups[category.value] = []
        for result in results:
            item = result.item
            if item.vector.model != model or len(item.vector.values) != dimension:
                raise ValueError("product embedding compatibility mismatch")
            data = item.product.model_dump(mode="json")
            data["embedding"] = Embedding(
                model=item.vector.model, values=item.vector.values
            ).model_dump()
            groups[category.value].append(UnifiedItem(**data, garment_embedding=item.vector))
    return Phase2Bundle(
        owned_items=store.items, top_k_products=groups, product_ranking=ranking, issues=issues or []
    )


def to_m1_garment(item):
    """Explicit projection into unchanged strict M1; new attributes remain in Phase 2."""
    from ..schemas import Garment

    data = item.model_dump(mode="json")
    return Garment.model_validate({k: data[k] for k in Garment.model_fields})
