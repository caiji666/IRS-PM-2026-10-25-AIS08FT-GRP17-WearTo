"""Carry actual M4 image vectors into M3-normalized wardrobe records and an M5 index."""

from ..m3 import process_records
from ..m5.models import IndexedProduct, IndexIssue, ProductIndex


def wardrobe_index(report, encoder):
    index = ProductIndex(model=encoder.identity, dimension=encoder.dimension)
    for result in report.products:
        data = result.garment.model_dump(mode="json")
        data["colors"] = result.predicted_colors
        normalized = process_records([data], source=f"wardrobe-upload:{result.upload.upload_id}")
        if not normalized.products:
            index.issues.extend(
                IndexIssue(product_id=result.garment.product_id, reason=i.reason)
                for i in normalized.issues
            )
            continue
        index.items.append(
            IndexedProduct(
                product=normalized.products[0],
                vector=result.vector,
                encoding_reason="M4 local uploaded image decoded and encoded; category/color predictions unverified",
            )
        )
    index.issues.extend(
        IndexIssue(reason=f"upload {i.upload_id}: {i.reason}") for i in report.issues
    )
    return ProductIndex.model_validate(index.model_dump())
