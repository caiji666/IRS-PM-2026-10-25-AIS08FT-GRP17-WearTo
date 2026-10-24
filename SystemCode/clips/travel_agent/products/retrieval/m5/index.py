from .images import load_image
from .models import IndexedProduct, IndexIssue, ProductIndex


def caption(product):
    return ". ".join(
        x
        for x in [
            product.title,
            product.description,
            product.raw_category,
            ", ".join(product.colors),
        ]
        if x
    )


def build_index(products, encoder, image_root):
    """Text fallback is explicit; synthetic prepared images are never encoded."""
    products = list(products)
    report = ProductIndex(model=encoder.identity, dimension=encoder.dimension)
    pending, texts = [], []
    for product in products:
        if product.image_embedding_candidate:
            try:
                image = load_image(product.image_path, image_root)
                embedding = encoder.encode_images([image])[0]
                report.items.append(
                    IndexedProduct(
                        product=product,
                        vector=embedding,
                        encoding_reason="verified-image-metadata; raster decoded, semantics unverified",
                    )
                )
                continue
            except (OSError, ValueError) as exc:
                report.issues.append(
                    IndexIssue(
                        product_id=product.product_id, reason=f"image failed; text fallback: {exc}"
                    )
                )
        pending.append(product)
        texts.append(caption(product))
    embeddings = encoder.encode_texts(texts) if texts else []
    if len(embeddings) != len(pending):
        raise ValueError("encoder returned incorrect batch count")
    for product, embedding in zip(pending, embeddings):
        report.items.append(
            IndexedProduct(
                product=product,
                vector=embedding,
                encoding_reason="text fallback; prepared/unverified/missing/invalid image",
            )
        )
    # Restore input order; M6 handles sorting.
    order = {(p.source_type, p.product_id): i for i, p in enumerate(products)}
    report.items.sort(key=lambda x: order[(x.product.source_type, x.product.product_id)])
    return ProductIndex.model_validate(report.model_dump())
