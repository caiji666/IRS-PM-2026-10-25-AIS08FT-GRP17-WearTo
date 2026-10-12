"""Single-garment zero-shot image understanding with explicit abstention."""

from typing import Dict, List

from pydantic import Field

from ..m5.images import load_image
from ..m5.models import Vector, cosine
from ..schemas import Category, Contract, Embedding, Evidence, Garment, SourceType, WardrobeUpload

CATEGORY_PROMPTS = {
    "TOP": "a product photo of a shirt, t-shirt, blouse or sweater",
    "BOTTOM": "a product photo of pants, jeans, shorts or a skirt",
    "FOOTWEAR": "a product photo of shoes, sneakers, boots or sandals",
    "OUTERWEAR": "a product photo of a jacket, coat or blazer",
    "DRESS": "a product photo of a dress or jumpsuit",
    "ACCESSORY": "a product photo of a bag, scarf, hat or jewelry",
    "OTHER": "a photo of a scene without a single clothing item",
}
COLORS = [
    "black",
    "white",
    "gray",
    "red",
    "blue",
    "green",
    "yellow",
    "brown",
    "beige",
    "pink",
    "purple",
    "orange",
    "multicolored",
]


class Prediction(Contract):
    label: str
    top_similarity: float
    margin: float
    accepted: bool
    similarities: Dict[str, float]
    method: str = "FashionCLIP zero-shot cosine; uncalibrated heuristic thresholds"


class WardrobeResult(Contract):
    upload: WardrobeUpload
    garment: Garment
    vector: Vector
    category_prediction: Prediction
    color_prediction: Prediction
    predicted_colors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(
        default_factory=lambda: [
            "Single garment per photo assumed; no segmentation or object detection.",
            "Predicted category/color are unverified; size, fabric, inventory and ownership are not inferred.",
        ]
    )


class WardrobeIssue(Contract):
    upload_id: str
    reason: str


class WardrobeReport(Contract):
    products: List[WardrobeResult] = Field(default_factory=list)
    issues: List[WardrobeIssue] = Field(default_factory=list)


def predict(image_vector, labels, vectors, minimum, margin):
    pairs = [(label, cosine(image_vector, v)) for label, v in zip(labels, vectors)]
    pairs.sort(key=lambda x: (-x[1], x[0]))
    gap = pairs[0][1] - pairs[1][1]
    return Prediction(
        label=pairs[0][0],
        top_similarity=pairs[0][1],
        margin=gap,
        accepted=pairs[0][1] >= minimum and gap >= margin,
        similarities=dict(pairs),
    )


class WardrobeAnalyzer:
    def __init__(self, encoder, image_root, minimum_similarity=0.20, minimum_margin=0.03):
        if not -1 <= minimum_similarity <= 1 or not 0 <= minimum_margin <= 2:
            raise ValueError("invalid abstention thresholds")
        self.encoder, self.image_root = encoder, image_root
        self.minimum, self.margin = minimum_similarity, minimum_margin

    def analyze(self, uploads):
        report = WardrobeReport()
        if not uploads:
            return report
        category_vectors = self.encoder.encode_texts(list(CATEGORY_PROMPTS.values()))
        color_vectors = self.encoder.encode_texts(
            [f"a product photo of a {c} garment" for c in COLORS]
        )
        seen = set()
        for upload in uploads:
            try:
                if upload.upload_id in seen:
                    raise ValueError("duplicate upload_id; first valid upload wins")
                image = load_image(upload.image_path, self.image_root)
                image_vector = self.encoder.encode_images([image])[0]
                cat = predict(
                    image_vector,
                    list(CATEGORY_PROMPTS),
                    category_vectors,
                    self.minimum,
                    self.margin,
                )
                color = predict(image_vector, COLORS, color_vectors, self.minimum, self.margin)
                category = (
                    Category(cat.label)
                    if cat.accepted and cat.label in Category._value2member_map_
                    else Category.UNKNOWN
                )
                garment = Garment(
                    product_id=f"wardrobe:{upload.upload_id}",
                    title=f"User uploaded garment {upload.upload_id}",
                    raw_category=category.value,
                    category=category,
                    source_type=SourceType.WARDROBE,
                    image_path=upload.image_path,
                    embedding=Embedding(model=image_vector.model, values=image_vector.values),
                    evidence=Evidence(
                        notes=[
                            "Image decoded and encoded locally; predicted attributes are unverified."
                        ]
                    ),
                )
                report.products.append(
                    WardrobeResult(
                        upload=upload,
                        garment=garment,
                        vector=image_vector,
                        category_prediction=cat,
                        color_prediction=color,
                        predicted_colors=[color.label] if color.accepted else [],
                    )
                )
                seen.add(upload.upload_id)
            except (OSError, ValueError) as exc:
                report.issues.append(WardrobeIssue(upload_id=upload.upload_id, reason=str(exc)))
        return report
