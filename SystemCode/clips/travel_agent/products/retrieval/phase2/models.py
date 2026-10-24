from typing import Dict, List, Literal, Optional

from pydantic import Field, model_validator

from ..m3.models import NormalizedProduct
from ..m4.wardrobe import Prediction, WardrobeIssue
from ..m5.models import Vector
from ..m6.models import GroupedRankingReport
from ..schemas import Availability, Contract, SourceType


class UnifiedItem(NormalizedProduct):
    subcategory: str = "UNKNOWN"
    styles: List[str] = Field(default_factory=list)
    style_status: Literal["PREDICTED", "UNKNOWN"] = "UNKNOWN"
    owned: bool = False
    ownership_source: Optional[Literal["user_upload"]] = None
    additional_purchase_required: bool = True
    garment_embedding: Vector
    attribute_predictions: Dict[str, Prediction] = Field(default_factory=dict)

    @model_validator(mode="after")
    def coherent(self):
        if self.owned != (self.source_type == SourceType.WARDROBE):
            raise ValueError("owned must match WARDROBE source")
        if self.additional_purchase_required == self.owned:
            raise ValueError("purchase requirement must match ownership")
        if self.owned:
            if self.ownership_source != "user_upload" or self.garment_embedding.modality != "image":
                raise ValueError("owned item requires user-upload ownership and image embedding")
            if (
                self.price is not None
                or self.product_url is not None
                or self.availability != Availability.UNKNOWN
            ):
                raise ValueError(
                    "uploaded wardrobe must not assert price, purchase URL or merchant stock"
                )
        elif self.ownership_source is not None:
            raise ValueError("non-owned item cannot assert ownership")
        if (
            self.embedding is None
            or self.embedding.model != self.garment_embedding.model
            or self.embedding.values != self.garment_embedding.values
        ):
            raise ValueError("M1 embedding and garment embedding must match")
        if (self.style_status == "UNKNOWN") != (not self.styles):
            raise ValueError("style status inconsistent")
        return self


class WardrobeStore(Contract):
    schema_version: Literal["wardrobe.1"] = "wardrobe.1"
    items: List[UnifiedItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_owned(self):
        if any(not i.owned for i in self.items) or len({i.product_id for i in self.items}) != len(
            self.items
        ):
            raise ValueError("wardrobe store requires unique owned items")
        return self


class Phase2Bundle(Contract):
    schema_version: Literal["phase2.1"] = "phase2.1"
    owned_items: List[UnifiedItem]
    top_k_products: Dict[str, List[UnifiedItem]]
    product_ranking: GroupedRankingReport
    issues: List[WardrobeIssue] = Field(default_factory=list)
    warnings: List[str] = Field(
        default_factory=lambda: [
            "Uploaded ownership is user-declared, not independently verified.",
            "Category/subcategory/color/style predictions are uncalibrated; UNKNOWN is retained.",
            "Prepared products are fictional and use text embeddings; no live shopping or outfit compatibility is established.",
        ]
    )

    @model_validator(mode="after")
    def separation(self):
        if any(not i.owned for i in self.owned_items):
            raise ValueError("owned pool contains unowned item")
        if set(self.top_k_products) != {k.value for k in self.product_ranking.ranked_products}:
            raise ValueError("ranking group keys disagree")
        for category, items in self.top_k_products.items():
            expected = self.product_ranking.ranked_products[category]
            if [i.product_id for i in items] != [r.item.product.product_id for r in expected]:
                raise ValueError("unified items must preserve ranked selection and order")
            if any(i.owned or i.category.value != category for i in items):
                raise ValueError("invalid product group")
        return self
