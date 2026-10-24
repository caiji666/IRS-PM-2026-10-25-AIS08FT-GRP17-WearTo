from typing import Dict, List, Literal, Optional

from pydantic import Field, model_validator

from ..m5.models import IndexedProduct, Vector
from ..schemas import Availability, Category, Contract, SourceType


class RankingConfig(Contract):
    context_weight: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
    social_weight: float = Field(default=0.3, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def nonzero(self):
        if self.context_weight + self.social_weight <= 0:
            raise ValueError("at least one weight must be positive")
        return self


class RankingRequest(Contract):
    context: Optional[Vector] = None
    social_references: List[Vector] = Field(default_factory=list)
    config: RankingConfig = Field(default_factory=RankingConfig)
    categories: List[Category] = Field(default_factory=list)
    source_types: List[SourceType] = Field(default_factory=list)
    size: Optional[str] = None
    currency: Optional[str] = Field(default=None, pattern=r"^[A-Z]{3}$")
    max_price: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    availability: Optional[Availability] = None
    limit: int = Field(default=20, ge=1, le=1000)

    @model_validator(mode="after")
    def filters(self):
        if self.max_price is not None and self.currency is None:
            raise ValueError("max_price requires currency; no FX conversion")
        active = (self.config.context_weight if self.context is not None else 0) + (
            self.config.social_weight if self.social_references else 0
        )
        if active <= 0:
            raise ValueError("no weighted context/social signal available")
        return self


class RankedProduct(Contract):
    item: IndexedProduct
    score: float = Field(ge=-1, le=1, allow_inf_nan=False)
    context_similarity: Optional[float] = None
    social_similarity: Optional[float] = None
    context_weight: float
    social_weight: float


class RankingReport(Contract):
    schema_version: Literal["m6.2"] = "m6.2"
    social_aggregation: Literal["max"] = "max"
    products: List[RankedProduct] = Field(default_factory=list)
    total_matches: int = Field(ge=0)
    warnings: List[str] = Field(
        default_factory=lambda: [
            "Similarity is semantic relevance, not outfit compatibility or purchasing verification.",
            "Prepared records and synthetic prices are illustrative; stock is not live verified.",
        ]
    )


class GroupedRankingRequest(RankingRequest):
    # Global limit does not cap category results.
    k_per_category: int = Field(default=5, ge=1, le=1000)


class GroupedRankingReport(Contract):
    schema_version: Literal["m6.grouped.1"] = "m6.grouped.1"
    social_aggregation: Literal["max"] = "max"
    k_per_category: int = Field(ge=1, le=1000)
    ranked_products: Dict[Category, List[RankedProduct]]
    total_matches_per_category: Dict[Category, int]
    total_matches: int = Field(ge=0)
    excluded_unknown_matches: int = Field(default=0, ge=0)
    warnings: List[str] = Field(
        default_factory=lambda: [
            "Similarity is semantic relevance, not outfit compatibility or purchasing verification.",
            "Prepared records and synthetic prices are illustrative; stock is not live verified.",
        ]
    )

    @model_validator(mode="after")
    def consistent_groups(self):
        if set(self.ranked_products) != set(self.total_matches_per_category):
            raise ValueError("category groups and counts must have identical keys")
        if self.total_matches != sum(self.total_matches_per_category.values()):
            raise ValueError("total_matches must equal the sum of category counts")
        for category, results in self.ranked_products.items():
            count = self.total_matches_per_category[category]
            if count < 0 or len(results) != min(self.k_per_category, count):
                raise ValueError("category result count must be min(K, available)")
            if any(result.item.product.category != category for result in results):
                raise ValueError("product in wrong category group")
            identities = [(r.item.product.source_type, r.item.product.product_id) for r in results]
            if len(identities) != len(set(identities)):
                raise ValueError("duplicate product in category group")
        return self
