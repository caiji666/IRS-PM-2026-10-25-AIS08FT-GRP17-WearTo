from datetime import date
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Category(str, Enum):
    TOP = "TOP"
    BOTTOM = "BOTTOM"
    FOOTWEAR = "FOOTWEAR"
    OUTERWEAR = "OUTERWEAR"
    UNKNOWN = "UNKNOWN"


class SourceType(str, Enum):
    PREPARED_CATALOGUE = "PREPARED_CATALOGUE"
    WARDROBE = "WARDROBE"
    ECOMMERCE_API = "ECOMMERCE_API"


class Availability(str, Enum):
    UNKNOWN = "UNKNOWN"
    IN_STOCK = "IN_STOCK"
    OUT_OF_STOCK = "OUT_OF_STOCK"


class Evidence(Contract):
    price_verified: bool = False
    availability_verified: bool = False
    sizes_verified: bool = False
    image_verified: bool = False
    product_url_verified: bool = False
    category_verified: bool = False
    notes: List[str] = Field(default_factory=list)


class Price(Contract):
    amount: float = Field(ge=0, allow_inf_nan=False)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    is_synthetic: bool = True


class Embedding(Contract):
    model: str = Field(min_length=1)
    values: List[float] = Field(min_length=1)

    # Transport only; no embedding generation or comparison in M1/M2.
    @model_validator(mode="after")
    def finite(self):
        import math

        if not all(math.isfinite(v) for v in self.values):
            raise ValueError("embedding must contain finite numbers")
        return self


class Trip(Contract):
    destination: str = Field(min_length=1)
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    @model_validator(mode="after")
    def dates(self):
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date precedes start_date")
        return self


class ExpandedFashionContext(Contract):
    trip: Trip
    query: Optional[str] = None
    activities: List[str] = Field(default_factory=list)
    style_preferences: List[str] = Field(default_factory=list)
    weather_notes: Optional[str] = None


class SocialReference(Contract):
    reference_id: str = Field(min_length=1)
    image_path: Optional[str] = None
    url: Optional[HttpUrl] = None
    description: Optional[str] = None
    embedding: Optional[Embedding] = None


class WardrobeUpload(Contract):
    upload_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    # Opaque upload metadata. Image contents are not read or understood.


class RawProduct(Contract):
    product_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    raw_category: str = Field(min_length=1)
    description: str = ""
    source_type: SourceType = SourceType.PREPARED_CATALOGUE
    price: Optional[Price] = None
    sizes: List[str] = Field(default_factory=list)
    availability: Availability = Availability.UNKNOWN
    image_path: Optional[str] = None
    product_url: Optional[HttpUrl] = None
    embedding: Optional[Embedding] = None
    evidence: Evidence = Field(default_factory=Evidence)

    @model_validator(mode="after")
    def prepared_honesty(self):
        if self.source_type == SourceType.PREPARED_CATALOGUE:
            if self.availability != Availability.UNKNOWN or self.evidence.availability_verified:
                raise ValueError("prepared catalogue cannot assert live inventory")
            if self.price and (not self.price.is_synthetic or self.evidence.price_verified):
                raise ValueError("prepared price must be synthetic and unverified")
            if self.product_url or self.evidence.product_url_verified:
                raise ValueError("prepared samples cannot assert purchase URLs")
        return self


class Garment(RawProduct):
    category: Category


class ProductRetrievalRequest(Contract):
    schema_version: Literal["1.0"] = "1.0"
    context: ExpandedFashionContext
    context_embedding: Optional[Embedding] = None
    top_social_references: List[SocialReference] = Field(default_factory=list)
    wardrobe_uploads: List[WardrobeUpload] = Field(default_factory=list)
    # Already structured user-provided records only; no image analysis.
    wardrobe_garments: List[Garment] = Field(default_factory=list)
    query: Optional[str] = None
    categories: List[Category] = Field(default_factory=list)
    size: Optional[str] = None
    max_price: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    currency: Optional[str] = Field(default=None, pattern=r"^[A-Z]{3}$")
    limit: int = Field(default=20, ge=1, le=1000)

    @model_validator(mode="after")
    def filters(self):
        if self.max_price is not None and self.currency is None:
            raise ValueError("max_price requires currency; no FX conversion")
        if any(g.source_type != SourceType.WARDROBE for g in self.wardrobe_garments):
            raise ValueError("wardrobe_garments must have WARDROBE source")
        return self


class RecordIssue(Contract):
    record_index: int
    product_id: Optional[str] = None
    reason: str


class ProductRetrievalResponse(Contract):
    schema_version: Literal["1.0"] = "1.0"
    products: List[Garment]
    wardrobe_garments: List[Garment] = Field(default_factory=list)
    pending_wardrobe_uploads: List[WardrobeUpload] = Field(default_factory=list)
    total_matches: int
    issues: List[RecordIssue] = Field(default_factory=list)
    warnings: List[str] = Field(
        default_factory=lambda: [
            "Prepared samples are illustrative; prices, sizes and stock are not live verified.",
            "Uploads are passed through without image understanding; embeddings are transport only.",
        ]
    )
