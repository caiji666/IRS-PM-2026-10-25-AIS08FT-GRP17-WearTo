from typing import Any, Dict, List, Literal

from pydantic import Field, model_validator

from ..schemas import Availability, Contract, Garment, SourceType


class Provenance(Contract):
    source: str = Field(min_length=1)
    record_index: int = Field(ge=1)
    original: Dict[str, Any]


class NormalizedProduct(Garment):
    colors: List[str] = Field(default_factory=list)
    provenance: Provenance
    # This indicates metadata eligibility only, not a successfully encoded image.
    image_embedding_candidate: bool = False

    @model_validator(mode="after")
    def minimum_requirements(self):
        if not self.product_id.strip() or not self.title.strip():
            raise ValueError("text retrieval requires nonempty product_id and title")
        if self.availability != Availability.UNKNOWN and not self.evidence.availability_verified:
            raise ValueError("known stock requires verified availability evidence")
        if self.image_embedding_candidate and (
            not self.image_path
            or not self.evidence.image_verified
            or self.source_type == SourceType.PREPARED_CATALOGUE
        ):
            raise ValueError("image candidate requires verified non-prepared image metadata")
        return self


class ProcessingIssue(Contract):
    record_index: int = Field(ge=1)
    stage: Literal["M2", "M3"] = "M3"
    reason: str


class ProcessingReport(Contract):
    schema_version: Literal["m3.1"] = "m3.1"
    products: List[NormalizedProduct] = Field(default_factory=list)
    issues: List[ProcessingIssue] = Field(default_factory=list)
    input_count: int = Field(ge=0)
    warnings: List[str] = Field(
        default_factory=lambda: [
            "Prepared catalogue is illustrative, not live purchasable inventory.",
            "No currency conversion, image inspection, embedding or ranking performed.",
        ]
    )
