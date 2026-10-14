"""Pydantic v2 data models for the Student 5 module.

Mirrors the group proposal data contract (Section 7.1, Section 5.3.4 and Appendix A).
Anything the proposal does not define is marked PROVISIONAL: it exists only for Student 5's
independent development and mock testing and is not the final team interface.
See docs/data_contract.md for the field-by-field provenance.

These models only validate and store data. Scores (OutfitProductRelevance, FinalOutfitScore)
are calculated in ranking_engine, and budget / size feasibility is decided in constraint_engine.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    StringConstraints,
    field_validator,
    model_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# A URL, temporary path or adapter-specific reference to an image. Strict, so raw image bytes are
# rejected instead of being decoded into a string; nothing is downloaded or opened.
ReferenceStr = Annotated[StrictStr, StringConstraints(strip_whitespace=True, min_length=1)]

# CompatibilityScore = sigmoid(MLP logit), so it lies in [0, 1].
CompatibilityScoreValue = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]

# ProductScore = alpha * cosine + (1 - alpha) * max cosine, with alpha in [0, 1], so it lies
# in [-1, 1]. OutfitProductRelevance (a mean of ProductScores) has the same range, and
# FinalOutfitScore = beta * [0, 1] + (1 - beta) * [-1, 1] lies in [-(1 - beta), 1] ⊆ [-1, 1].
CosineScoreValue = Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]

Money = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


class ContractModel(BaseModel):
    """Base for all data contract models: unknown fields are rejected, assignment is validated."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RecommendationIntent(StrEnum):
    """Stage 1 intent labels. Values are from the proposal; the class name is PROVISIONAL."""

    MODIFY_REQUIREMENTS = "MODIFY_REQUIREMENTS"
    CONFIRM_OUTFIT = "CONFIRM_OUTFIT"
    REQUEST_EXPLANATION = "REQUEST_EXPLANATION"


class SessionStatus(StrEnum):
    """Values of `session_status`.

    AWAITING_CONFIRMATION is from the proposal. CONFIRMED is PROVISIONAL: the proposal says the
    selection is saved and the loop ends, but does not name the status.
    """

    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    CONFIRMED = "CONFIRMED"  # PROVISIONAL


class ProductCategory(StrEnum):
    """Normalised garment categories (proposal Section 5.3.4)."""

    TOP = "TOP"
    BOTTOM = "BOTTOM"
    FOOTWEAR = "FOOTWEAR"
    OUTERWEAR = "OUTERWEAR"


REQUIRED_OUTFIT_ROLES = (ProductCategory.TOP, ProductCategory.BOTTOM, ProductCategory.FOOTWEAR)


class PreviewMode(StrEnum):
    """PROVISIONAL. Stage 14 presentation modes named after the proposal's description:
    portrait-based preview, product cards only, generic mannequin, or flat-lay."""

    PORTRAIT = "PORTRAIT"
    PRODUCT_CARDS = "PRODUCT_CARDS"
    MANNEQUIN = "MANNEQUIN"
    FLAT_LAY = "FLAT_LAY"


class PreviewStatus(StrEnum):
    """PROVISIONAL. Outcome of a Stage 14 preview attempt."""

    SUCCESS = "SUCCESS"
    FALLBACK = "FALLBACK"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class TripShoppingRequest(ContractModel):
    """User-provided trip and shopping facts (proposal Appendix A).

    Only facts stated by the user; weather, activities and style are never invented here.
    """

    destination: NonEmptyStr
    start_date: date
    end_date: date
    activities: list[NonEmptyStr] = Field(default_factory=list)
    budget: Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]
    currency: NonEmptyStr
    style: list[NonEmptyStr] = Field(default_factory=list)
    excluded_items: list[NonEmptyStr] = Field(default_factory=list)
    # PROVISIONAL: the proposal mentions size as a user constraint (UC01) but defines no field.
    # None means the user gave no size, so the size constraint is not enabled.
    user_size: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _check_dates(self) -> TripShoppingRequest:
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be earlier than start_date")
        return self


TRIP_REQUEST_FIELDS = frozenset(TripShoppingRequest.model_fields)
TRIP_REQUEST_LIST_FIELDS = frozenset({"activities", "style", "excluded_items"})


class TripShoppingRequestPatch(ContractModel):
    """Requirement change produced by Stage 1 (proposal Appendix A).

    Only the structure is validated here; merging into TripShoppingRequest happens later in
    patch_processor. The key/intent rules below are PROVISIONAL.
    """

    intent: RecommendationIntent
    set: dict[str, Any] = Field(default_factory=dict)
    add: dict[str, list[Any]] = Field(default_factory=dict)
    remove: dict[str, Any] = Field(default_factory=dict)

    @field_validator("set", "add", "remove")
    @classmethod
    def _keys_are_request_fields(cls, value: dict[str, Any], info) -> dict[str, Any]:
        unknown = sorted(k for k in value if k not in TRIP_REQUEST_FIELDS)
        if unknown:
            raise ValueError(
                f"unknown TripShoppingRequest field(s) in '{info.field_name}': {unknown}"
            )
        if info.field_name == "add":
            not_lists = sorted(k for k in value if k not in TRIP_REQUEST_LIST_FIELDS)
            if not_lists:
                raise ValueError(f"'add' only applies to list fields, got: {not_lists}")
        return value

    @model_validator(mode="after")
    def _operations_disjoint(self) -> TripShoppingRequestPatch:
        # A field may appear in at most one of set/add/remove, so applying the patch never
        # depends on operation order.
        operations = {"set": self.set, "add": self.add, "remove": self.remove}
        overlaps = {
            field: [op for op, fields in operations.items() if field in fields]
            for field in TRIP_REQUEST_FIELDS
        }
        conflicts = {field: ops for field, ops in sorted(overlaps.items()) if len(ops) > 1}
        if conflicts:
            raise ValueError(f"field(s) appear in more than one of set/add/remove: {conflicts}")
        return self

    @model_validator(mode="after")
    def _only_modify_carries_changes(self) -> TripShoppingRequestPatch:
        if self.intent is not RecommendationIntent.MODIFY_REQUIREMENTS and (
            self.set or self.add or self.remove
        ):
            raise ValueError(f"{self.intent} patch must have empty set/add/remove")
        return self


# ---------------------------------------------------------------------------
# Products and outfits
# ---------------------------------------------------------------------------


class Product(ContractModel):
    """Normalised product after Stage 8 (proposal Section 5.3.4) with its ProductScore."""

    product_id: NonEmptyStr
    source: NonEmptyStr
    title: NonEmptyStr
    description: str | None = None
    product_url: NonEmptyStr | None = None
    image_url: NonEmptyStr | None = None
    price: Money
    currency: NonEmptyStr
    availability: StrictBool
    category: ProductCategory
    product_score: CosineScoreValue
    # PROVISIONAL name for the proposal's "size/variants" (sizes currently in stock).
    available_sizes: list[NonEmptyStr] | None = None
    # PROVISIONAL: the proposal enforces size "only when reliable size information is
    # available". True means available_sizes can be trusted (an empty list then means no size
    # in stock); False means size is not checked for this product, whatever the list says.
    size_data_reliable: StrictBool = False
    brand: str | None = None
    colour: str | None = None
    material: str | None = None
    collected_at: datetime | None = None
    updated_at: datetime | None = None
    # PROVISIONAL: marks an item from the user's existing wardrobe. The original price is kept;
    # excluding it from the purchase budget is the constraint engine's job.
    is_user_owned: StrictBool = False
    # PROVISIONAL: optional reference (e.g. cache key) to the FashionCLIP embedding.
    embedding_ref: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _purchasable_needs_link(self) -> Product:
        if not self.is_user_owned and self.product_url is None:
            raise ValueError("product_url is required for purchasable products")
        if self.size_data_reliable and self.available_sizes is None:
            raise ValueError("size_data_reliable=True requires available_sizes (use [] for none)")
        return self


def _check_outfit_roles(items: list[Product]) -> None:
    """One TOP, BOTTOM and FOOTWEAR, optional OUTERWEAR, no repeated role or product."""
    role_counts = Counter(item.category for item in items)
    duplicated = sorted(role for role, n in role_counts.items() if n > 1)
    if duplicated:
        raise ValueError(f"duplicate outfit role(s): {duplicated}")
    missing = [role.value for role in REQUIRED_OUTFIT_ROLES if role not in role_counts]
    if missing:
        raise ValueError(f"outfit is missing required role(s): {missing}")
    id_counts = Counter(item.product_id for item in items)
    repeated = sorted(pid for pid, n in id_counts.items() if n > 1)
    if repeated:
        raise ValueError(f"duplicate product_id in outfit: {repeated}")


class CandidateOutfit(ContractModel):
    """Stage 9 candidate outfit. `items` shape is PROVISIONAL.

    `outfit_id` is only checked to be a non-empty string; no ID pattern is prescribed.
    `compatibility_score` may be None because Stage 11 runs after Stage 10.
    """

    outfit_id: NonEmptyStr
    items: list[Product] = Field(min_length=3, max_length=4)
    compatibility_score: CompatibilityScoreValue | None = None

    @field_validator("items")
    @classmethod
    def _check_roles(cls, items: list[Product]) -> list[Product]:
        _check_outfit_roles(items)
        return items


class FeasibleCandidateOutfit(CandidateOutfit):
    """Stage 10 output ("Feasible Candidate Outfits"). Model name is PROVISIONAL.

    Guards the availability invariant; budget and size depend on the request and are checked
    by constraint_engine.
    """

    @field_validator("items")
    @classmethod
    def _purchases_available(cls, items: list[Product]) -> list[Product]:
        unavailable = [i.product_id for i in items if not i.is_user_owned and not i.availability]
        if unavailable:
            raise ValueError(
                f"feasible outfit contains items that are not available: {unavailable}"
            )
        return items


class ScoredOutfit(FeasibleCandidateOutfit):
    """Feasible outfit with stored Stage 11/12 results. Model name is PROVISIONAL.

    Holds already-computed scores only; ranking_engine calculates them.
    """

    compatibility_score: CompatibilityScoreValue
    outfit_product_relevance: CosineScoreValue
    final_outfit_score: CosineScoreValue


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


class RecommendationItem(ContractModel):
    """One displayed outfit in a RecommendationSet. Structure is PROVISIONAL.

    Existing (user-owned) items are separated from items to buy; `total_purchase_cost`
    covers purchase items only.
    """

    outfit_id: NonEmptyStr
    display_position: int = Field(ge=1)
    existing_items: list[Product] = Field(default_factory=list)
    purchase_items: list[Product] = Field(default_factory=list)
    total_purchase_cost: Money
    currency: NonEmptyStr
    compatibility_score: CompatibilityScoreValue
    outfit_product_relevance: CosineScoreValue
    final_outfit_score: CosineScoreValue

    @model_validator(mode="after")
    def _check_items_and_cost(self) -> RecommendationItem:
        if any(not p.is_user_owned for p in self.existing_items):
            raise ValueError("existing_items must only contain user-owned items")
        if any(p.is_user_owned for p in self.purchase_items):
            raise ValueError("purchase_items must not contain user-owned items")
        # Stage 10 availability constraint applies to purchases only, not to the user's own clothes.
        unavailable = [p.product_id for p in self.purchase_items if not p.availability]
        if unavailable:
            raise ValueError(f"purchase_items contain items that are not available: {unavailable}")
        _check_outfit_roles(self.existing_items + self.purchase_items)
        mismatched = sorted({p.currency for p in self.purchase_items} - {self.currency})
        if mismatched:
            raise ValueError(
                f"purchase item currency {mismatched} differs from currency {self.currency}"
            )
        expected = sum((p.price for p in self.purchase_items), Decimal("0"))
        if self.total_purchase_cost != expected:
            raise ValueError(
                f"total_purchase_cost {self.total_purchase_cost} must equal the sum of "
                f"purchase_items prices {expected}"
            )
        return self


class RecommendationSet(ContractModel):
    """Top-3 outfits of one recommendation round (proposal Stage 12). Structure is PROVISIONAL."""

    recommendation_round: int = Field(ge=1)
    recommendations: list[RecommendationItem] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def _check_positions(self) -> RecommendationSet:
        positions = [r.display_position for r in self.recommendations]
        if len(set(positions)) != len(positions):
            raise ValueError(f"duplicate display_position: {positions}")
        if positions != list(range(1, len(positions) + 1)):
            raise ValueError(f"display_position must be 1..n in list order, got {positions}")
        ids = [r.outfit_id for r in self.recommendations]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate outfit_id: {ids}")
        scores = [r.final_outfit_score for r in self.recommendations]
        if any(a < b for a, b in zip(scores, scores[1:])):
            raise ValueError(f"final_outfit_score must be non-increasing by position, got {scores}")
        return self


# ---------------------------------------------------------------------------
# Dialogue state
# ---------------------------------------------------------------------------


class ConversationalRecommendationState(ContractModel):
    """Session-level state (proposal Appendix A). Field names are from the proposal;
    the consistency rules below are PROVISIONAL."""

    session_id: NonEmptyStr
    request_version: int = Field(ge=1)
    current_request: TripShoppingRequest
    recommendation_round: int = Field(ge=0)
    last_recommended_outfit_ids: list[NonEmptyStr] = Field(default_factory=list, max_length=3)
    selected_outfit_id: NonEmptyStr | None = None
    session_status: SessionStatus

    @field_validator("last_recommended_outfit_ids")
    @classmethod
    def _unique_ids(cls, ids: list[str]) -> list[str]:
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate outfit IDs: {ids}")
        return ids

    @model_validator(mode="after")
    def _check_selection(self) -> ConversationalRecommendationState:
        confirmed = self.session_status is SessionStatus.CONFIRMED
        if confirmed and self.selected_outfit_id is None:
            raise ValueError("CONFIRMED state requires selected_outfit_id")
        if not confirmed and self.selected_outfit_id is not None:
            raise ValueError(f"selected_outfit_id must be null while {self.session_status}")
        if (
            self.selected_outfit_id is not None
            and self.selected_outfit_id not in self.last_recommended_outfit_ids
        ):
            raise ValueError("selected_outfit_id must be one of last_recommended_outfit_ids")
        return self


# ---------------------------------------------------------------------------
# Stage 14
# ---------------------------------------------------------------------------

# Stage 14 models only describe data. Validation never downloads, opens or stores images, and
# never handles credentials; providers, prompts and temporary-file clean-up belong to the later
# preview module. Generating a preview is independent of confirmation: a failed preview never
# invalidates a ConfirmedOutfit.


class PreviewItemReference(ContractModel):
    """PROVISIONAL. One confirmed-outfit product as input to the preview generator.

    Pairs each image with its product and garment role. `image_reference` is a URL, temporary
    path or adapter-specific reference (never raw bytes); it is not checked for reachability.
    Mirrors Product: `title` is required and `description` is optional. The title is never
    copied into `description`; the later prompt builder uses the description when present and
    the title otherwise.
    """

    product_id: NonEmptyStr
    category: ProductCategory
    image_reference: ReferenceStr | None = None
    title: NonEmptyStr
    description: NonEmptyStr | None = None
    is_user_owned: StrictBool


class PreviewRequest(ContractModel):
    """PROVISIONAL. Input to the AI-Generated Outfit Preview (proposal Stage 14).

    `user_portrait` is a temporary reference/path, never raw image bytes, and may only be given
    with explicit `portrait_consent`. `items` keep the order in which the caller lists them.
    """

    selected_outfit_id: NonEmptyStr
    user_portrait: ReferenceStr | None = None
    portrait_consent: StrictBool = False
    items: list[PreviewItemReference] = Field(min_length=1)
    preview_mode: PreviewMode

    @field_validator("items")
    @classmethod
    def _unique_products(cls, items: list[PreviewItemReference]) -> list[PreviewItemReference]:
        counts = Counter(item.product_id for item in items)
        repeated = sorted(pid for pid, n in counts.items() if n > 1)
        if repeated:
            raise ValueError(f"duplicate product_id in preview items: {repeated}")
        return items

    @model_validator(mode="after")
    def _portrait_rules(self) -> PreviewRequest:
        if self.preview_mode is PreviewMode.PORTRAIT and self.user_portrait is None:
            raise ValueError("PORTRAIT preview_mode requires user_portrait")
        if self.user_portrait is not None and not self.portrait_consent:
            raise ValueError("user_portrait requires explicit portrait_consent=true")
        return self


class PreviewResult(ContractModel):
    """PROVISIONAL. Outcome of one Stage 14 preview attempt, returned separately from
    ConfirmedOutfit.

    Status matrix:
      SUCCESS  — delivered_mode set and equal to requested_mode;
      FALLBACK — delivered_mode set, may differ from requested_mode;
      FAILED   — delivered_mode and output_reference both null.
    For SUCCESS and FALLBACK, a generative delivered mode (PORTRAIT, MANNEQUIN, FLAT_LAY) needs an
    output_reference; PRODUCT_CARDS can be rendered from the request items, so it may have none.
    `output_reference` is never raw bytes. `disclaimer` is required; its wording is produced by
    the later preview module.
    """

    selected_outfit_id: NonEmptyStr
    status: PreviewStatus
    requested_mode: PreviewMode
    delivered_mode: PreviewMode | None = None
    output_reference: ReferenceStr | None = None
    provider_name: NonEmptyStr
    model_name: NonEmptyStr | None = None
    disclaimer: NonEmptyStr
    warnings: list[NonEmptyStr] = Field(default_factory=list)
    generation_time_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None

    @model_validator(mode="after")
    def _status_consistent(self) -> PreviewResult:
        if self.status is PreviewStatus.FAILED:
            if self.delivered_mode is not None:
                raise ValueError("FAILED result must have delivered_mode = null")
            if self.output_reference is not None:
                raise ValueError("FAILED result must not carry an output_reference")
            return self
        if self.delivered_mode is None:
            raise ValueError(f"{self.status} requires delivered_mode")
        if self.status is PreviewStatus.SUCCESS and self.delivered_mode is not self.requested_mode:
            raise ValueError(
                "SUCCESS requires delivered_mode == requested_mode; use FALLBACK otherwise"
            )
        if self.delivered_mode is not PreviewMode.PRODUCT_CARDS and self.output_reference is None:
            raise ValueError(
                f"{self.status} with generative delivered_mode {self.delivered_mode} requires "
                f"output_reference"
            )
        return self


class ConfirmedOutfit(ContractModel):
    """PROVISIONAL. The final selection saved when the user confirms (proposal Stage 13)."""

    session_id: NonEmptyStr
    selected_outfit_id: NonEmptyStr
    recommendation: RecommendationItem

    @model_validator(mode="after")
    def _ids_match(self) -> ConfirmedOutfit:
        if self.recommendation.outfit_id != self.selected_outfit_id:
            raise ValueError(
                f"selected_outfit_id {self.selected_outfit_id!r} does not match "
                f"recommendation outfit_id {self.recommendation.outfit_id!r}"
            )
        return self
