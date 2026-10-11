"""Shared contracts. Structural validation is not business feasibility validation."""

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TripRequest(Contract):
    destination: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    activity: str | None = None
    budget_minor: int | None = Field(default=None, gt=0, strict=True)
    currency: str = Field(default="SGD", pattern=r"^[A-Z]{3}$")
    excluded_items: list[str] = Field(default_factory=list)
    owned_item_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def dates_are_ordered(self):
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("End date cannot precede start date")
        return self


class FashionContext(Contract):
    request: TripRequest
    hard_requirements: list[str] = Field(default_factory=list)
    soft_preferences: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class ProductCandidate(Contract):
    id: str
    role: Literal["top", "bottom", "footwear", "outerwear"]
    origin: Literal["owned", "new"]
    purchase_cost_minor: int = Field(ge=0, strict=True)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    availability: Literal["available", "unavailable", "unknown"] = "unknown"
    image_ref: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def owned_cost_is_zero(self):
        if self.origin == "owned" and self.purchase_cost_minor != 0:
            raise ValueError("Owned items must have zero purchase cost")
        return self


class OutfitResult(Contract):
    outfit_id: str
    item_ids: list[str]
    total_new_cost_minor: int = Field(ge=0, strict=True)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    constraint_status: Literal["unverified", "passed", "failed"] = "unverified"
    violations: list[str] = Field(default_factory=list)


class ModuleResult(Contract):
    status: Literal["ok", "needs_input", "no_candidates", "not_implemented", "error"]
    data: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
