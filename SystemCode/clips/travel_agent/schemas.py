"""Shared contracts. Structural validation is not business feasibility validation."""

from datetime import date
from decimal import Decimal
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


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


# Proposal Appendix A contracts. The earlier TripRequest/FashionContext remain
# available for scaffold consumers, but understanding emits these contracts.

SUPPORTED_CURRENCIES = {
    "SGD": 2,
    "USD": 2,
    "EUR": 2,
    "GBP": 2,
    "JPY": 0,
    "KRW": 0,
    "CNY": 2,
    "MYR": 2,
    "THB": 2,
    "VND": 0,
    "IDR": 2,
}


class TripShoppingRequest(Contract):
    destination: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    activities: list[str] = Field(default_factory=list)
    budget: Decimal | None = None  # Major currency units, as in Appendix A.
    currency: str | None = None
    style: list[str] = Field(default_factory=list)
    excluded_items: list[str] = Field(default_factory=list)
    owned_item_ids: list[str] = Field(default_factory=list)
    # Optional, explicitly documented integration extensions; never inferred.
    places: list[str] = Field(default_factory=list)
    dress_codes: dict[str, str] = Field(default_factory=dict)
    enter_interior: bool | None = Field(default=None, strict=True)
    gender: Literal["male", "female"] | None = None
    cold_sensitive: bool | None = Field(default=None, strict=True)
    price_preference: Literal["lower"] | None = None

    @field_validator(
        "activities", "style", "excluded_items", "owned_item_ids", "places", mode="before"
    )
    @classmethod
    def empty_optional_lists(cls, value):
        return [] if value is None else value

    @field_validator("dress_codes", mode="before")
    @classmethod
    def empty_optional_map(cls, value):
        return {} if value is None else value

    @field_validator("destination")
    @classmethod
    def clean_destination(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Destination cannot be blank")
        return value.strip() if value else value

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def calendar_dates_only(cls, value):
        if value is not None and not isinstance(value, (str, date)):
            raise ValueError("Dates must be ISO calendar dates")
        return value

    @field_validator("budget", mode="before")
    @classmethod
    def positive_money(cls, value):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            raise ValueError("Budget must be a positive JSON number in major currency units")
        amount = Decimal(str(value))
        if not amount.is_finite() or amount <= 0:
            raise ValueError("Budget must be positive and finite")
        return amount

    @field_validator("currency")
    @classmethod
    def supported_currency(cls, value):
        if value is not None and value not in SUPPORTED_CURRENCIES:
            raise ValueError("Unsupported currency; use a supported ISO currency code")
        return value

    @field_validator("activities", "style", "excluded_items", "owned_item_ids", "places")
    @classmethod
    def nonempty_unique_strings(cls, values):
        if any(not value.strip() for value in values):
            raise ValueError("List entries cannot be blank")
        return list(dict.fromkeys(value.strip() for value in values))

    @field_validator("dress_codes")
    @classmethod
    def nonempty_dress_codes(cls, values):
        if any(not key.strip() or not value.strip() for key, value in values.items()):
            raise ValueError("Dress codes require nonempty activity keys and values")
        return values

    @model_validator(mode="after")
    def valid_range_and_money(self):
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("End date cannot precede start date")
        if self.budget is not None and self.currency:
            factor = Decimal(10) ** SUPPORTED_CURRENCIES[self.currency]
            if self.budget * factor != (self.budget * factor).to_integral_value():
                raise ValueError("Budget has more decimal places than the currency permits")
        return self

    @field_serializer("budget", when_used="json")
    def serialize_budget(self, value):
        if value is None:
            return None
        return int(value) if value == value.to_integral_value() else float(value)

    def budget_minor_units(self) -> int | None:
        if self.budget is None or self.currency is None:
            return None
        return int(self.budget * (Decimal(10) ** SUPPORTED_CURRENCIES[self.currency]))


class TripShoppingRequestPatch(Contract):
    intent: Literal["MODIFY_REQUIREMENTS"] = "MODIFY_REQUIREMENTS"
    set: dict[str, Any] = Field(default_factory=dict)
    add: dict[str, list[str]] = Field(default_factory=dict)
    remove: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def legal_fields_and_operations(self):
        fields = set(TripShoppingRequest.model_fields)
        list_fields = {"activities", "style", "excluded_items", "owned_item_ids", "places"}
        if (set(self.set) | set(self.add) | set(self.remove)) - fields:
            raise ValueError("Unknown patch field")
        if set(self.add) - list_fields:
            raise ValueError("add only accepts list fields")
        if set(self.set) & (set(self.add) | set(self.remove)):
            raise ValueError("Cannot set and add/remove the same field in one patch")
        for key, value in self.remove.items():
            if key in list_fields and (
                not isinstance(value, list) or any(not isinstance(v, str) for v in value)
            ):
                raise ValueError("List removals must be string arrays; [] resets the list")
            if key not in list_fields and value is not None:
                raise ValueError("Scalar/map resets must use null")
        return self


class ClimateMonth(Contract):
    month: int = Field(ge=1, le=12)
    sample_days: int = Field(gt=0)
    mean_daily_min_celsius: float = Field(allow_inf_nan=False)
    mean_daily_max_celsius: float = Field(allow_inf_nan=False)
    wet_day_frequency: float | None = Field(default=None, ge=0, le=1)
    snowfall_day_frequency: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def ordered_temperatures(self):
        if self.mean_daily_min_celsius > self.mean_daily_max_celsius:
            raise ValueError("Historical mean temperature range is reversed")
        return self


class ClimateReference(Contract):
    dataset_version: str
    baseline_start_year: int
    baseline_end_year: int
    model: str
    method: str
    monthly: list[ClimateMonth] = Field(min_length=1)
    source_urls: list[str] = Field(min_length=1)
    updated_at: str
    location_basis: str
    attribution: str
    grid_latitude: float | None = Field(default=None, ge=-90, le=90)
    grid_longitude: float | None = Field(default=None, ge=-180, le=180)
    grid_elevation_meters: float | None = Field(default=None, allow_inf_nan=False)
    limitations: list[str] = Field(default_factory=list)


class WeatherContext(Contract):
    temp_min_celsius: float | None = Field(default=None, allow_inf_nan=False)
    temp_max_celsius: float | None = Field(default=None, allow_inf_nan=False)
    snow: bool | None = Field(default=None, strict=True)
    precipitation_probability: float | None = Field(default=None, ge=0, le=1)
    humidity_percent: float | None = Field(default=None, ge=0, le=100)
    weather_class: Literal["COLD", "COOL", "MILD", "HOT"] | None = None
    status: Literal["ok", "unknown", "out_of_range", "needs_location"] = "unknown"
    data_type: Literal["forecast", "historical_climatology", "synthetic_fixture"] | None = None
    destination: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    source: str | None = None
    retrieved_at: str | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    location_label: str | None = None
    location_basis: str | None = None
    location_candidates: list[dict[str, Any]] = Field(default_factory=list)
    reason: str | None = None
    climate_reference: ClimateReference | None = None
    routing_reason: str | None = None
    reason_code: str | None = None
    next_steps: list[str] = Field(default_factory=list)
    acquisition: dict[str, Any] | None = None
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_weather(self):
        if self.data_type == "historical_climatology" and (
            self.snow is not None or self.precipitation_probability is not None
        ):
            raise ValueError("Historical frequencies are not trip-date snow or rain forecasts")
        if (
            self.temp_min_celsius is not None
            and self.temp_max_celsius is not None
            and self.temp_min_celsius > self.temp_max_celsius
        ):
            raise ValueError("Weather temperature range is reversed")
        if self.status == "ok" and not all(
            (self.data_type, self.destination, self.start_date, self.end_date, self.source)
        ):
            raise ValueError("Usable weather must include type, location, dates and source")
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("Weather date range is reversed")
        return self


class EnrichedTripContext(Contract):
    trip_request: TripShoppingRequest
    weather_context: WeatherContext


class WeatherClothingRequirements(Contract):
    outerwear: list[str] = Field(default_factory=list)
    top: list[str] = Field(default_factory=list)
    bottom: list[str] = Field(default_factory=list)
    footwear: list[str] = Field(default_factory=list)
    properties: list[str] = Field(default_factory=list)


class ConversationState(Contract):
    session_id: str = "local_demo"
    request_version: int = Field(default=0, ge=0, strict=True)
    current_request: TripShoppingRequest = Field(default_factory=TripShoppingRequest)
    pending_request: TripShoppingRequest | None = None
    recommendation_round: int = Field(default=0, ge=0, strict=True)
    last_recommended_outfit_ids: list[str] = Field(default_factory=list)
    selected_outfit_id: str | None = None
    session_status: Literal[
        "COLLECTING_REQUIREMENTS", "READY", "AWAITING_CONFIRMATION", "CONFIRMED"
    ] = "COLLECTING_REQUIREMENTS"


class OutfitConfirmation(Contract):
    intent: Literal["CONFIRM_OUTFIT"]
    outfit_id: str | None = None
    display_position: int | None = Field(default=None, ge=1, le=3, strict=True)


class ExplanationRequest(Contract):
    intent: Literal["REQUEST_EXPLANATION"]


class SeasonContext(Contract):
    season: Literal["spring", "summer", "autumn", "winter"] | None = None
    seasons: list[Literal["spring", "summer", "autumn", "winter"]] = Field(default_factory=list)
    status: Literal["derived", "multiple", "unknown", "unsupported"] = "unknown"
    method: str = "meteorological_calendar_v1"
    hemisphere: Literal["north", "south"] | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    location_basis: str | None = None
    sources: list[str] = Field(default_factory=list)
    review_status: Literal["pending"] = "pending"
    reason: str | None = None
