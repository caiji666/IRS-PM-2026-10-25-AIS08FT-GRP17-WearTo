"""Student 4 → Student 5 hand-off: adapt Student 4's raw output to list[CandidateOutfit].

This is the single integration point for Student 4's compatibility model. It only converts and
validates; it does not filter, score, rank, confirm or preview, and it never imports Student 4
code, loads a model, reads configuration or touches the network.

Expected payload (PROVISIONAL; to be confirmed with Student 4):

    {
      "status": "ok",
      "model_version": "<string, optional>",
      "source": "<string, optional>",
      "outfits": [
        {"outfit_id": "outfit_001", "compatibility_score": 0.86, "items": [<product>, ...]}
      ]
    }

Each item uses the Product field names. Money is given either as `price` (validated by
Product) or as integer `price_minor` in the currency's minor unit (e.g. SGD 3990 → 39.90),
which is converted with Decimal only. If an item carries a non-empty `image_path` (an
upstream-downloaded local product image), it is used as `image_url`, so local files are
preferred. `model_version` and `source` are run metadata and are returned beside the outfits,
never stored in CandidateOutfit.

Everything here is PROVISIONAL until Student 4 hands off its implementation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from travel_agent.recommendation.student5.models import CandidateOutfit, Product

DEFAULT_SOURCE = "student4"

# ISO 4217 minor-unit exponents that differ from the default of 2.
_MINOR_UNIT_EXPONENTS = {
    "JPY": 0,
    "KRW": 0,
    "VND": 0,
    "CLP": 0,
    "ISK": 0,
    "UGX": 0,
    "XAF": 0,
    "XOF": 0,
    "BHD": 3,
    "IQD": 3,
    "JOD": 3,
    "KWD": 3,
    "LYD": 3,
    "OMR": 3,
    "TND": 3,
}


class Student4AdapterError(ValueError):
    """PROVISIONAL. Structured error: `code` names the problem, `location` the payload path."""

    def __init__(self, code: str, location: str, message: str) -> None:
        super().__init__(f"{code} at {location}: {message}")
        self.code = code
        self.location = location
        self.message = message


@dataclass(frozen=True)
class Student4AdapterResult:
    """PROVISIONAL. Adapted outfits plus run metadata (kept out of CandidateOutfit)."""

    outfits: list[CandidateOutfit]
    model_version: str | None
    source: str


def _price_from_minor(minor: Any, currency: Any, location: str) -> Decimal:
    if isinstance(minor, bool) or not isinstance(minor, int) or minor < 0:
        raise Student4AdapterError(
            "invalid_price", location, "price_minor must be a non-negative integer"
        )
    if not isinstance(currency, str) or not currency.strip():
        raise Student4AdapterError(
            "invalid_price", location, "price_minor needs a currency to know its minor unit"
        )
    exponent = _MINOR_UNIT_EXPONENTS.get(currency.strip().upper(), 2)
    return Decimal(minor).scaleb(-exponent)


def _adapt_item(raw: Any, location: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise Student4AdapterError("invalid_item", location, "item must be an object")
    item = dict(raw)
    if "price_minor" in item:
        if "price" in item:
            raise Student4AdapterError(
                "invalid_price",
                f"{location}.price_minor",
                "give either price or price_minor, not both",
            )
        item["price"] = _price_from_minor(
            item.pop("price_minor"), item.get("currency"), f"{location}.price_minor"
        )
    image_path = item.pop("image_path", None)
    if image_path:
        item["image_url"] = image_path
    try:
        Product.model_validate(item)
    except ValidationError as exc:
        raise Student4AdapterError(
            "invalid_item", location, f"item does not satisfy the Product contract: {exc}"
        ) from None
    return item


def _check_score(score: Any, location: str) -> float:
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise Student4AdapterError(
            "invalid_compatibility_score",
            location,
            "compatibility_score must be a number in [0, 1]",
        )
    return score


def _adapt_outfit(raw: Any, location: str) -> CandidateOutfit:
    if not isinstance(raw, Mapping):
        raise Student4AdapterError("invalid_outfit", location, "outfit must be an object")
    outfit_id = raw.get("outfit_id")
    if not isinstance(outfit_id, str) or not outfit_id.strip():
        raise Student4AdapterError(
            "invalid_outfit_id", f"{location}.outfit_id", "outfit_id must be a non-empty string"
        )
    score = _check_score(raw.get("compatibility_score"), f"{location}.compatibility_score")
    raw_items = raw.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise Student4AdapterError(
            "invalid_items", f"{location}.items", "items must be a non-empty list"
        )
    items = [_adapt_item(item, f"{location}.items[{n}]") for n, item in enumerate(raw_items)]
    try:
        return CandidateOutfit.model_validate(
            {"outfit_id": outfit_id, "compatibility_score": score, "items": items}
        )
    except ValidationError as exc:
        raise Student4AdapterError(
            "invalid_outfit",
            location,
            f"outfit does not satisfy the CandidateOutfit contract: {exc}",
        ) from None


def _metadata(payload: Mapping[str, Any], source: str | None) -> tuple[str | None, str]:
    model_version = payload.get("model_version")
    if model_version is not None and not isinstance(model_version, str):
        raise Student4AdapterError(
            "invalid_metadata", "model_version", "model_version must be a string"
        )
    chosen = source if source is not None else payload.get("source", DEFAULT_SOURCE)
    if not isinstance(chosen, str) or not chosen.strip():
        raise Student4AdapterError(
            "invalid_metadata", "source", "source must be a non-empty string"
        )
    return model_version, chosen


def adapt_student4_output(
    payload: Mapping[str, Any], *, source: str | None = None
) -> Student4AdapterResult:
    """PROVISIONAL. Convert Student 4's raw output into CandidateOutfits plus run metadata.

    Pure and deterministic; the payload is not modified. Raises Student4AdapterError (a
    ValueError with `code` and `location`) for any contract violation. Outfit order is kept.
    """
    if not isinstance(payload, Mapping):
        raise Student4AdapterError("invalid_payload", "$", "payload must be an object")
    if payload.get("status") != "ok":
        raise Student4AdapterError(
            "status_not_ok", "status", "Student 4 did not report status 'ok'"
        )
    raw_outfits = payload.get("outfits")
    if not isinstance(raw_outfits, list) or not raw_outfits:
        raise Student4AdapterError("no_outfits", "outfits", "outfits must be a non-empty list")
    model_version, chosen_source = _metadata(payload, source)
    outfits = []
    seen: set[str] = set()
    for n, raw in enumerate(raw_outfits):
        outfit = _adapt_outfit(raw, f"outfits[{n}]")
        if outfit.outfit_id in seen:
            raise Student4AdapterError(
                "duplicate_outfit_id", f"outfits[{n}].outfit_id", "outfit_id values must be unique"
            )
        seen.add(outfit.outfit_id)
        outfits.append(outfit)
    return Student4AdapterResult(outfits=outfits, model_version=model_version, source=chosen_source)
