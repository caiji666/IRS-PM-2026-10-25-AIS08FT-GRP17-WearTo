"""Student 4 → Student 5 hand-off: adapt Student 4's raw output to list[CandidateOutfit].

This is the single integration point for Student 4's compatibility model. It only converts and
validates; it does not filter, score, rank, confirm or preview, and it never imports Student 4
code, loads a model, reads configuration or files, or touches the network.

Expected payload (PROVISIONAL; the outer wrapper is still to be confirmed with Student 4):

    {
      "status": "ok",
      "model_version": "<string, optional>",
      "source": "<string, optional>",
      "outfits": [<outfit>, ...]
    }

Each outfit has `outfit_id`, `compatibility_score` and exactly one of two item shapes:

- `items`: a list of objects that already use the Product field names (the original contract).
  Money is given either as `price` (validated by Product) or as integer `price_minor` in the
  currency's minor unit (e.g. SGD 3990 → 39.90), converted with Decimal only. A non-empty
  `image_path` (an upstream-downloaded local image) is used as `image_url`. Unknown fields are
  rejected.
- `products`: an object keyed by category (`TOP`, `BOTTOM`, `FOOTWEAR`, optional
  `OUTERWEAR`), each value a product object in the Student 3 / Student 4 format (the shape
  Student 4's compatibility service returns). It is normalised to Product fields:

      price.amount / price.currency       → price / currency
      sizes                               → available_sizes
      source_type, else provenance.source → source
      owned                               → is_user_owned
      category key (and `category`)       → ProductCategory
      provenance.original.api_item.image.imageUrl, else image_path → image_url
      availability (only with evidence.availability_verified == true):
                   IN_STOCK / AVAILABLE → True, OUT_OF_STOCK / UNAVAILABLE → False
      evidence.sizes_verified             → size_data_reliable (missing → False)

  Availability is never guessed: `UNKNOWN` raises `availability_unknown`, and a known status
  whose `evidence.availability_verified` is missing or not `true` raises
  `availability_unverified`. Such an outfit does not reach the recommendation.
  Fields that Product does not define (embeddings, provenance, evidence, ...) are dropped.

ProductScore comes from the product's own `product_score`. If it is absent the adapter raises
`missing_product_score`; it never invents a score and never uses compatibility_score instead.
(The keyword-only `product_score_by_id` lookup is TEMPORARY, for adapter tests only: the
service does not pass it, and the final ProductScore location is decided once Student 4's new
output arrives.)

`model_version` and `source` are run metadata and are returned beside the outfits, never stored
in CandidateOutfit.

Everything here is PROVISIONAL until Student 4 hands off its final output.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from travel_agent.recommendation.student5.models import CandidateOutfit, Product, ProductCategory

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

# PROVISIONAL: availability vocabulary of the Student 3 / Student 4 product objects.
_AVAILABILITY = {
    "IN_STOCK": True,
    "AVAILABLE": True,
    "OUT_OF_STOCK": False,
    "UNAVAILABLE": False,
}
_UNKNOWN_AVAILABILITY = "UNKNOWN"

_CATEGORY_ORDER = (
    ProductCategory.TOP,
    ProductCategory.BOTTOM,
    ProductCategory.FOOTWEAR,
    ProductCategory.OUTERWEAR,
)
_PRODUCT_FIELDS = frozenset(Product.model_fields)


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


@dataclass(frozen=True)
class _Context:
    # TEMPORARY: product_id → ProductScore lookup, for adapter tests only (see module docstring).
    product_score_by_id: Mapping[str, Any] | None


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


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


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float, Decimal))
        and math.isfinite(value)
    )


def _product_score(item: dict[str, Any], context: _Context, location: str) -> None:
    """Fill `product_score` from the lookup when the product has none; never guess one."""
    if "product_score" in item:
        return
    lookup = context.product_score_by_id
    product_id = item.get("product_id")
    if lookup is None or not isinstance(product_id, str) or product_id not in lookup:
        raise Student4AdapterError(
            "missing_product_score",
            f"{location}.product_score",
            "no product_score on the product and none in product_score_by_id",
        )
    score = lookup[product_id]
    if not _is_number(score) or not -1 <= score <= 1:
        raise Student4AdapterError(
            "invalid_product_score",
            f"{location}.product_score",
            "product_score_by_id values must be numbers in [-1, 1]",
        )
    item["product_score"] = float(score)


def _validated(item: dict[str, Any], location: str) -> dict[str, Any]:
    try:
        Product.model_validate(item)
    except ValidationError as exc:
        raise Student4AdapterError(
            "invalid_item", location, f"item does not satisfy the Product contract: {exc}"
        ) from None
    return item


# ---------------------------------------------------------------------------
# `items` outfits (original contract)
# ---------------------------------------------------------------------------


def _adapt_item(raw: Any, location: str, context: _Context) -> dict[str, Any]:
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
    _product_score(item, context, location)
    return _validated(item, location)


# ---------------------------------------------------------------------------
# `products` outfits (Student 3 / Student 4 product objects keyed by category)
# ---------------------------------------------------------------------------


def _category(value: Any) -> ProductCategory | None:
    if not isinstance(value, str):
        return None
    try:
        return ProductCategory(value.strip().upper())
    except ValueError:
        return None


def _get(mapping: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(mapping, Mapping):
            return None
        mapping = mapping.get(key)
    return mapping


def _product_price(product: Mapping[str, Any], location: str) -> tuple[Decimal, Any]:
    top_currency = product.get("currency")
    if "price_minor" in product and "price" not in product:
        price = _price_from_minor(product["price_minor"], top_currency, f"{location}.price_minor")
        return price, top_currency
    price = product.get("price")
    if isinstance(price, Mapping):
        amount, currency = price.get("amount"), price.get("currency", top_currency)
    elif _is_number(price):
        amount, currency = price, top_currency
    else:
        raise Student4AdapterError(
            "invalid_price",
            f"{location}.price",
            "price must be an object with amount and currency",
        )
    if not _is_number(amount) or amount < 0:
        raise Student4AdapterError(
            "invalid_price", f"{location}.price.amount", "amount must be a non-negative number"
        )
    if not isinstance(currency, str) or not currency.strip():
        raise Student4AdapterError(
            "invalid_price", f"{location}.price.currency", "currency must be a non-empty string"
        )
    if top_currency is not None and top_currency != currency:
        raise Student4AdapterError(
            "invalid_price", f"{location}.currency", "currency conflicts with price.currency"
        )
    return Decimal(str(amount)), currency


def _availability(product: Mapping[str, Any], location: str) -> bool:
    """Known stock status confirmed by `evidence.availability_verified == true`; never guessed."""
    raw = product.get("availability")
    value = raw.strip().upper() if isinstance(raw, str) else raw
    if value == _UNKNOWN_AVAILABILITY:
        raise Student4AdapterError(
            "availability_unknown",
            f"{location}.availability",
            "stock status is UNKNOWN; the product cannot be recommended",
        )
    if isinstance(value, bool):
        available = value
    elif value in _AVAILABILITY:
        available = _AVAILABILITY[value]
    else:
        raise Student4AdapterError(
            "invalid_availability",
            f"{location}.availability",
            "availability must be IN_STOCK, AVAILABLE, OUT_OF_STOCK or UNAVAILABLE",
        )
    evidence = product.get("evidence")
    verified = evidence.get("availability_verified") if isinstance(evidence, Mapping) else None
    if verified is not True:
        raise Student4AdapterError(
            "availability_unverified",
            f"{location}.evidence.availability_verified",
            "stock status is not verified (evidence.availability_verified must be true)",
        )
    return available


def _normalise_product(
    product: Mapping[str, Any], category: ProductCategory, location: str, context: _Context
) -> dict[str, Any]:
    # Product fields given as such are kept; everything else is dropped.
    item = {key: value for key, value in product.items() if key in _PRODUCT_FIELDS}
    item["category"] = category.value
    item["price"], item["currency"] = _product_price(product, location)
    if "available_sizes" not in item and "sizes" in product:
        item["available_sizes"] = product["sizes"]
    if "source" not in item:
        source = product.get("source_type") or _get(product, "provenance", "source")
        if source is not None:
            item["source"] = source
    if "owned" in product:
        if "is_user_owned" in item and item["is_user_owned"] != product["owned"]:
            raise Student4AdapterError("invalid_item", location, "owned and is_user_owned disagree")
        item["is_user_owned"] = product["owned"]
    image = _get(product, "provenance", "original", "api_item", "image", "imageUrl")
    if isinstance(image, str) and image.strip():
        item["image_url"] = image
    elif product.get("image_path"):
        item["image_url"] = product["image_path"]
    item["availability"] = _availability(product, location)
    if "size_data_reliable" not in item:
        evidence = product.get("evidence")
        item["size_data_reliable"] = (
            evidence["sizes_verified"]
            if isinstance(evidence, Mapping) and "sizes_verified" in evidence
            else False
        )
    _product_score(item, context, location)
    return _validated(item, location)


def _adapt_products(raw: Any, location: str, context: _Context) -> list[dict[str, Any]]:
    if not isinstance(raw, Mapping) or not raw:
        raise Student4AdapterError(
            "invalid_products", location, "products must be a non-empty object keyed by category"
        )
    by_category: dict[ProductCategory, dict[str, Any]] = {}
    for key, product in raw.items():
        product_location = f"{location}.{key}"
        category = _category(key)
        if category is None:
            raise Student4AdapterError(
                "invalid_category",
                product_location,
                "category key must be TOP, BOTTOM, FOOTWEAR or OUTERWEAR",
            )
        if category in by_category:
            raise Student4AdapterError(
                "invalid_category", product_location, "category appears more than once"
            )
        if not isinstance(product, Mapping):
            raise Student4AdapterError(
                "invalid_item", product_location, "product must be an object"
            )
        if "category" in product and _category(product["category"]) is not category:
            raise Student4AdapterError(
                "invalid_category",
                f"{product_location}.category",
                "product category does not match its category key",
            )
        by_category[category] = _normalise_product(product, category, product_location, context)
    return [by_category[c] for c in _CATEGORY_ORDER if c in by_category]


# ---------------------------------------------------------------------------
# Outfits and payload
# ---------------------------------------------------------------------------


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


def _adapt_outfit(raw: Any, location: str, context: _Context) -> CandidateOutfit:
    if not isinstance(raw, Mapping):
        raise Student4AdapterError("invalid_outfit", location, "outfit must be an object")
    outfit_id = raw.get("outfit_id")
    if not isinstance(outfit_id, str) or not outfit_id.strip():
        raise Student4AdapterError(
            "invalid_outfit_id", f"{location}.outfit_id", "outfit_id must be a non-empty string"
        )
    score = _check_score(raw.get("compatibility_score"), f"{location}.compatibility_score")
    if "products" in raw:
        if "items" in raw:
            raise Student4AdapterError(
                "invalid_items", location, "give either items or products, not both"
            )
        items = _adapt_products(raw["products"], f"{location}.products", context)
    else:
        raw_items = raw.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise Student4AdapterError(
                "invalid_items", f"{location}.items", "items must be a non-empty list"
            )
        items = [
            _adapt_item(item, f"{location}.items[{n}]", context) for n, item in enumerate(raw_items)
        ]
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
    payload: Mapping[str, Any],
    *,
    source: str | None = None,
    product_score_by_id: Mapping[str, float] | None = None,
) -> Student4AdapterResult:
    """PROVISIONAL. Convert Student 4's raw output into CandidateOutfits plus run metadata.

    `product_score_by_id` is TEMPORARY and for adapter tests only: the service never passes it,
    and the final ProductScore location is decided when Student 4's new output arrives. Pure and
    deterministic; the payload is not modified. Raises Student4AdapterError (a ValueError with
    `code` and `location`) for any contract violation. Outfit order is kept.
    """
    if not isinstance(payload, Mapping):
        raise Student4AdapterError("invalid_payload", "$", "payload must be an object")
    if product_score_by_id is not None and not isinstance(product_score_by_id, Mapping):
        raise Student4AdapterError(
            "invalid_product_score_by_id",
            "product_score_by_id",
            "product_score_by_id must be an object mapping product_id to product_score",
        )
    if payload.get("status") != "ok":
        raise Student4AdapterError(
            "status_not_ok", "status", "Student 4 did not report status 'ok'"
        )
    raw_outfits = payload.get("outfits")
    if not isinstance(raw_outfits, list) or not raw_outfits:
        raise Student4AdapterError("no_outfits", "outfits", "outfits must be a non-empty list")
    model_version, chosen_source = _metadata(payload, source)
    context = _Context(product_score_by_id=product_score_by_id)
    outfits = []
    seen: set[str] = set()
    for n, raw in enumerate(raw_outfits):
        outfit = _adapt_outfit(raw, f"outfits[{n}]", context)
        if outfit.outfit_id in seen:
            raise Student4AdapterError(
                "duplicate_outfit_id", f"outfits[{n}].outfit_id", "outfit_id values must be unique"
            )
        seen.add(outfit.outfit_id)
        outfits.append(outfit)
    return Student4AdapterResult(outfits=outfits, model_version=model_version, source=chosen_source)
