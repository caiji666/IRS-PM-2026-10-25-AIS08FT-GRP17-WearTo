"""Currency normalisation boundary for Student 5 recommendations."""

from __future__ import annotations

import json
import os
import ssl
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Callable, Mapping

import certifi

from travel_agent.recommendation.student5.models import (
    CandidateOutfit,
    RecommendationSet,
    TripShoppingRequest,
)

ECB_DAILY_RATES_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
DEFAULT_CACHE_PATH = Path("runtime") / "cache" / "ecb_fx_rates.json"
MAX_CACHE_AGE = timedelta(days=7)


class CurrencyConversionError(ValueError):
    """A requested currency cannot be converted safely."""


@dataclass(frozen=True)
class ExchangeRateSnapshot:
    rate_date: date
    rates_per_eur: Mapping[str, Decimal]
    source: str

    def __post_init__(self) -> None:
        normalised = {
            code.strip().upper(): Decimal(rate) for code, rate in self.rates_per_eur.items()
        }
        if normalised.get("EUR") != Decimal("1"):
            raise CurrencyConversionError("exchange rates must include EUR with rate 1")
        invalid = sorted(code for code, rate in normalised.items() if not code or rate <= 0)
        if invalid:
            raise CurrencyConversionError(f"exchange rates must be positive: {invalid}")
        object.__setattr__(self, "rates_per_eur", normalised)


@dataclass(frozen=True)
class CurrencyNormalisation:
    request: TripShoppingRequest
    outfits: list[CandidateOutfit]
    original_budget: Decimal
    original_currency: str
    snapshot: ExchangeRateSnapshot


def convert_amount(
    amount: Decimal,
    source_currency: str,
    target_currency: str,
    snapshot: ExchangeRateSnapshot,
) -> Decimal:
    source = source_currency.strip().upper()
    target = target_currency.strip().upper()
    missing = sorted({source, target} - set(snapshot.rates_per_eur))
    if missing:
        raise CurrencyConversionError(
            f"currency {', '.join(missing)} is not available in the {snapshot.source} rate snapshot"
        )
    value = Decimal(amount)
    if source == target:
        return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    converted = value / snapshot.rates_per_eur[source] * snapshot.rates_per_eur[target]
    return converted.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def normalise_to_sgd(
    request: TripShoppingRequest,
    outfits: list[CandidateOutfit],
    snapshot: ExchangeRateSnapshot,
) -> CurrencyNormalisation:
    original_currency = request.currency.strip().upper()
    normalised_request = TripShoppingRequest.model_validate(
        {
            **request.model_dump(),
            "budget": convert_amount(request.budget, original_currency, "SGD", snapshot),
            "currency": "SGD",
        }
    )
    normalised_outfits: list[CandidateOutfit] = []
    for outfit in outfits:
        items = []
        for item in outfit.items:
            original_item_currency = item.currency.strip().upper()
            items.append(
                {
                    **item.model_dump(),
                    "price": convert_amount(item.price, original_item_currency, "SGD", snapshot),
                    "currency": "SGD",
                    "original_price": item.price,
                    "original_currency": original_item_currency,
                    "fx_rate_date": snapshot.rate_date,
                    "fx_source": snapshot.source,
                }
            )
        normalised_outfits.append(
            CandidateOutfit.model_validate({**outfit.model_dump(), "items": items})
        )
    return CurrencyNormalisation(
        request=normalised_request,
        outfits=normalised_outfits,
        original_budget=request.budget,
        original_currency=original_currency,
        snapshot=snapshot,
    )


def recommendation_in_currency(
    recommendation_set: RecommendationSet,
    target_currency: str,
    snapshot: ExchangeRateSnapshot,
) -> RecommendationSet:
    """Return a display copy whose prices and totals use the user's requested currency."""
    target = target_currency.strip().upper()
    recommendations = []
    for recommendation in recommendation_set.recommendations:
        existing_items = []
        purchase_items = []
        for source, destination in (
            (recommendation.existing_items, existing_items),
            (recommendation.purchase_items, purchase_items),
        ):
            for product in source:
                destination.append(
                    {
                        **product.model_dump(),
                        "price": convert_amount(product.price, product.currency, target, snapshot),
                        "currency": target,
                    }
                )
        recommendations.append(
            {
                **recommendation.model_dump(),
                "existing_items": existing_items,
                "purchase_items": purchase_items,
                "total_purchase_cost": sum(
                    (item["price"] for item in purchase_items), Decimal("0")
                ),
                "currency": target,
            }
        )
    return RecommendationSet.model_validate(
        {
            **recommendation_set.model_dump(),
            "recommendations": recommendations,
        }
    )


def parse_ecb_rates(raw_xml: bytes) -> ExchangeRateSnapshot:
    try:
        root = ET.fromstring(raw_xml)
        dated_cube = next(element for element in root.iter() if element.attrib.get("time"))
        rate_date = date.fromisoformat(dated_cube.attrib["time"])
        rates = {"EUR": Decimal("1")}
        for element in dated_cube:
            currency = element.attrib.get("currency")
            rate = element.attrib.get("rate")
            if currency and rate:
                rates[currency] = Decimal(rate)
    except (ET.ParseError, StopIteration, KeyError, ValueError) as exc:
        raise CurrencyConversionError("ECB returned invalid exchange-rate data") from exc
    snapshot = ExchangeRateSnapshot(rate_date=rate_date, rates_per_eur=rates, source="ECB")
    if "SGD" not in snapshot.rates_per_eur:
        raise CurrencyConversionError("ECB exchange-rate data does not contain SGD")
    return snapshot


def _read_cache(path: Path) -> tuple[ExchangeRateSnapshot, date | None] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return (
            ExchangeRateSnapshot(
                rate_date=date.fromisoformat(raw["rate_date"]),
                rates_per_eur={
                    currency: Decimal(rate) for currency, rate in raw["rates_per_eur"].items()
                },
                source=raw["source"],
            ),
            date.fromisoformat(raw["checked_on"]) if raw.get("checked_on") else None,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, snapshot: ExchangeRateSnapshot, checked_on: date) -> None:
    payload = {
        "checked_on": checked_on.isoformat(),
        "rate_date": snapshot.rate_date.isoformat(),
        "source": snapshot.source,
        "rates_per_eur": {currency: str(rate) for currency, rate in snapshot.rates_per_eur.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _download_ecb_rates() -> bytes:
    request = urllib.request.Request(
        ECB_DAILY_RATES_URL,
        headers={"User-Agent": "travel-style-agent/1.0"},
    )
    verified_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=10, context=verified_context) as response:
        return response.read()


def load_ecb_snapshot(
    *,
    cache_path: Path | None = None,
    today: date | None = None,
    download: Callable[[], bytes] | None = None,
) -> ExchangeRateSnapshot:
    path = cache_path or Path(os.environ.get("ECB_FX_CACHE_PATH", str(DEFAULT_CACHE_PATH)))
    current_date = today or date.today()
    cached_entry = _read_cache(path)
    cached = cached_entry[0] if cached_entry is not None else None
    checked_on = cached_entry[1] if cached_entry is not None else None
    if cached is not None and (cached.rate_date == current_date or checked_on == current_date):
        return cached

    fetch = download or _download_ecb_rates
    try:
        fresh = parse_ecb_rates(fetch())
        if fresh.rate_date > current_date:
            raise CurrencyConversionError("ECB exchange-rate date is in the future")
        if current_date - fresh.rate_date > MAX_CACHE_AGE:
            raise CurrencyConversionError("ECB did not provide sufficiently fresh exchange rates")
        _write_cache(path, fresh, current_date)
        return fresh
    except (OSError, CurrencyConversionError) as exc:
        if cached is not None and timedelta(0) <= current_date - cached.rate_date <= MAX_CACHE_AGE:
            return cached
        raise CurrencyConversionError(
            "fresh exchange rates are unavailable and no cache from the last 7 days exists"
        ) from exc
