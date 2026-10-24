from datetime import date
from decimal import Decimal

import pytest

from travel_agent.recommendation.currency_converter import (
    CurrencyConversionError,
    ExchangeRateSnapshot,
    convert_amount,
    normalise_to_sgd,
)
from travel_agent.recommendation.student5.models import (
    CandidateOutfit,
    Product,
    TripShoppingRequest,
)


def snapshot():
    return ExchangeRateSnapshot(
        rate_date=date(2026, 10, 21),
        rates_per_eur={
            "EUR": Decimal("1"),
            "SGD": Decimal("1.5"),
            "USD": Decimal("1.2"),
            "CNY": Decimal("8"),
            "MYR": Decimal("5"),
            "JPY": Decimal("180"),
        },
        source="ECB",
    )


def request():
    return TripShoppingRequest.model_validate(
        {
            "destination": "Seoul",
            "start_date": "2026-12-20",
            "end_date": "2026-12-27",
            "budget": "800",
            "currency": "CNY",
        }
    )


def product(product_id, price, currency, *, category="TOP", owned=False):
    return Product.model_validate(
        {
            "product_id": product_id,
            "source": "test",
            "title": product_id,
            "product_url": f"https://example.com/{product_id}",
            "price": price,
            "currency": currency,
            "availability": True,
            "category": category,
            "product_score": 0.5,
            "is_user_owned": owned,
        }
    )


def test_cross_rate_conversion_uses_decimal_and_sgd_is_unchanged():
    rates = snapshot()
    assert convert_amount(Decimal("120"), "USD", "SGD", rates) == Decimal("150.00")
    assert convert_amount(Decimal("800"), "CNY", "SGD", rates) == Decimal("150.00")
    assert convert_amount(Decimal("12.34"), "SGD", "SGD", rates) == Decimal("12.34")


def test_normalisation_preserves_original_money_and_converts_every_item_to_sgd():
    outfit = CandidateOutfit(
        outfit_id="o1",
        items=[
            product("usd_top", "120", "USD"),
            product("owned_jpy", "9000", "JPY", category="BOTTOM", owned=True),
            product("myr_shoes", "100", "MYR", category="FOOTWEAR"),
        ],
        compatibility_score=0.8,
    )
    result = normalise_to_sgd(request(), [outfit], snapshot())
    assert (result.request.budget, result.request.currency) == (Decimal("150.00"), "SGD")
    converted = result.outfits[0].items
    assert (converted[0].price, converted[0].currency) == (Decimal("150.00"), "SGD")
    assert (converted[0].original_price, converted[0].original_currency) == (
        Decimal("120"),
        "USD",
    )
    assert (converted[1].price, converted[1].currency) == (Decimal("75.00"), "SGD")
    assert result.original_budget == Decimal("800")
    assert result.original_currency == "CNY"


def test_unsupported_currency_fails_instead_of_assuming_one_to_one():
    with pytest.raises(CurrencyConversionError, match="XYZ"):
        convert_amount(Decimal("10"), "XYZ", "SGD", snapshot())


def test_currency_codes_are_case_insensitive():
    assert convert_amount(Decimal("120"), "usd", "sgd", snapshot()) == Decimal("150.00")
