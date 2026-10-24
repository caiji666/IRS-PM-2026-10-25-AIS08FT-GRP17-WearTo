"""Tests for the Student 4 → Student 5 adapter (Module 9).

All payloads here are hand-written CONTRACT FIXTURES that follow the expected Student 4 output
format. They are not real model results; every compatibility_score below is an arbitrary test
value, not produced by Student 4's compatibility model.
"""

import copy
from decimal import Decimal

import pytest

from travel_agent.recommendation.student5.models import (
    CandidateOutfit,
    Product,
    ProductCategory,
)
from travel_agent.recommendation.student5.student4_adapter import (
    Student4AdapterError,
    adapt_student4_output,
)


def contract_item(category, **overrides):
    """Contract fixture item in the expected Student 4 format (not real data)."""
    item = {
        "product_id": f"p_{category.lower()}",
        "category": category,
        "source": "ebay",
        "title": f"Contract fixture {category.lower()}",
        "description": f"{category.lower()} description",
        "product_url": f"https://shop.example.com/{category.lower()}",
        "image_url": f"https://img.example.com/{category.lower()}.jpg",
        "price_minor": 3990,
        "currency": "SGD",
        "availability": True,
        "product_score": 0.5,
        "available_sizes": ["M", "L"],
        "size_data_reliable": True,
        "is_user_owned": False,
    }
    item.update(overrides)
    return item


_DEFAULT_ITEMS = object()


def contract_outfit(outfit_id="outfit_001", score=0.86, items=_DEFAULT_ITEMS):
    return {
        "outfit_id": outfit_id,
        "compatibility_score": score,
        "items": items
        if items is not _DEFAULT_ITEMS
        else [contract_item("TOP"), contract_item("BOTTOM"), contract_item("FOOTWEAR")],
    }


def contract_payload(*outfits, **overrides):
    """Contract fixture payload (expected Student 4 output shape; not a real model result)."""
    payload = {
        "status": "ok",
        "model_version": "contract-fixture-version",
        "source": "student4-contract-fixture",
        "outfits": list(outfits) or [contract_outfit()],
    }
    payload.update(overrides)
    return payload


def adapter_error(payload, code):
    with pytest.raises(Student4AdapterError) as exc:
        adapt_student4_output(payload)
    assert exc.value.code == code
    return exc.value


# ---------------------------------------------------------------------------
# Valid payloads
# ---------------------------------------------------------------------------


class TestValidPayload:
    def test_outerwear_optional(self):
        outfit = contract_outfit(
            items=[contract_item(c) for c in ("TOP", "BOTTOM", "FOOTWEAR", "OUTERWEAR")]
        )
        assert len(adapt_student4_output(contract_payload(outfit)).outfits[0].items) == 4


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


class TestPrices:
    @pytest.mark.parametrize(
        "minor,currency,expected",
        [
            (1, "SGD", "0.01"),  # two decimal places
            (3990, "JPY", "3990"),  # zero decimal places
            (12345, "KWD", "12.345"),  # three decimal places
        ],
    )
    def test_minor_unit_exponents(self, minor, currency, expected):
        item = contract_item("TOP", price_minor=minor, currency=currency)
        others = [contract_item(c, currency=currency) for c in ("BOTTOM", "FOOTWEAR")]
        price = (
            adapt_student4_output(contract_payload(contract_outfit(items=[item, *others])))
            .outfits[0]
            .items[0]
            .price
        )
        assert price == Decimal(expected) and str(price) == expected

    def test_price_and_price_minor_together_rejected(self):
        item = contract_item("TOP", price="39.90")
        adapter_error(
            contract_payload(
                contract_outfit(items=[item, contract_item("BOTTOM"), contract_item("FOOTWEAR")])
            ),
            "invalid_price",
        )


# ---------------------------------------------------------------------------
# Upstream local image path
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Structural errors
# ---------------------------------------------------------------------------


class TestErrors:
    @pytest.mark.parametrize("status", ["error"])
    def test_status_not_ok(self, status):
        error = adapter_error(contract_payload(status=status), "status_not_ok")
        assert error.location == "status"
        assert isinstance(error, ValueError)

    def test_duplicate_outfit_id(self):
        error = adapter_error(
            contract_payload(contract_outfit("o1"), contract_outfit("o1")), "duplicate_outfit_id"
        )
        assert error.location == "outfits[1].outfit_id"


# ---------------------------------------------------------------------------
# Purity and metadata
# ---------------------------------------------------------------------------


class TestPurityAndMetadata:
    def test_payload_not_modified(self):
        payload = contract_payload(
            contract_outfit(
                items=[
                    contract_item("TOP", image_path="product_images/t.jpg"),
                    contract_item("BOTTOM"),
                    contract_item("FOOTWEAR"),
                ]
            )
        )
        snapshot = copy.deepcopy(payload)
        adapt_student4_output(payload)
        assert payload == snapshot

    def test_metadata_kept_outside_candidate_outfit(self):
        result = adapt_student4_output(contract_payload())
        assert result.model_version == "contract-fixture-version"
        assert result.source == "student4-contract-fixture"
        dumped = result.outfits[0].model_dump()
        assert "model_version" not in dumped and "contract-fixture-version" not in str(dumped)
        assert set(dumped) == {"outfit_id", "items", "compatibility_score"}


# ---------------------------------------------------------------------------
# Category-keyed `products` outfits (Student 4 compatibility_service output shape)
# ---------------------------------------------------------------------------
#
# CONTRACT FIXTURES modelled on the product objects that Student 4 copies from Student 3's
# Top-K output (`bundle.top_k_products`). All values are made up; the embeddings are tiny
# stand-ins, not real FashionCLIP vectors.


def student4_product(role, **overrides):
    """Contract fixture product in the Student 3 / Student 4 product-object format."""
    key = role.lower()
    product = {
        "product_id": f"ebay:fixture:{key}",
        "title": f"Contract fixture {key}",
        "raw_category": role,
        "description": f"{key} description",
        "source_type": "ECOMMERCE_API",
        "price": {"amount": 64.0, "currency": "SGD", "is_synthetic": False},
        "sizes": ["M", "L"],
        "availability": "IN_STOCK",
        "image_path": f"product_images/{key}.jpg",
        "product_url": f"https://www.example.com/itm/{key}",
        "evidence": {
            "price_verified": True,
            "availability_verified": True,
            "sizes_verified": True,
            "image_verified": True,
            "notes": ["contract fixture"],
        },
        "category": role,
        "colors": ["black"],
        "provenance": {
            "source": "ebay:fixture:EBAY_SG",
            "record_index": 1,
            "original": {
                "api_item": {"image": {"imageUrl": f"https://img.example.com/{key}.jpg"}},
                "fetched_at": "2026-10-19T00:00:00+00:00",
            },
        },
        "owned": False,
        "ownership_source": None,
        "additional_purchase_required": True,
        "embedding": {"model": "fixture", "values": [0.1, 0.2, 0.3]},
        "garment_embedding": {"model": "fixture", "modality": "image", "values": [0.1, 0.2]},
        "product_score": 0.4,
    }
    product.update(overrides)
    return product


def products_outfit(outfit_id="O0001", score=0.32, products=None, **product_overrides):
    """Contract fixture outfit with a category-keyed `products` object."""
    if products is None:
        products = {
            cat: student4_product(cat, **product_overrides) for cat in ("TOP", "BOTTOM", "FOOTWEAR")
        }
    return {"outfit_id": outfit_id, "compatibility_score": score, "products": products}


def adapted_items(outfit, **kwargs):
    result = adapt_student4_output(contract_payload(outfit), **kwargs)
    return {item.category: item for item in result.outfits[0].items}


def products_error(outfit, code, **kwargs):
    with pytest.raises(Student4AdapterError) as exc:
        adapt_student4_output(contract_payload(outfit), **kwargs)
    assert exc.value.code == code
    return exc.value


class TestCategoryKeyedProducts:
    def test_products_dict_becomes_items_list(self):
        result = adapt_student4_output(contract_payload(products_outfit()))
        [outfit] = result.outfits
        assert isinstance(outfit, CandidateOutfit)
        assert (outfit.outfit_id, outfit.compatibility_score) == ("O0001", 0.32)
        assert [item.category for item in outfit.items] == [
            ProductCategory.TOP,
            ProductCategory.BOTTOM,
            ProductCategory.FOOTWEAR,
        ]
        assert all(isinstance(item, Product) for item in outfit.items)

    def test_price_amount_and_currency(self):
        items = adapted_items(
            products_outfit(price={"amount": 39.9, "currency": "SGD", "is_synthetic": False})
        )
        assert items[ProductCategory.TOP].price == Decimal("39.9")
        assert items[ProductCategory.TOP].currency == "SGD"

    def test_sizes_become_available_sizes(self):
        items = adapted_items(products_outfit(sizes=["S", "M"]))
        assert items[ProductCategory.TOP].available_sizes == ["S", "M"]

    def test_source_type_becomes_source(self):
        items = adapted_items(products_outfit())
        assert items[ProductCategory.TOP].source == "ECOMMERCE_API"

    def test_missing_source_type_falls_back_to_provenance_source(self):
        outfit = products_outfit()
        del outfit["products"]["TOP"]["source_type"]
        items = adapted_items(outfit)
        assert items[ProductCategory.TOP].source == "ebay:fixture:EBAY_SG"

    def test_owned_becomes_is_user_owned(self):
        outfit = products_outfit()
        outfit["products"]["TOP"]["owned"] = True
        items = adapted_items(outfit)
        assert items[ProductCategory.TOP].is_user_owned is True
        assert items[ProductCategory.BOTTOM].is_user_owned is False

    def test_missing_required_role_rejected_by_candidate_outfit(self):
        products = {cat: student4_product(cat) for cat in ("TOP", "BOTTOM")}
        products_error(products_outfit(products=products), "invalid_outfit")

    def test_api_image_url_preferred_over_image_path(self):
        items = adapted_items(products_outfit())
        assert items[ProductCategory.TOP].image_url == "https://img.example.com/top.jpg"

    def test_missing_api_image_url_falls_back_to_image_path(self):
        outfit = products_outfit()
        del outfit["products"]["TOP"]["provenance"]["original"]["api_item"]["image"]
        items = adapted_items(outfit)
        assert items[ProductCategory.TOP].image_url == "product_images/top.jpg"

    def test_extra_fields_are_dropped(self):
        result = adapt_student4_output(contract_payload(products_outfit()))
        top = result.outfits[0].items[0]
        dumped = top.model_dump()
        for key in (
            "embedding",
            "garment_embedding",
            "provenance",
            "evidence",
            "raw_category",
            "colors",
            "sizes",
            "owned",
            "source_type",
            "image_path",
            "additional_purchase_required",
        ):
            assert key not in dumped


class TestAvailabilityMapping:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("IN_STOCK", True),
            ("AVAILABLE", True),
            ("OUT_OF_STOCK", False),
            ("UNAVAILABLE", False),
        ],
    )
    def test_verified_known_values(self, value, expected):
        result = adapt_student4_output(contract_payload(products_outfit(availability=value)))
        assert all(item.availability is expected for item in result.outfits[0].items)

    @pytest.mark.parametrize("value", ["UNKNOWN"])
    def test_unknown_is_rejected(self, value):
        outfit = products_outfit()
        outfit["products"]["TOP"]["availability"] = value
        error = products_error(outfit, "availability_unknown")
        assert error.location == "outfits[0].products.TOP.availability"

    @pytest.mark.parametrize("value", [False, "true"])
    @pytest.mark.parametrize("availability", ["IN_STOCK"])
    def test_known_status_needs_availability_verified_true(self, availability, value):
        outfit = products_outfit(availability=availability)
        outfit["products"]["BOTTOM"]["evidence"]["availability_verified"] = value
        error = products_error(outfit, "availability_unverified")
        assert error.location == "outfits[0].products.BOTTOM.evidence.availability_verified"


class TestSizeEvidence:
    def test_sizes_verified_true_makes_size_data_reliable(self):
        items = adapted_items(products_outfit())
        assert items[ProductCategory.TOP].size_data_reliable is True


class TestProductScoreSources:
    """`product_score_by_id` is a TEMPORARY adapter-only lookup; the service never passes it."""

    def _without_scores(self):
        outfit = products_outfit()
        for product in outfit["products"].values():
            del product["product_score"]
        return outfit

    def test_product_score_on_the_product_is_used(self):
        items = adapted_items(products_outfit(product_score=0.61))
        assert items[ProductCategory.TOP].product_score == 0.61

    def test_product_score_on_the_product_wins_over_the_lookup(self):
        items = adapted_items(
            products_outfit(product_score=0.61),
            product_score_by_id={"ebay:fixture:top": 0.11},
        )
        assert items[ProductCategory.TOP].product_score == 0.61

    def test_lookup_by_product_id(self):
        scores = {"ebay:fixture:top": 0.4, "ebay:fixture:bottom": 0.3, "ebay:fixture:footwear": 0.2}
        items = adapted_items(self._without_scores(), product_score_by_id=scores)
        assert [items[c].product_score for c in sorted(items)] == [
            scores["ebay:fixture:bottom"],
            scores["ebay:fixture:footwear"],
            scores["ebay:fixture:top"],
        ]

    def test_missing_product_score_is_an_error_not_a_guess(self):
        error = products_error(self._without_scores(), "missing_product_score")
        assert error.location == "outfits[0].products.TOP.product_score"

    def test_compatibility_score_is_never_used_as_product_score(self):
        outfit = self._without_scores()
        outfit["compatibility_score"] = 0.9
        products_error(outfit, "missing_product_score")

    def test_lookup_missing_one_product_is_an_error(self):
        error = products_error(
            self._without_scores(),
            "missing_product_score",
            product_score_by_id={"ebay:fixture:top": 0.4, "ebay:fixture:bottom": 0.3},
        )
        assert error.location == "outfits[0].products.FOOTWEAR.product_score"

    @pytest.mark.parametrize("value", [1.5])
    def test_invalid_looked_up_score_rejected(self, value):
        scores = {
            "ebay:fixture:top": value,
            "ebay:fixture:bottom": 0.3,
            "ebay:fixture:footwear": 0.2,
        }
        products_error(self._without_scores(), "invalid_product_score", product_score_by_id=scores)


class TestLegacyItemsStayStrict:
    def test_legacy_items_still_reject_unknown_fields(self):
        items = [contract_item(cat) for cat in ("TOP", "BOTTOM", "FOOTWEAR")]
        items[0]["embedding"] = [0.1]
        adapter_error(contract_payload(contract_outfit(items=items)), "invalid_item")


# ---------------------------------------------------------------------------
# One representative per independent adapter branch
# ---------------------------------------------------------------------------


def _top(outfit):
    return outfit["products"]["TOP"]


def _set_top(**changes):
    def change(outfit):
        _top(outfit).update(changes)

    return change


@pytest.mark.parametrize(
    "payload, kwargs, code",
    [
        ("x", {}, "invalid_payload"),
        (None, {"product_score_by_id": [0.1]}, "invalid_product_score_by_id"),
    ],
    ids=["payload_not_object", "lookup_not_object"],
)
def test_adapter_rejects_bad_top_level_arguments(payload, kwargs, code):
    with pytest.raises(Student4AdapterError) as exc:
        adapt_student4_output(contract_payload() if payload is None else payload, **kwargs)
    assert exc.value.code == code


@pytest.mark.parametrize(
    "change, code",
    [
        (lambda i: i.update(currency=" "), "invalid_price"),
        (lambda i: i.update(price_minor=True), "invalid_price"),
    ],
    ids=["price_minor_blank_currency", "price_minor_bool"],
)
def test_legacy_item_price_minor_rejected(change, code):
    items = [contract_item(c) for c in ("TOP", "BOTTOM", "FOOTWEAR")]
    change(items[0])
    adapter_error(contract_payload(contract_outfit(items=items)), code)


def test_legacy_item_not_an_object_rejected():
    items = ["x", contract_item("BOTTOM"), contract_item("FOOTWEAR")]
    adapter_error(contract_payload(contract_outfit(items=items)), "invalid_item")


def test_legacy_item_with_plain_price_and_empty_image_path():
    top = contract_item("TOP", price="39.90", image_path="")
    del top["price_minor"]
    items = [top, contract_item("BOTTOM"), contract_item("FOOTWEAR")]
    [outfit] = adapt_student4_output(contract_payload(contract_outfit(items=items))).outfits
    assert outfit.items[0].price == Decimal("39.90")
    assert outfit.items[0].image_url == "https://img.example.com/top.jpg"


def test_products_numeric_price_and_price_minor():
    outfit = products_outfit()
    _top(outfit).update(price=39.9, currency="SGD")
    bottom = outfit["products"]["BOTTOM"]
    del bottom["price"]
    bottom.update(price_minor=3990, currency="SGD")
    items = adapted_items(outfit)
    assert items[ProductCategory.TOP].price == Decimal("39.9")
    assert items[ProductCategory.BOTTOM].price == Decimal("39.90")


def test_products_given_as_product_fields_are_kept():
    outfit = products_outfit()
    _top(outfit).update(
        available_sizes=["XS"], source="given_source", size_data_reliable=False, availability=True
    )
    bottom = outfit["products"]["BOTTOM"]
    for key in ("owned", "image_path", "source_type"):
        del bottom[key]
    bottom["provenance"] = {"source": "provenance_source"}  # no API image either
    del bottom["evidence"]["sizes_verified"]
    items = adapted_items(outfit)
    top, bottom = items[ProductCategory.TOP], items[ProductCategory.BOTTOM]
    assert (top.available_sizes, top.source, top.size_data_reliable) == (
        ["XS"],
        "given_source",
        False,
    )
    assert top.availability is True
    assert (bottom.source, bottom.image_url, bottom.is_user_owned) == (
        "provenance_source",
        None,
        False,
    )
    assert bottom.size_data_reliable is False


def test_source_argument_overrides_payload_source():
    result = adapt_student4_output(contract_payload(), source="override")
    assert result.source == "override"
