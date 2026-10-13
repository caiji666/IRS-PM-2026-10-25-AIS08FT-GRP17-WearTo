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
    TripShoppingRequest,
)
from travel_agent.recommendation.student5.orchestrator import Student5Orchestrator
from travel_agent.recommendation.student5.student4_adapter import (
    Student4AdapterError,
    Student4AdapterResult,
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
    def test_converts_to_candidate_outfits(self):
        result = adapt_student4_output(contract_payload())
        assert isinstance(result, Student4AdapterResult)
        [outfit] = result.outfits
        assert isinstance(outfit, CandidateOutfit)
        assert all(isinstance(item, Product) for item in outfit.items)

    def test_outfit_id_and_score_kept(self):
        result = adapt_student4_output(
            contract_payload(
                contract_outfit("outfit_001", 0.86), contract_outfit("outfit_002", 0.42)
            )
        )
        assert [(o.outfit_id, o.compatibility_score) for o in result.outfits] == [
            ("outfit_001", 0.86),
            ("outfit_002", 0.42),
        ]

    @pytest.mark.parametrize("score", [0, 0.0, 1, 1.0])
    def test_boundary_scores_accepted(self, score):
        assert (
            adapt_student4_output(contract_payload(contract_outfit(score=score)))
            .outfits[0]
            .compatibility_score
            == score
        )

    def test_product_fields_preserved(self):
        item = contract_item(
            "TOP",
            description=None,
            brand="B",
            colour="navy",
            material="wool",
            embedding_ref="emb/top",
        )
        outfit = contract_outfit(
            items=[
                item,
                contract_item("BOTTOM"),
                contract_item("FOOTWEAR", is_user_owned=True, product_url=None),
            ]
        )
        top, bottom, footwear = adapt_student4_output(contract_payload(outfit)).outfits[0].items
        assert (
            top.product_id,
            top.category,
            top.source,
            top.title,
            top.description,
            top.product_url,
            top.image_url,
            top.currency,
            top.availability,
            top.product_score,
            top.available_sizes,
            top.size_data_reliable,
            top.is_user_owned,
        ) == (
            "p_top",
            ProductCategory.TOP,
            "ebay",
            "Contract fixture top",
            None,
            "https://shop.example.com/top",
            "https://img.example.com/top.jpg",
            "SGD",
            True,
            0.5,
            ["M", "L"],
            True,
            False,
        )
        assert (top.brand, top.colour, top.material, top.embedding_ref) == (
            "B",
            "navy",
            "wool",
            "emb/top",
        )
        assert footwear.is_user_owned is True

    def test_outerwear_optional(self):
        outfit = contract_outfit(
            items=[contract_item(c) for c in ("TOP", "BOTTOM", "FOOTWEAR", "OUTERWEAR")]
        )
        assert len(adapt_student4_output(contract_payload(outfit)).outfits[0].items) == 4

    def test_order_of_outfits_preserved(self):
        ids = ["z", "a", "m"]
        result = adapt_student4_output(contract_payload(*(contract_outfit(i) for i in ids)))
        assert [o.outfit_id for o in result.outfits] == ids


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


class TestPrices:
    def test_price_minor_converted_exactly(self):
        item = contract_item("TOP", price_minor=3990)
        price = (
            adapt_student4_output(
                contract_payload(
                    contract_outfit(
                        items=[item, contract_item("BOTTOM"), contract_item("FOOTWEAR")]
                    )
                )
            )
            .outfits[0]
            .items[0]
            .price
        )
        assert price == Decimal("39.90")
        assert isinstance(price, Decimal)
        assert str(price) == "39.90"

    @pytest.mark.parametrize(
        "minor,currency,expected",
        [
            (1, "SGD", "0.01"),
            (0, "SGD", "0.00"),
            (10, "usd", "0.10"),
            (3990, "JPY", "3990"),
            (12345, "KWD", "12.345"),
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

    def test_existing_price_validated_not_converted(self):
        item = contract_item("TOP", price="39.90")
        del item["price_minor"]
        price = (
            adapt_student4_output(
                contract_payload(
                    contract_outfit(
                        items=[item, contract_item("BOTTOM"), contract_item("FOOTWEAR")]
                    )
                )
            )
            .outfits[0]
            .items[0]
            .price
        )
        assert price == Decimal("39.90")

    @pytest.mark.parametrize("value", [39.9, "3990", -1, True, None])
    def test_invalid_price_minor_rejected(self, value):
        item = contract_item("TOP", price_minor=value)
        error = adapter_error(
            contract_payload(
                contract_outfit(items=[item, contract_item("BOTTOM"), contract_item("FOOTWEAR")])
            ),
            "invalid_price",
        )
        assert error.location == "outfits[0].items[0].price_minor"

    def test_price_and_price_minor_together_rejected(self):
        item = contract_item("TOP", price="39.90")
        adapter_error(
            contract_payload(
                contract_outfit(items=[item, contract_item("BOTTOM"), contract_item("FOOTWEAR")])
            ),
            "invalid_price",
        )

    def test_missing_price_rejected_by_product(self):
        item = contract_item("TOP")
        del item["price_minor"]
        adapter_error(
            contract_payload(
                contract_outfit(items=[item, contract_item("BOTTOM"), contract_item("FOOTWEAR")])
            ),
            "invalid_item",
        )

    def test_price_minor_needs_currency(self):
        item = contract_item("TOP")
        del item["currency"]
        adapter_error(
            contract_payload(
                contract_outfit(items=[item, contract_item("BOTTOM"), contract_item("FOOTWEAR")])
            ),
            "invalid_price",
        )


# ---------------------------------------------------------------------------
# Upstream local image path
# ---------------------------------------------------------------------------


class TestImagePath:
    def test_image_path_preferred_over_image_url(self):
        item = contract_item("TOP", image_path="product_images/p_top.jpg")
        top = (
            adapt_student4_output(
                contract_payload(
                    contract_outfit(
                        items=[item, contract_item("BOTTOM"), contract_item("FOOTWEAR")]
                    )
                )
            )
            .outfits[0]
            .items[0]
        )
        assert top.image_url == "product_images/p_top.jpg"

    @pytest.mark.parametrize("value", [None, ""])
    def test_empty_image_path_keeps_image_url(self, value):
        item = contract_item("TOP", image_path=value)
        top = (
            adapt_student4_output(
                contract_payload(
                    contract_outfit(
                        items=[item, contract_item("BOTTOM"), contract_item("FOOTWEAR")]
                    )
                )
            )
            .outfits[0]
            .items[0]
        )
        assert top.image_url == "https://img.example.com/top.jpg"


# ---------------------------------------------------------------------------
# Structural errors
# ---------------------------------------------------------------------------


class TestErrors:
    @pytest.mark.parametrize("status", ["error", "OK", None, ""])
    def test_status_not_ok(self, status):
        error = adapter_error(contract_payload(status=status), "status_not_ok")
        assert error.location == "status"
        assert isinstance(error, ValueError)

    def test_status_missing(self):
        payload = contract_payload()
        del payload["status"]
        adapter_error(payload, "status_not_ok")

    @pytest.mark.parametrize("outfits", [[], None, "x", {"outfit_id": "o"}])
    def test_outfits_missing_or_empty(self, outfits):
        payload = contract_payload()
        payload["outfits"] = outfits
        assert adapter_error(payload, "no_outfits").location == "outfits"

    def test_outfits_key_missing(self):
        payload = contract_payload()
        del payload["outfits"]
        adapter_error(payload, "no_outfits")

    def test_payload_not_mapping(self):
        adapter_error([contract_outfit()], "invalid_payload")

    def test_outfit_not_mapping(self):
        error = adapter_error(contract_payload(contract_outfit(), "oops"), "invalid_outfit")
        assert error.location == "outfits[1]"

    @pytest.mark.parametrize("value", [None, "", "   ", 7])
    def test_missing_or_blank_outfit_id(self, value):
        outfit = contract_outfit()
        outfit["outfit_id"] = value
        assert adapter_error(contract_payload(outfit), "invalid_outfit_id").location == (
            "outfits[0].outfit_id"
        )

    def test_outfit_id_key_missing(self):
        outfit = contract_outfit()
        del outfit["outfit_id"]
        adapter_error(contract_payload(outfit), "invalid_outfit_id")

    def test_duplicate_outfit_id(self):
        error = adapter_error(
            contract_payload(contract_outfit("o1"), contract_outfit("o1")), "duplicate_outfit_id"
        )
        assert error.location == "outfits[1].outfit_id"

    @pytest.mark.parametrize("score", [-0.01, 1.01, float("nan"), float("inf"), "0.5", True])
    def test_invalid_score(self, score):
        error = adapter_error(
            contract_payload(contract_outfit(score=score)), "invalid_compatibility_score"
        )
        assert error.location == "outfits[0].compatibility_score"

    def test_missing_score(self):
        outfit = contract_outfit()
        del outfit["compatibility_score"]
        adapter_error(contract_payload(outfit), "invalid_compatibility_score")

    @pytest.mark.parametrize("items", [None, [], "x"])
    def test_items_missing(self, items):
        adapter_error(contract_payload(contract_outfit(items=items)), "invalid_items")

    def test_item_not_mapping(self):
        error = adapter_error(
            contract_payload(
                contract_outfit(items=[contract_item("TOP"), "x", contract_item("FOOTWEAR")])
            ),
            "invalid_item",
        )
        assert error.location == "outfits[0].items[1]"

    @pytest.mark.parametrize(
        "change",
        [
            {"category": "HAT"},
            {"product_score": 1.5},
            {"availability": "yes"},
            {"title": ""},
            {"unexpected_field": 1},
        ],
    )
    def test_item_rejected_by_product_contract(self, change):
        bad = {**contract_item("TOP"), **change}
        error = adapter_error(
            contract_payload(
                contract_outfit(items=[bad, contract_item("BOTTOM"), contract_item("FOOTWEAR")])
            ),
            "invalid_item",
        )
        assert error.location == "outfits[0].items[0]"

    @pytest.mark.parametrize(
        "categories",
        [
            ("TOP", "BOTTOM"),
            ("TOP", "FOOTWEAR"),
            ("BOTTOM", "FOOTWEAR"),
            ("TOP", "TOP", "BOTTOM", "FOOTWEAR"),
        ],
    )
    def test_missing_or_duplicate_roles_rejected_by_candidate_outfit(self, categories):
        items = [contract_item(c, product_id=f"p{n}") for n, c in enumerate(categories)]
        error = adapter_error(contract_payload(contract_outfit(items=items)), "invalid_outfit")
        assert error.location == "outfits[0]"


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

    def test_deterministic(self):
        assert adapt_student4_output(contract_payload()) == adapt_student4_output(
            contract_payload()
        )

    def test_metadata_kept_outside_candidate_outfit(self):
        result = adapt_student4_output(contract_payload())
        assert result.model_version == "contract-fixture-version"
        assert result.source == "student4-contract-fixture"
        dumped = result.outfits[0].model_dump()
        assert "model_version" not in dumped and "contract-fixture-version" not in str(dumped)
        assert set(dumped) == {"outfit_id", "items", "compatibility_score"}

    def test_metadata_defaults(self):
        payload = contract_payload()
        del payload["model_version"], payload["source"]
        result = adapt_student4_output(payload)
        assert result.model_version is None
        assert result.source == "student4"

    def test_source_argument_overrides_default(self):
        payload = contract_payload()
        del payload["source"]
        assert adapt_student4_output(payload, source="student4-v2").source == "student4-v2"

    @pytest.mark.parametrize("field,value", [("model_version", 3), ("source", "")])
    def test_invalid_metadata_rejected(self, field, value):
        adapter_error(contract_payload(**{field: value}), "invalid_metadata")

    def test_result_is_frozen(self):
        result = adapt_student4_output(contract_payload())
        with pytest.raises(AttributeError):
            result.model_version = "changed"

    def test_metadata_does_not_affect_scoring(self):
        request = TripShoppingRequest.model_validate(
            {
                "destination": "Seoul",
                "start_date": "2026-12-20",
                "end_date": "2026-12-27",
                "budget": 500,
                "currency": "SGD",
            }
        )
        a = adapt_student4_output(contract_payload(model_version="v1", source="s1"))
        b = adapt_student4_output(contract_payload(model_version="v2", source="s2"))
        run_a = Student5Orchestrator(beta=0.5).run(request=request, candidate_outfits=a.outfits)
        run_b = Student5Orchestrator(beta=0.5).run(request=request, candidate_outfits=b.outfits)
        assert run_a.recommendation_set == run_b.recommendation_set
