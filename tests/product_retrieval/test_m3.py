import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from travel_agent.products.retrieval.catalogue import PreparedCatalogueProvider
from travel_agent.products.retrieval.m3 import NormalizedProduct, process_catalogue, process_records
from travel_agent.products.retrieval.m3.pipeline import category

ROOT = Path(
    __import__("os").environ.get("STUDENT3_DATA_ROOT", str(Path(__file__).resolve().parents[2]))
)


def row(**changes):
    return dict(product_id="a", title=" Shirt ", raw_category=" T-Shirt ", **changes)


def test_clean_attributes_and_preserve_input():
    data = row(
        colors=[" Grey ", "gray", "黑色"],
        sizes=["medium", "M", " uk 8 "],
        price={"amount": "25.50", "currency": " sgd ", "is_synthetic": True},
    )
    before = copy.deepcopy(data)
    result = process_records([data], source="fixture")
    p = result.products[0]
    assert p.title == "Shirt" and p.category.value == "TOP"
    assert p.colors == ["gray", "black"] and p.sizes == ["M", "UK 8"]
    assert p.price.amount == 25.5 and p.price.currency == "SGD"
    assert p.availability.value == "UNKNOWN"
    assert p.provenance.original == before and data == before
    NormalizedProduct.model_validate_json(p.model_dump_json())


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("jeans", "BOTTOM"),
        ("boots", "FOOTWEAR"),
        ("coat", "OUTERWEAR"),
        ("dress", "UNKNOWN"),
        ("shirt jacket", "UNKNOWN"),
    ],
)
def test_categories(raw, expected):
    assert category(raw).value == expected


@pytest.mark.parametrize(
    "changes",
    [
        dict(title=" "),
        dict(product_id=" "),
        dict(price={"amount": -1, "currency": "SGD"}),
        dict(price={"amount": "NaN", "currency": "SGD"}),
        dict(price={"amount": True, "currency": "SGD"}),
        dict(price={"amount": 1}),
        dict(sizes=[2]),
        dict(extra=1),
        dict(availability="probably"),
        dict(availability="IN_STOCK"),
    ],
)
def test_rejections(changes):
    data = row()
    data.update(changes)
    result = process_records([data])
    assert not result.products and result.issues


def test_missing_and_duplicates():
    a = row()
    a.pop("raw_category")
    result = process_records([{}, a, a, 42, row(product_url="https://example.com")])
    assert len(result.products) == 1 and result.products[0].category.value == "UNKNOWN"
    assert {i.record_index for i in result.issues} == {1, 2, 3, 4, 5}
    assert process_records([]).input_count == 0


def test_first_valid_and_source_identity():
    bad = row()
    bad["title"] = ""
    result = process_records([bad, row(), row(source_type="WARDROBE")])
    assert len(result.products) == 2


def test_stock_evidence():
    p = process_records([row(source_type="ECOMMERCE_API", availability="in stock")]).products[0]
    assert p.availability.value == "UNKNOWN"
    p = process_records(
        [
            row(
                source_type="ECOMMERCE_API",
                availability="out of stock",
                evidence={"availability_verified": True},
            )
        ]
    ).products[0]
    assert p.availability.value == "OUT_OF_STOCK"


def test_minimum_image_and_text_requirements():
    p = process_records([row()]).products[0]
    assert not p.image_embedding_candidate and p.price is None
    data = p.model_dump()
    data["image_embedding_candidate"] = True
    with pytest.raises(ValidationError):
        NormalizedProduct.model_validate(data)
    p = process_records(
        [
            row(
                source_type="WARDROBE",
                image_path="uninspected.jpg",
                evidence={"image_verified": True},
            )
        ]
    ).products[0]
    assert p.image_embedding_candidate


@pytest.mark.parametrize("name", ["catalogue.json", "catalogue.csv"])
def test_m2_adapter(name):
    provider = PreparedCatalogueProvider(ROOT / "data" / name)
    before = [g.model_dump() for g in provider.products]
    result = process_catalogue(ROOT / "data" / name)
    assert result.input_count == 5 and len(result.products) == 5
    assert [g.model_dump() for g in provider.products] == before
    assert all(
        p.availability.value == "UNKNOWN" and not p.image_embedding_candidate
        for p in result.products
    )
    assert [p.provenance.record_index for p in result.products] == [1, 2, 3, 4, 5]


def test_m2_errors_keep_indices(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{}, row(), row(), {**row(), "product_id": "b"}]))
    result = process_catalogue(path)
    assert result.input_count == 4
    assert [p.provenance.record_index for p in result.products] == [2, 4]
    assert [i.record_index for i in result.issues if i.stage == "M2"] == [1, 3]


def test_cli(tmp_path):
    output = tmp_path / "output.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "travel_agent.products.retrieval.m3",
            "--catalogue",
            str(ROOT / "data/catalogue.json"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and len(json.loads(output.read_text())["products"]) == 5
    fail = subprocess.run(
        [
            sys.executable,
            "-m",
            "travel_agent.products.retrieval.m3",
            "--input",
            str(tmp_path / "missing"),
        ],
        capture_output=True,
        text=True,
    )
    assert fail.returncode == 2
