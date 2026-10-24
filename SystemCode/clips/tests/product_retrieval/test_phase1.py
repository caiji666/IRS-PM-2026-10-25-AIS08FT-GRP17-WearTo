import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from travel_agent.products.retrieval.catalogue import CatalogueError, PreparedCatalogueProvider
from travel_agent.products.retrieval.schemas import (
    Availability,
    Category,
    Embedding,
    ExpandedFashionContext,
    Garment,
    ProductRetrievalRequest,
    ProductRetrievalResponse,
    RawProduct,
    Trip,
    WardrobeUpload,
)

ROOT = Path(
    __import__("os").environ.get("STUDENT3_DATA_ROOT", str(Path(__file__).resolve().parents[2]))
)


def req(**changes):
    return ProductRetrievalRequest(
        context=ExpandedFashionContext(trip=Trip(destination="Singapore")), **changes
    )


def write(tmp_path, records):
    p = tmp_path / "catalogue.json"
    p.write_text(json.dumps(records))
    return p


def test_fixtures():
    for name in ("request.json", "filter_request.json", "integration_request.json"):
        ProductRetrievalRequest.model_validate_json((ROOT / "fixtures" / name).read_text())
    Garment.model_validate_json((ROOT / "fixtures/wardrobe_garment.json").read_text())


@pytest.mark.parametrize("name", ["catalogue.json", "catalogue.csv"])
def test_load_and_categories(name):
    provider = PreparedCatalogueProvider(ROOT / "data" / name)
    response = provider.retrieve(req())
    assert len(response.products) == 5
    assert {g.category for g in response.products} == set(Category)
    assert not response.issues
    assert all(
        g.availability == Availability.UNKNOWN and g.product_url is None for g in response.products
    )
    ProductRetrievalResponse.model_validate_json(response.model_dump_json())


def test_filters():
    p = PreparedCatalogueProvider(ROOT / "data/catalogue.json")
    assert len(p.retrieve(req(query="WHITE", categories=[Category.TOP])).products) == 1
    assert len(p.retrieve(req(size="m", currency="SGD", max_price=30)).products) == 1
    assert not p.retrieve(req(currency="USD")).products
    assert not p.retrieve(req(max_price=20, currency="SGD")).products
    response = p.retrieve(req(limit=2))
    assert len(response.products) == 2 and response.total_matches == 5


@pytest.mark.parametrize(
    "change",
    [
        dict(availability="IN_STOCK"),
        dict(product_url="https://example.com/item"),
        dict(price={"amount": -1, "currency": "SGD"}),
        dict(price={"amount": 25, "currency": "SGD", "is_synthetic": False}),
        dict(evidence={"availability_verified": True}),
        dict(unexpected=True),
    ],
)
def test_invalid_contract(change):
    with pytest.raises(ValidationError):
        RawProduct(product_id="a", title="A", raw_category="shirt", **change)


def test_request_validation():
    with pytest.raises(ValidationError):
        req(max_price=10)
    with pytest.raises(ValidationError):
        req(limit=0)
    with pytest.raises(ValidationError):
        Trip(destination="X", start_date="2026-10-10", end_date="2026-10-09")
    with pytest.raises(ValidationError):
        Embedding(model="transport", values=[float("nan")])


def test_duplicates_and_bad_records(tmp_path):
    valid = dict(product_id="a", title="A", raw_category="unknown")
    p = PreparedCatalogueProvider(write(tmp_path, [{}, valid, valid, 42]))
    assert len(p.products) == 1 and len(p.issues) == 3
    assert [i.record_index for i in p.issues] == [1, 3, 4]


@pytest.mark.parametrize("content", ["{bad", "{}"])
def test_bad_file(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(CatalogueError):
        PreparedCatalogueProvider(path)


def test_missing_file(tmp_path):
    with pytest.raises(CatalogueError):
        PreparedCatalogueProvider(tmp_path / "missing.json")


def test_csv_bad_cell(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("product_id,title,raw_category,sizes\na,A,shirt,invalid-json\n")
    assert len(PreparedCatalogueProvider(path).issues) == 1


def test_passthrough():
    g = Garment.model_validate_json((ROOT / "fixtures/wardrobe_garment.json").read_text())
    r = req(
        wardrobe_garments=[g],
        wardrobe_uploads=[WardrobeUpload(upload_id="u", image_path="missing.jpg")],
        context_embedding=Embedding(model="external", values=[1, 2]),
    )
    result = PreparedCatalogueProvider(ROOT / "data/catalogue.json").retrieve(r)
    assert result.wardrobe_garments == [g] and len(result.pending_wardrobe_uploads) == 1


def test_cli(tmp_path):
    import subprocess
    import sys

    success = subprocess.run(
        [
            sys.executable,
            "-m",
            "travel_agent.products.retrieval.cli",
            "--catalogue",
            str(ROOT / "data/catalogue.csv"),
            "--request",
            str(ROOT / "fixtures/filter_request.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert success.returncode == 0 and len(json.loads(success.stdout)["products"]) == 1
    failure = subprocess.run(
        [
            sys.executable,
            "-m",
            "travel_agent.products.retrieval.cli",
            "--catalogue",
            str(tmp_path / "missing.json"),
            "--request",
            str(ROOT / "fixtures/request.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert failure.returncode == 2 and "Error:" in failure.stderr


def test_version_and_response_fixture():
    with pytest.raises(ValidationError):
        req(schema_version="2.0")
    ProductRetrievalResponse.model_validate_json((ROOT / "fixtures/response.json").read_text())
