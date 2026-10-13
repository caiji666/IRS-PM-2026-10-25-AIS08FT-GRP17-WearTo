"""Deterministic unit doubles live only here; opt-in smoke uses actual weights."""

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from travel_agent.products.retrieval.m3 import process_catalogue, process_records
from travel_agent.products.retrieval.m4.wardrobe import WardrobeAnalyzer, predict
from travel_agent.products.retrieval.m5 import (
    CachedEncoder,
    FashionCLIPEncoder,
    build_index,
    cosine,
    vector,
)
from travel_agent.products.retrieval.m5.images import load_image
from travel_agent.products.retrieval.m5.models import ProductIndex, Vector
from travel_agent.products.retrieval.m6 import (
    RankedProductProvider,
    RankingConfig,
    RankingRequest,
    rank,
)
from travel_agent.products.retrieval.schemas import (
    Embedding,
    ExpandedFashionContext,
    ProductRetrievalRequest,
    ProductRetrievalResponse,
    Trip,
    WardrobeUpload,
)

ROOT = Path(
    __import__("os").environ.get("STUDENT3_DATA_ROOT", str(Path(__file__).resolve().parents[2]))
)


class FakeEncoder:
    identity = "test-only-v1"
    dimension = 3

    def __init__(self):
        self.text_calls = 0
        self.image_calls = 0

    def encode_texts(self, texts):
        self.text_calls += len(texts)
        return [
            vector(self.identity, "text", [1, 0, 0] if ("shirt" in t or "red" in t) else [0, 1, 0])
            for t in texts
        ]

    def encode_images(self, images):
        self.image_calls += len(images)
        return [vector(self.identity, "image", [1, 0, 0]) for image in images]


def product(pid="a", **changes):
    r = {"product_id": pid, "title": "white shirt", "raw_category": "shirt"}
    r.update(changes)
    return process_records([r]).products[0]


def index(vectors=None):
    e = FakeEncoder()
    result = build_index([product("a"), product("b", raw_category="pants")], e, ROOT)
    if vectors:
        for i, v in zip(result.items, vectors):
            i.vector = vector(e.identity, "text", v)
    return result


def req(**changes):
    return ProductRetrievalRequest(
        context=ExpandedFashionContext(trip=Trip(destination="Singapore"), query="shirt"), **changes
    )


@pytest.mark.parametrize("values", [[0, 0], [float("nan"), 1], [float("inf"), 1], [[1, 2]]])
def test_invalid_vectors(values):
    with pytest.raises(ValueError):
        vector("x", "text", values)


def test_normalization_and_cosine():
    a = vector("m", "text", [3, 4])
    b = vector("m", "image", [3, 4])
    assert cosine(a, b) == pytest.approx(1)
    assert np.linalg.norm(a.values) == pytest.approx(1)
    assert cosine(a, vector("m", "text", [-3, -4])) == pytest.approx(-1)
    with pytest.raises(ValueError):
        cosine(a, vector("other", "text", [3, 4]))
    with pytest.raises(ValueError):
        cosine(a, vector("m", "text", [1, 0, 0]))
    with pytest.raises(ValidationError):
        Vector(model="m", modality="text", values=[2, 0])


def test_cache_content_model_modality_and_corruption(tmp_path):
    e = FakeEncoder()
    c = CachedEncoder(e, tmp_path)
    a = c.encode_texts(["shirt"])[0]
    c.encode_texts(["shirt"])
    assert c.hits == 1 and e.text_calls == 1
    c.encode_texts(["different"])
    assert c.misses == 2
    c.encode_images([Image.new("RGB", (4, 4), "red")])
    assert e.image_calls == 1
    c.encode_images([Image.new("RGB", (4, 4), "blue")])
    assert e.image_calls == 2
    path = next(
        p
        for p in tmp_path.glob("*.json")
        if json.loads(p.read_text())["values"] == a.values
        and json.loads(p.read_text())["modality"] == "text"
    )
    path.write_text("{corrupt")
    c.encode_texts(["shirt"])
    assert e.text_calls == 3
    e2 = FakeEncoder()
    e2.identity = "test-only-v2"
    CachedEncoder(e2, tmp_path).encode_texts(["shirt"])
    assert e2.text_calls == 1


def test_cache_empty_batch_and_wrong_backend(tmp_path):
    c = CachedEncoder(FakeEncoder(), tmp_path)
    assert c.encode_texts([]) == []

    class Wrong(FakeEncoder):
        def encode_texts(self, texts):
            return [vector("wrong", "text", [1, 0, 0]) for t in texts]

    with pytest.raises(ValueError):
        CachedEncoder(Wrong(), tmp_path).encode_texts(["shirt"])


def test_image_decode_and_path_protection(tmp_path):
    image = tmp_path / "red.png"
    Image.new("RGB", (16, 12), "red").save(image)
    assert load_image("red.png", tmp_path).size == (16, 12)
    with pytest.raises(ValueError):
        load_image("../outside.png", tmp_path)
    outside = tmp_path.parent / "outside.png"
    Image.new("RGB", (4, 4)).save(outside)
    (tmp_path / "link.png").symlink_to(outside)
    with pytest.raises(ValueError):
        load_image("link.png", tmp_path)
    bad = tmp_path / "bad.png"
    bad.write_text("not png")
    with pytest.raises(ValueError):
        load_image(bad, tmp_path)
    with pytest.raises(ValueError):
        load_image("placeholder.svg", tmp_path)


def test_index_prepared_text_fallback():
    e = FakeEncoder()
    products = process_catalogue(ROOT / "data/catalogue.json").products
    before = copy.deepcopy([p.model_dump() for p in products])
    result = build_index(products, e, ROOT)
    assert len(result.items) == 5 and e.image_calls == 0 and e.text_calls == 5
    assert all(i.vector.modality == "text" for i in result.items)
    assert [p.model_dump() for p in products] == before
    ProductIndex.model_validate_json(result.model_dump_json())


def test_index_image_and_error_fallback(tmp_path):
    Image.new("RGB", (16, 16), "red").save(tmp_path / "image.png")
    p = product(source_type="WARDROBE", image_path="image.png", evidence={"image_verified": True})
    e = FakeEncoder()
    result = build_index([p], e, tmp_path)
    assert result.items[0].vector.modality == "image"
    bad = product(
        "bad", source_type="WARDROBE", image_path="missing.png", evidence={"image_verified": True}
    )
    result = build_index([bad], e, tmp_path)
    assert result.items[0].vector.modality == "text" and result.issues


def test_index_compatibility_and_duplicates():
    data = index().model_dump()
    data["dimension"] = 5
    with pytest.raises(ValidationError):
        ProductIndex.model_validate(data)
    data = index().model_dump()
    data["items"].append(data["items"][0])
    with pytest.raises(ValidationError):
        ProductIndex.model_validate(data)


def test_prediction_abstention():
    e = FakeEncoder()
    image = vector(e.identity, "image", [1, 0, 0])
    p = predict(
        image,
        ["a", "b"],
        [vector(e.identity, "text", [1, 0, 0]), vector(e.identity, "text", [1, 0, 0])],
        0.2,
        0.03,
    )
    assert not p.accepted and p.margin == 0
    p = predict(
        image,
        ["a", "b"],
        [vector(e.identity, "text", [1, 0, 0]), vector(e.identity, "text", [0, 1, 0])],
        0.2,
        0.03,
    )
    assert p.accepted and p.label == "a"


def test_wardrobe_reports_and_no_stock_inference(tmp_path):
    Image.new("RGB", (16, 16), "red").save(tmp_path / "image.png")
    report = WardrobeAnalyzer(FakeEncoder(), tmp_path).analyze(
        [
            WardrobeUpload(upload_id="a", image_path="image.png"),
            WardrobeUpload(upload_id="a", image_path="image.png"),
            WardrobeUpload(upload_id="bad", image_path="bad.jpg"),
        ]
    )
    assert len(report.products) == 1 and len(report.issues) == 2
    p = report.products[0]
    assert p.garment.source_type.value == "WARDROBE" and p.garment.availability.value == "UNKNOWN"
    assert (
        p.garment.price is None
        and p.garment.sizes == []
        and not p.garment.evidence.category_verified
    )
    assert p.vector.modality == "image"
    assert WardrobeAnalyzer(FakeEncoder(), tmp_path).analyze([]).products == []
    with pytest.raises(ValueError):
        WardrobeAnalyzer(FakeEncoder(), tmp_path, minimum_margin=-1)


def test_wardrobe_other_and_color_unknown(tmp_path):
    Image.new("RGB", (16, 16)).save(tmp_path / "image.png")

    class Accessory(FakeEncoder):
        def encode_texts(self, texts):
            return [
                vector(self.identity, "text", [1, 0, 0] if "bag, scarf" in t else [0, 1, 0])
                for t in texts
            ]

    r = (
        WardrobeAnalyzer(Accessory(), tmp_path)
        .analyze([WardrobeUpload(upload_id="a", image_path="image.png")])
        .products[0]
    )
    assert r.category_prediction.label == "ACCESSORY" and r.garment.category.value == "UNKNOWN"
    assert r.predicted_colors == []


def test_hybrid_math_and_top_k():
    idx = index([[1, 0, 0], [0, 1, 0]])
    q = RankingRequest(
        context=vector(idx.model, "text", [1, 0, 0]),
        social_references=[vector(idx.model, "image", [0, 1, 0])],
        limit=1,
    )
    result = rank(idx, q)
    assert result.total_matches == 2 and len(result.products) == 1
    assert result.products[0].item.product.product_id == "a"
    assert result.products[0].score == pytest.approx(0.7)
    q.config = RankingConfig(context_weight=0.2, social_weight=0.8)
    assert rank(idx, q).products[0].item.product.product_id == "b"


def test_missing_signal_renormalize_and_social_max():
    idx = index([[1, 0, 0], [0, 1, 0]])
    q = RankingRequest(
        social_references=[
            vector(idx.model, "image", [1, 0, 0]),
            vector(idx.model, "image", [0, 1, 0]),
        ]
    )
    r = rank(idx, q)
    assert r.products[0].score == pytest.approx(1.0)
    assert r.products[0].social_weight == 1 and r.products[0].context_weight == 0
    assert [p.item.product.product_id for p in r.products] == ["a", "b"]
    q = RankingRequest(context=vector(idx.model, "text", [1, 0, 0]))
    assert rank(idx, q).products[0].context_weight == 1


@pytest.mark.parametrize(
    "filters, count",
    [
        ({"categories": ["TOP"]}, 1),
        ({"source_types": ["WARDROBE"]}, 0),
        ({"size": "M"}, 0),
        ({"currency": "USD"}, 0),
        ({"currency": "SGD", "max_price": 20}, 0),
        ({"availability": "IN_STOCK"}, 0),
        ({"availability": "UNKNOWN"}, 2),
    ],
)
def test_filters(filters, count):
    idx = index()
    q = RankingRequest(context=vector(idx.model, "text", [1, 0, 0]), **filters)
    assert rank(idx, q).total_matches == count


def test_filters_exact_currency_and_known_sizes():
    p = product(price={"amount": 25, "currency": "SGD", "is_synthetic": True}, sizes=["M"])
    idx = build_index([p], FakeEncoder(), ROOT)
    q = RankingRequest(
        context=vector(idx.model, "text", [1, 0, 0]), size="m", currency="SGD", max_price=30
    )
    assert rank(idx, q).total_matches == 1


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"config": {"context_weight": 0, "social_weight": 0}},
        {"max_price": 20, "context": {"model": "m", "modality": "text", "values": [1, 0]}},
        {"context": {"model": "m", "modality": "text", "values": [1, 0]}, "limit": 0},
    ],
)
def test_request_validation(data):
    with pytest.raises(ValidationError):
        RankingRequest.model_validate(data)


def test_incompatible_query_and_empty_index():
    idx = index()
    for v in [vector("other", "text", [1, 0, 0]), vector(idx.model, "text", [1, 0])]:
        with pytest.raises(ValueError):
            rank(idx, RankingRequest(context=v))
    idx.items = []
    assert (
        rank(idx, RankingRequest(context=vector(idx.model, "text", [1, 0, 0]))).total_matches == 0
    )


def test_legacy_provider_contract_and_social_fallback():
    e = FakeEncoder()
    p = RankedProductProvider(index(), e, ROOT)
    request = req(
        top_social_references=[
            {"reference_id": "a", "description": "shirt"},
            {"reference_id": "b", "image_path": "missing.jpg", "description": "shirt"},
            {"reference_id": "c", "url": "https://example.com/image.jpg"},
        ],
        wardrobe_uploads=[{"upload_id": "u", "image_path": "missing.jpg"}],
    )
    result = p.retrieve(request)
    ProductRetrievalResponse.model_validate_json(result.model_dump_json())
    assert result.total_matches == 2 and len(result.pending_wardrobe_uploads) == 1
    assert all(x.embedding.model == e.identity for x in result.products)
    assert any("fallback" in w for w in result.warnings)
    assert any("not fetched" in w for w in result.warnings)
    assert all(x.availability.value == "UNKNOWN" for x in result.products)
    with pytest.raises(ValueError):
        p.retrieve(req(context_embedding=Embedding(model="wrong", values=[1, 0, 0])))
    with pytest.raises(ValueError):
        p.retrieve(
            req(
                top_social_references=[
                    {"reference_id": "a", "embedding": {"model": "wrong", "values": [1, 0, 0]}}
                ]
            )
        )


def test_external_normalizes_nonunit_rejects_zero():
    e = FakeEncoder()
    p = RankedProductProvider(index(), e, ROOT)
    assert p.retrieve(req(context_embedding=Embedding(model=e.identity, values=[2, 0, 0]))).products
    with pytest.raises(ValueError):
        p.retrieve(req(context_embedding=Embedding(model=e.identity, values=[0, 0, 0])))


@pytest.mark.skipif(
    os.environ.get("ITSA_REAL_MODEL_TEST") != "1",
    reason="opt-in real FashionCLIP smoke; requires local weights",
)
def test_real_fashionclip_roundtrip():
    encoder = FashionCLIPEncoder(ROOT / ".cache/huggingface", offline=True, batch_size=2)
    texts = encoder.encode_texts(["a white shirt", "black trousers"])
    assert len(encoder.encode_texts(["shirt " * 150])[0].values) == 512
    images = encoder.encode_images([Image.new("RGB", (224, 224), "red")])
    assert len(texts) == 2 and len(texts[0].values) == 512 and len(images[0].values) == 512
    assert np.linalg.norm(images[0].values) == pytest.approx(1, abs=1e-5)
    assert -1 <= cosine(texts[0], images[0]) <= 1
    assert cosine(texts[0], texts[0]) == pytest.approx(1)
    idx = build_index(process_catalogue(ROOT / "data/catalogue.json").products, encoder, ROOT)
    result = RankedProductProvider(idx, encoder, ROOT).retrieve(req())
    assert result.total_matches == 5
    ProductRetrievalResponse.model_validate_json(result.model_dump_json())


def test_wardrobe_m3_m5_m6_adapter(tmp_path):
    from travel_agent.products.retrieval.m4.adapter import wardrobe_index

    e = FakeEncoder()
    Image.new("RGB", (16, 16), "red").save(tmp_path / "photo.png")
    report = WardrobeAnalyzer(e, tmp_path).analyze(
        [WardrobeUpload(upload_id="my-shirt", image_path="photo.png")]
    )
    idx = wardrobe_index(report, e)
    assert len(idx.items) == 1 and idx.items[0].vector.modality == "image"
    assert idx.items[0].product.provenance.source == "wardrobe-upload:my-shirt"
    ranked = RankedProductProvider(idx, e, tmp_path).retrieve_ranked(req())
    assert ranked.total_matches == 1
    assert idx.items[0].product.availability.value == "UNKNOWN"


def test_verified_stock_filters():
    p = product(
        source_type="ECOMMERCE_API",
        availability="IN_STOCK",
        evidence={"availability_verified": True},
    )
    idx = build_index([p], FakeEncoder(), ROOT)
    assert (
        rank(
            idx,
            RankingRequest(context=vector(idx.model, "text", [1, 0, 0]), availability="IN_STOCK"),
        ).total_matches
        == 1
    )
    assert (
        rank(
            idx,
            RankingRequest(
                context=vector(idx.model, "text", [1, 0, 0]), availability="OUT_OF_STOCK"
            ),
        ).total_matches
        == 0
    )


def test_downloader_verifies_weights_and_reuses_snapshot(tmp_path, monkeypatch):
    import hashlib

    from travel_agent.products.retrieval.m5 import download as module

    monkeypatch.setattr(module, "FILES", ["model.safetensors"])
    monkeypatch.setattr(module, "WEIGHT_SHA256", hashlib.sha256(b"good").hexdigest())
    calls = []

    def fetch(arguments, check):
        calls.append(arguments)
        Path(arguments[-1]).write_bytes(b"good")

    monkeypatch.setattr(module.subprocess, "run", fetch)
    snapshot = module.download(tmp_path)
    module.download(tmp_path)
    assert len(calls) == 1 and (snapshot / "model.safetensors").read_bytes() == b"good"
    (snapshot / "model.safetensors").write_bytes(b"corrupted")

    def invalid_fetch(arguments, check):
        Path(arguments[-1]).write_bytes(b"bad")

    monkeypatch.setattr(module.subprocess, "run", invalid_fetch)
    with pytest.raises(ValueError, match="SHA-256"):
        module.download(tmp_path)
    assert (snapshot / "model.safetensors").read_bytes() == b"corrupted"
