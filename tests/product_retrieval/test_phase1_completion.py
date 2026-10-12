import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from travel_agent.products.retrieval.catalogue import PreparedCatalogueProvider
from travel_agent.products.retrieval.m3 import process_catalogue, process_records
from travel_agent.products.retrieval.m5 import FashionCLIPEncoder, build_index, vector
from travel_agent.products.retrieval.m6 import (
    GroupedRankingReport,
    GroupedRankingRequest,
    RankedProductProvider,
    RankingConfig,
    RankingRequest,
    rank,
    rank_by_category,
)
from travel_agent.products.retrieval.schemas import (
    Category,
    ProductRetrievalRequest,
    ProductRetrievalResponse,
)

ROOT = Path(
    __import__("os").environ.get("STUDENT3_DATA_ROOT", str(Path(__file__).resolve().parents[2]))
)


class Encoder:
    identity = "test-grouped-only"
    dimension = 2

    def encode_texts(self, texts):
        return [vector(self.identity, "text", [1, 0]) for t in texts]

    def encode_images(self, images):
        return [vector(self.identity, "image", [1, 0]) for i in images]


def make_index(count=7):
    rows = []
    for cat in Category:
        for i in range(count):
            rows.append(
                {
                    "product_id": f"{cat.value}-{i:02d}",
                    "title": f"Sample {cat.value} {i}",
                    "raw_category": cat.value,
                    "price": {"amount": 10 + i, "currency": "SGD", "is_synthetic": True},
                    "sizes": ["M"],
                }
            )
    return build_index(process_records(rows).products, Encoder(), ROOT)


def request(idx, **kwargs):
    return GroupedRankingRequest(context=vector(idx.model, "text", [1, 0]), **kwargs)


def test_independent_top5_no_global_prelimit():
    idx = make_index()
    report = rank_by_category(idx, request(idx, limit=1))
    assert list(report.ranked_products) == [
        Category.TOP,
        Category.BOTTOM,
        Category.FOOTWEAR,
        Category.OUTERWEAR,
    ]
    assert all(len(x) == 5 for x in report.ranked_products.values())
    assert all(x == 7 for x in report.total_matches_per_category.values())
    assert report.total_matches == 28 and report.excluded_unknown_matches == 7
    for cat, items in report.ranked_products.items():
        assert [r.item.product.product_id for r in items] == [
            f"{cat.value}-{i:02d}" for i in range(5)
        ]
        assert all(r.item.product.category == cat for r in items)
    GroupedRankingReport.model_validate_json(report.model_dump_json())


def test_category_selection_and_explicit_unknown():
    idx = make_index()
    result = rank_by_category(
        idx, request(idx, categories=["BOTTOM", "TOP", "BOTTOM"], k_per_category=2)
    )
    assert list(result.ranked_products) == [Category.BOTTOM, Category.TOP]
    assert result.total_matches == 14 and result.excluded_unknown_matches == 0
    result = rank_by_category(idx, request(idx, categories=["UNKNOWN"]))
    assert (
        len(result.ranked_products[Category.UNKNOWN]) == 5 and result.excluded_unknown_matches == 0
    )


def test_shortage_empty_and_filters_no_padding():
    idx = make_index(2)
    result = rank_by_category(idx, request(idx, k_per_category=5))
    assert all(len(x) == 2 for x in result.ranked_products.values())
    assert any("only 2" in w for w in result.warnings)
    result = rank_by_category(idx, request(idx, size="XL"))
    assert result.total_matches == 0 and all(x == [] for x in result.ranked_products.values())
    result = rank_by_category(idx, request(idx, currency="SGD", max_price=10))
    assert all(len(x) == 1 for x in result.ranked_products.values())
    result = rank_by_category(idx, request(idx, currency="USD"))
    assert result.total_matches == 0
    result = rank_by_category(idx, request(idx, availability="IN_STOCK"))
    assert result.total_matches == 0
    idx.items = []
    assert rank_by_category(idx, request(idx)).total_matches == 0


@pytest.mark.parametrize("k", [0, -1, 1001])
def test_invalid_k(k):
    idx = make_index()
    with pytest.raises(ValidationError):
        request(idx, k_per_category=k)


def test_max_changes_ranking_and_uses_best_not_average():
    idx = make_index(1)
    idx.items = idx.items[:2]
    idx.items[0].vector = vector(idx.model, "text", [1, 0])
    idx.items[1].vector = vector(idx.model, "text", [1, 1])
    q = RankingRequest(
        social_references=[vector(idx.model, "image", [1, 0]), vector(idx.model, "image", [0, 1])]
    )
    r = rank(idx, q)
    assert r.products[0].item.product.product_id.endswith("TOP-00")
    assert r.products[0].social_similarity == pytest.approx(1)
    assert r.products[1].social_similarity == pytest.approx(2**-0.5)
    assert r.social_aggregation == "max" and r.schema_version == "m6.2"
    q.context = vector(idx.model, "text", [0, 1])
    q.config = RankingConfig(context_weight=0.7, social_weight=0.3)
    result = rank(idx, q)
    chosen = next(p for p in result.products if p.item.product.product_id == "TOP-00")
    assert chosen.score == pytest.approx(0.3)


def test_negative_reference_scores_not_clamped_to_zero():
    idx = make_index(1)
    q = RankingRequest(
        social_references=[vector(idx.model, "image", [-1, 0]), vector(idx.model, "image", [-1, 1])]
    )
    assert rank(idx, q).products[0].social_similarity == pytest.approx(-(2**-0.5))


def test_global_limit_and_m1_contract_preserved():
    idx = make_index()
    e = Encoder()
    provider = RankedProductProvider(idx, e, ROOT)
    req = ProductRetrievalRequest.model_validate(
        {"context": {"trip": {"destination": "Singapore"}}, "limit": 1}
    )
    global_result = provider.retrieve_ranked(req)
    assert len(global_result.products) == 1 and global_result.total_matches == 35
    grouped = provider.retrieve_grouped(req)
    assert sum(len(items) for items in grouped.ranked_products.values()) == 20
    legacy = provider.retrieve(req)
    assert len(legacy.products) == 1 and legacy.total_matches == 35
    ProductRetrievalResponse.model_validate_json(legacy.model_dump_json())
    with pytest.raises(ValidationError):
        provider.retrieve_grouped(req, 0)


def test_report_rejects_inconsistent_group_counts():
    idx = make_index()
    data = rank_by_category(idx, request(idx)).model_dump()
    data["total_matches"] = 1
    with pytest.raises(ValidationError):
        GroupedRankingReport.model_validate(data)
    data = rank_by_category(idx, request(idx)).model_dump()
    data["ranked_products"][Category.TOP] = []
    with pytest.raises(ValidationError):
        GroupedRankingReport.model_validate(data)


@pytest.mark.parametrize("suffix", ["json", "csv"])
def test_expanded_catalogue_and_provenance(suffix):
    path = ROOT / f"data/catalogue_phase1.{suffix}"
    m2 = PreparedCatalogueProvider(path)
    assert len(m2.products) == 34 and not m2.issues
    m3 = process_catalogue(path)
    assert len(m3.products) == 34
    counts = {cat: sum(p.category == cat for p in m3.products) for cat in Category}
    assert all(counts[cat] == 8 for cat in Category if cat != Category.UNKNOWN)
    assert counts[Category.UNKNOWN] == 2
    assert all(
        p.availability.value == "UNKNOWN"
        and p.product_url is None
        and not p.image_embedding_candidate
        for p in m3.products
    )
    assert all(p.price is None or p.price.is_synthetic for p in m3.products)
    assert all(p.provenance.source == str(path.resolve()) for p in m3.products)
    assert all(not p.evidence.price_verified and not p.evidence.image_verified for p in m3.products)
    assert len(PreparedCatalogueProvider(ROOT / "data/catalogue.json").products) == 5
    result = rank_by_category(
        build_index(m3.products, Encoder(), ROOT),
        GroupedRankingRequest(context=vector(Encoder.identity, "text", [1, 0])),
    )
    assert all(len(items) == 5 for items in result.ranked_products.values())


@pytest.mark.skipif(
    os.environ.get("ITSA_REAL_MODEL_TEST") != "1",
    reason="requires actual local FashionCLIP weights",
)
def test_real_expanded_catalogue_grouped_top5():
    encoder = FashionCLIPEncoder(ROOT / ".cache/huggingface", offline=True)
    normalized = process_catalogue(ROOT / "data/catalogue_phase1.json")
    idx = build_index(normalized.products, encoder, ROOT)
    req = ProductRetrievalRequest.model_validate_json(
        (ROOT / "fixtures/phase1_request.json").read_text()
    )
    result = RankedProductProvider(idx, encoder, ROOT).retrieve_grouped(req)
    assert result.total_matches == 32 and result.excluded_unknown_matches == 2
    assert all(len(items) == 5 for items in result.ranked_products.values())
    for items in result.ranked_products.values():
        assert [r.score for r in items] == sorted([r.score for r in items], reverse=True)
    GroupedRankingReport.model_validate_json(result.model_dump_json())
