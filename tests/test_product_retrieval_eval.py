import csv
from types import SimpleNamespace

import pytest

from evaluation.product_retrieval_eval import (
    ARM_WEIGHTS,
    CATEGORIES,
    judged_metrics,
    read_labels,
)


def _case():
    products = []
    groups = {}
    for category in CATEGORIES:
        first = SimpleNamespace(
            product_id=f"{category}-good", category=SimpleNamespace(value=category)
        )
        second = SimpleNamespace(
            product_id=f"{category}-bad", category=SimpleNamespace(value=category)
        )
        products.extend([SimpleNamespace(product=first), SimpleNamespace(product=second)])
        groups[category] = [SimpleNamespace(item=SimpleNamespace(product=first))]
    rankings = {
        arm: SimpleNamespace(k_per_category=1, ranked_products=groups) for arm in ARM_WEIGHTS
    }
    return {"id": "test", "index": SimpleNamespace(items=products), "rankings": rankings}


def test_judged_metrics_perfect_top_one():
    case = _case()
    labels = {
        ("test", item.product.product_id): (
            item.product.category.value,
            3 if item.product.product_id.endswith("good") else 0,
        )
        for item in case["index"].items
    }
    metrics = judged_metrics([case], labels)
    assert metrics["aggregate"]["precision_at_k"]["hybrid"] == 1.0
    assert metrics["aggregate"]["recall_at_k_indexed_pool"]["hybrid"] == 1.0
    assert metrics["aggregate"]["ndcg_at_k"]["hybrid"] == 1.0
    assert metrics["aggregate"]["category_accuracy"] == 1.0


def test_incomplete_labels_rejected(tmp_path):
    path = tmp_path / "labels.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["case_id", "product_id", "true_category", "relevance_0_to_3"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "test",
                "product_id": "TOP-good",
                "true_category": "TOP",
                "relevance_0_to_3": "3",
            }
        )
    with pytest.raises(ValueError, match="Labels incomplete"):
        read_labels(path, [_case()])
