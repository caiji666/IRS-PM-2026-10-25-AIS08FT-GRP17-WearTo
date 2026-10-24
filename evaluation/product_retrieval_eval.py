"""Offline, reproducible evaluation of saved Production product-retrieval runs.

The three ranking arms share the exact same indexed eBay candidates. No API call,
embedding generation, or wardrobe ranking occurs here.
"""

import argparse
import csv
import hashlib
import json
import math
import random
from datetime import datetime
from pathlib import Path

from travel_agent.products.retrieval.m5.models import ProductIndex, vector
from travel_agent.products.retrieval.m6.models import GroupedRankingRequest, RankingConfig
from travel_agent.products.retrieval.m6.ranking import SUPPORTED_CATEGORIES, rank_by_category

ARM_WEIGHTS = {"context_only": (1.0, 0.0), "reference_only": (0.0, 1.0), "hybrid": (0.5, 0.5)}
CATEGORIES = tuple(category.value for category in SUPPORTED_CATEGORIES)
LABEL_FIELDS = [
    "case_id",
    "product_id",
    "assigned_category",
    "title",
    "image_path",
    "product_url",
    "price",
    "currency",
    "true_category",
    "relevance_0_to_3",
]


def load_case(path: Path):
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get("environment") != "production":
        raise ValueError(f"Production-only evaluation: {path}")
    request = report["request_used"]
    index = ProductIndex.model_validate(report["index"])
    context_data = request["context_embedding"]
    references = [
        ref["embedding"] for ref in request["top_social_references"] if ref.get("embedding")
    ]
    if not context_data or not references:
        raise ValueError(f"Both embedding signals required: {path}")
    if index.model != context_data["model"] or any(
        ref["model"] != index.model for ref in references
    ):
        raise ValueError(f"Incompatible embedding identities: {path}")
    context = vector(index.model, "text", context_data["values"])
    social = [vector(index.model, "image", ref["values"]) for ref in references]
    case_id = path.parent.name
    rankings = {}
    for arm, (context_weight, social_weight) in ARM_WEIGHTS.items():
        request_rank = GroupedRankingRequest(
            context=context,
            social_references=social,
            config=RankingConfig(context_weight=context_weight, social_weight=social_weight),
            categories=request["categories"],
            size=request["size"],
            currency=request["currency"],
            max_price=request["max_price"],
            k_per_category=report["bundle"]["product_ranking"]["k_per_category"],
        )
        rankings[arm] = rank_by_category(index, request_rank)
    saved = report["bundle"]["product_ranking"]["ranked_products"]
    for category in CATEGORIES:
        actual = rankings["hybrid"].ranked_products[category]
        expected = saved[category]
        if [r.item.product.product_id for r in actual] != [
            r["item"]["product"]["product_id"] for r in expected
        ]:
            raise ValueError(f"Hybrid replay differs from saved ranking: {case_id}/{category}")
        if any(abs(r.score - old["score"]) > 1e-5 for r, old in zip(actual, expected)):
            raise ValueError(f"Hybrid scores differ from saved ranking: {case_id}/{category}")
    return {
        "id": case_id,
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "report": report,
        "index": index,
        "rankings": rankings,
    }


def _product_ids(ranking, category):
    return [entry.item.product.product_id for entry in ranking.ranked_products[category]]


def _fraction(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def objective_metrics(case):
    report = case["report"]
    items = case["index"].items
    hybrid = case["rankings"]["hybrid"]
    k = hybrid.k_per_category
    fields = ("price", "image_path", "product_url")
    completeness = {
        field: _fraction(sum(bool(getattr(item.product, field)) for item in items), len(items))
        for field in fields
    }
    completeness.update(
        {
            "verified_image_fraction": _fraction(
                sum(i.product.evidence.image_verified for i in items), len(items)
            ),
            "verified_stock_fraction": _fraction(
                sum(i.product.evidence.availability_verified for i in items), len(items)
            ),
        }
    )
    comparisons = {}
    for category in CATEGORIES:
        base = set(_product_ids(hybrid, category))
        comparisons[category] = {
            "eligible": hybrid.total_matches_per_category[category],
            "returned": len(base),
            "top_k_coverage": _fraction(len(base), k),
            "hybrid_context_overlap_at_k": _fraction(
                len(base & set(_product_ids(case["rankings"]["context_only"], category))), k
            ),
            "hybrid_reference_overlap_at_k": _fraction(
                len(base & set(_product_ids(case["rankings"]["reference_only"], category))), k
            ),
        }
    summary = report["summary"]
    return {
        "case_id": case["id"],
        "destination": report["request_used"]["context"]["trip"]["destination"],
        "source_file": case["path"],
        "source_sha256": case["sha256"],
        "marketplace": report["marketplace"],
        "embedding_model": case["index"].model,
        "k": k,
        "hybrid_replay_matches_saved": True,
        "search_requests": summary["search_requests"],
        "unique_retrieved_candidates": summary["unique_candidates"],
        "indexed_candidates": len(items),
        "indexed_fraction_of_retrieved": _fraction(len(items), summary["unique_candidates"]),
        "unknown_assigned_category": sum(i.product.category.value == "UNKNOWN" for i in items),
        "top_k": comparisons,
        "indexed_field_completeness": completeness,
    }


def label_rows(cases):
    rows = []
    for case in cases:
        for item in case["index"].items:
            p = item.product
            rows.append(
                {
                    "case_id": case["id"],
                    "product_id": p.product_id,
                    "assigned_category": p.category.value,
                    "title": p.title,
                    "image_path": p.image_path or "",
                    "product_url": str(p.product_url or ""),
                    "price": p.price.amount if p.price else "",
                    "currency": p.price.currency if p.price else "",
                    "true_category": "",
                    "relevance_0_to_3": "",
                }
            )
    random.Random(17).shuffle(rows)
    return rows


def read_labels(path: Path, cases):
    expected = {
        (case["id"], item.product.product_id) for case in cases for item in case["index"].items
    }
    labels = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = row["case_id"], row["product_id"]
            if key not in expected or key in labels:
                raise ValueError(f"Unexpected or duplicate label: {key}")
            category = row["true_category"].strip().upper()
            grade = row["relevance_0_to_3"].strip()
            if category not in (*CATEGORIES, "UNKNOWN") or grade not in ("0", "1", "2", "3"):
                raise ValueError(f"Every item needs true_category and relevance 0-3: {key}")
            labels[key] = (category, int(grade))
    if set(labels) != expected:
        raise ValueError(f"Labels incomplete: {len(labels)} of {len(expected)} rows")
    return labels


def judged_metrics(cases, labels):
    per_case = []
    all_category = []
    all_precision = []
    all_recall = []
    all_ndcg = []
    for case in cases:
        k = case["rankings"]["hybrid"].k_per_category
        true = {
            item.product.product_id: labels[(case["id"], item.product.product_id)]
            for item in case["index"].items
        }
        assignments = [
            (item.product.category.value, true[item.product.product_id][0])
            for item in case["index"].items
        ]
        accuracy = _fraction(sum(pred == gold for pred, gold in assignments), len(assignments))
        f1s = []
        for category in (*CATEGORIES, "UNKNOWN"):
            tp = sum(pred == gold == category for pred, gold in assignments)
            fp = sum(pred == category and gold != category for pred, gold in assignments)
            fn = sum(pred != category and gold == category for pred, gold in assignments)
            f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        category_metric = {"accuracy": accuracy, "macro_f1": round(sum(f1s) / len(f1s), 4)}
        all_category.append(category_metric)
        arms = {}
        for arm, ranking in case["rankings"].items():
            categories = {}
            for category in CATEGORIES:
                selected = _product_ids(ranking, category)
                relevant = {
                    pid for pid, (gold, grade) in true.items() if gold == category and grade >= 2
                }
                hits = len(set(selected) & relevant)
                graded = [
                    grade if gold == category else 0
                    for gold, grade in (true[pid] for pid in selected)
                ]
                gains = [grade if grade >= 2 else 0 for grade in graded]
                dcg = sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(gains))
                ideal = sorted(
                    (grade for gold, grade in true.values() if gold == category and grade >= 2),
                    reverse=True,
                )[:k]
                idcg = sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(ideal))
                result = {
                    "precision_at_k": _fraction(hits, k),
                    "recall_at_k_indexed_pool": _fraction(hits, len(relevant)),
                    "ndcg_at_k": _fraction(dcg, idcg),
                    "mean_relevance_0_to_3": _fraction(sum(graded), len(selected)),
                    "relevant_in_index": len(relevant),
                }
                categories[category] = result
                all_precision.append((arm, result["precision_at_k"]))
                if result["recall_at_k_indexed_pool"] is not None:
                    all_recall.append((arm, result["recall_at_k_indexed_pool"]))
                if result["ndcg_at_k"] is not None:
                    all_ndcg.append((arm, result["ndcg_at_k"]))
            arms[arm] = categories
        per_case.append({"case_id": case["id"], "category": category_metric, "arms": arms})
    aggregate = {
        "category_accuracy": _fraction(sum(x["accuracy"] for x in all_category), len(all_category)),
        "category_macro_f1": _fraction(sum(x["macro_f1"] for x in all_category), len(all_category)),
    }
    for metric, values in (
        ("precision_at_k", all_precision),
        ("recall_at_k_indexed_pool", all_recall),
        ("ndcg_at_k", all_ndcg),
    ):
        aggregate[metric] = {
            arm: _fraction(
                sum(value for name, value in values if name == arm),
                sum(name == arm for name, _ in values),
            )
            for arm in ARM_WEIGHTS
        }
    return {
        "per_case": per_case,
        "aggregate": aggregate,
        "recall_scope": "Only the saved indexed candidate pool; not all eBay listings.",
    }


def evaluate(paths, output_dir, label_path=None):
    cases = [load_case(path) for path in paths]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    if not cases:
        raise ValueError("Provide at least one saved report")
    labels = read_labels(label_path, cases) if label_path else None
    output_dir.mkdir(parents=True, exist_ok=False)
    objective = [objective_metrics(case) for case in cases]
    report = {
        "schema_version": "product-retrieval-eval.1",
        "created_at": datetime.now().astimezone().isoformat(),
        "design": (
            "Offline matched-candidate reranking; no live API request or new embedding computation."
        ),
        "arms": {
            arm: {"context_weight": cw, "social_weight": sw}
            for arm, (cw, sw) in ARM_WEIGHTS.items()
        },
        "cases": objective,
        "case_count": len(cases),
        "retrieved_total": sum(c["unique_retrieved_candidates"] for c in objective),
        "indexed_total": sum(c["indexed_candidates"] for c in objective),
        "top_k_slots_filled": sum(v["returned"] for c in objective for v in c["top_k"].values()),
        "top_k_slots_possible": sum(c["k"] * len(CATEGORIES) for c in objective),
        "human_evaluation": {"status": "pending_labels", "metrics": None},
    }
    if labels is not None:
        report["human_evaluation"] = {
            "status": "labels_supplied",
            "label_file": str(label_path),
            "metrics": judged_metrics(cases, labels),
        }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    with (output_dir / "labels_template.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(label_rows(cases))
    briefings = []
    for case in cases:
        request = case["report"]["request_used"]
        briefings.append(
            {
                "case_id": case["id"],
                "context": request["context"],
                "social_references": [
                    {
                        "reference_id": ref["reference_id"],
                        "image_path": ref.get("image_path"),
                        "url": ref.get("url"),
                        "description": ref.get("description"),
                    }
                    for ref in request["top_social_references"]
                ],
            }
        )
    (output_dir / "case_briefings.json").write_text(
        json.dumps(briefings, ensure_ascii=False, indent=2) + "\n"
    )
    for case in cases:
        for arm, ranking in case["rankings"].items():
            path = output_dir / f"{case['id']}_{arm}_top_k.json"
            path.write_text(
                json.dumps(
                    {
                        category: [
                            {"product_id": item.item.product.product_id, "score": item.score}
                            for item in ranking.ranked_products[category]
                        ]
                        for category in CATEGORIES
                    },
                    indent=2,
                )
                + "\n"
            )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports", nargs="+", type=Path, help="Saved production run_report.json files"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="Completed, adjudicated labels_template.csv")
    args = parser.parse_args()
    report = evaluate(args.reports, args.output_dir, args.labels)
    print(
        json.dumps(
            {
                "cases": report["case_count"],
                "indexed": report["indexed_total"],
                "top_k_filled": report["top_k_slots_filled"],
                "top_k_possible": report["top_k_slots_possible"],
                "human_status": report["human_evaluation"]["status"],
                "metrics_file": str(args.output_dir / "metrics.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
