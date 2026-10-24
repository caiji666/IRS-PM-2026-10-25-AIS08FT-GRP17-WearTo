"""Dynamic pools and wire contracts; test doubles never claim model accuracy."""

import copy

import pytest

from travel_agent.outfits.service import run
from travel_agent.recommendation.service import run as recommend
from travel_agent.recommendation.student5.student4_adapter import adapt_student4_output

np = pytest.importorskip("numpy")

IDENTITY = "explicit-outfit-test-encoder"
MANIFEST = {
    "encoder_identity": IDENTITY,
    "model_version": "explicit-test-model",
    "categories": ["OUTERWEAR", "TOP", "BOTTOM", "FOOTWEAR"],
    "feature_dimension": 2048,
}


class Model:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        self.features = features
        # Deterministic fixture only. Actual model inference is checked separately.
        return np.tile([0.25, 0.75], (len(features), 1))


def embedding(position=0):
    return {"model": IDENTITY, "values": [float(i == position) for i in range(512)]}


def item(category, name, owned=False):
    return {
        "product_id": name,
        "title": name,
        "category": category,
        "owned": owned,
        "source_type": "WARDROBE" if owned else "ECOMMERCE",
        "price": None if owned else {"amount": 25, "currency": "USD"},
        "availability": "UNKNOWN" if owned else "IN_STOCK",
        "evidence": {"price_verified": True, "availability_verified": not owned},
        "embedding": embedding(),
        "garment_embedding": embedding(),
        "product_url": None if owned else "https://www.ebay.com/itm/example",
        "colors": [],
        "sizes": [],
    }


def payload():
    groups = {c: [item(c, c.lower())] for c in MANIFEST["categories"]}
    ranked = {
        c: [{"item": {"product": p}, "score": 0.4} for p in rows] for c, rows in groups.items()
    }
    return {
        "products": {
            "bundle": {
                "top_k_products": groups,
                "owned_items": [
                    item("TOP", "owned-top", True),
                    item("UNKNOWN", "owned-unknown", True),
                ],
                "product_ranking": {"ranked_products": ranked},
            },
            "scoring": {"context_weight": 0.5, "social_weight": 0.5},
            "request_used": {
                "context_embedding": embedding(),
                "top_social_references": [{"embedding": embedding(1)}, {"embedding": embedding()}],
            },
        },
        "understanding": {"trip_request": {"currency": "USD"}},
    }


def test_full_combination_count_owned_zero_padding_and_student5_contract():
    spec = payload()
    original = copy.deepcopy(spec)
    model = Model()
    result = run(spec, model=model, manifest=MANIFEST)
    assert result.status == "ok", result.reason
    data = result.data
    assert data["summary"]["all_combination_count"] == 4
    assert len(data["outfits"]) == 4
    assert len(data["excluded_items"]) == 1
    assert model.features.shape == (4, 2048)
    assert np.all(model.features[:2, :512] == 0)  # Optional outerwear is truly zero padded.
    assert any(any(p["is_user_owned"] for p in row["items"]) for row in data["outfits"])
    owned = data["product_catalog"]["owned-top"]
    assert owned["price"] is None and owned["availability"] == "UNKNOWN"
    assert owned["product_score"] == 1  # Same max-over-references relevance formula as M6.
    assert len(adapt_student4_output(data).outfits) == 4
    assert spec == original
    s5 = recommend(
        {
            "action": "recommend",
            "student4": data,
            "request": {
                "destination": "Tokyo",
                "start_date": "2026-10-23",
                "end_date": "2026-10-25",
                "budget": 500,
                "currency": "USD",
            },
        }
    )
    assert s5.status == "ok", s5.reason
    assert len(s5.data["recommendation_set"]["recommendations"]) <= 3


def test_ids_do_not_depend_on_pool_order_and_inputs_are_dynamic():
    first = run(payload(), model=Model(), manifest=MANIFEST).data
    spec = payload()
    spec["products"]["bundle"]["owned_items"].reverse()
    second = run(spec, model=Model(), manifest=MANIFEST).data
    assert first["all_scored_combinations"] == second["all_scored_combinations"]
    spec["products"]["bundle"]["owned_items"].append(item("TOP", "another-owned", True))
    third = run(spec, model=Model(), manifest=MANIFEST).data
    assert third["summary"]["all_combination_count"] == 6


def test_unknown_stock_is_scored_but_never_invented_as_a_stock_boolean():
    spec = payload()
    p = spec["products"]["bundle"]["top_k_products"]["TOP"][0]
    p["availability"] = "UNKNOWN"
    p["evidence"]["availability_verified"] = False
    result = run(spec, model=Model(), manifest=MANIFEST)
    assert len(result.data["all_scored_combinations"]) == 4
    assert len(result.data["outfits"]) == 2
    assert result.data["projection_issues"][0]["product_id"] == "top"
    assert result.data["product_catalog"]["top"]["availability"] == "UNKNOWN"


def test_bad_identity_norm_or_missing_owned_signals_fails_before_inference():
    for mutation in ("identity", "norm", "signals"):
        spec = payload()
        if mutation == "signals":
            spec["products"].pop("request_used")
        else:
            p = spec["products"]["bundle"]["top_k_products"]["TOP"][0]
            p["garment_embedding"]["model"] = "wrong" if mutation == "identity" else IDENTITY
            if mutation == "norm":
                p["garment_embedding"]["values"][0] = 10
        assert run(spec, model=Model(), manifest=MANIFEST).status == "needs_input"
    assert run({}).status == "needs_input"


def test_limit_does_not_silently_truncate_and_absent_role_yields_no_candidates():
    spec = payload()
    spec["max_combinations"] = 3
    assert run(spec, model=Model(), manifest=MANIFEST).status == "needs_input"
    spec.pop("max_combinations")
    spec["products"]["bundle"]["top_k_products"]["FOOTWEAR"] = []
    result = run(spec, model=Model(), manifest=MANIFEST)
    assert result.status == "no_candidates"
    assert result.data["all_scored_combinations"] == []
