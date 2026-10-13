"""Shared contract, upstream compatibility and lazy tool integration tests."""

import importlib.util
import json
from pathlib import Path

import pytest

from travel_agent.products.retrieval.integration import (
    ProductToolInput,
    Settings,
    map_understanding,
    shared_candidates,
)
from travel_agent.products.service import run
from travel_agent.schemas import ProductCandidate
from travel_agent.understanding.service import run as understand

ROOT = Path(__file__).resolve().parents[1]
HEAVY = all(importlib.util.find_spec(n) is not None for n in ("torch", "pandas", "numpy", "PIL"))


def understanding():
    return understand(
        json.loads((ROOT / "examples/understanding/seoul_winter.json").read_text())
    ).model_dump(mode="json")


def test_empty_tool_requires_explicit_inputs():
    assert run({}).status == "needs_input"
    assert run({"source": "photos20"}).status == "needs_input"
    assert run({"invented_field": True}).status == "error"


def test_student1_mapping_preserves_constraints_without_single_item_budget():
    original = understanding()
    context, sidecar = map_understanding(original)
    assert context.trip.destination == "Seoul"
    assert context.style_preferences == original["data"]["trip_request"]["style"]
    assert sidecar["student1"] == original
    assert sidecar["hard_constraints"]["budget_minor"] == 50000
    assert sidecar["feasibility_status"] == "unverified"


def test_incomplete_understanding_is_blocked():
    original = understanding()
    original["data"]["ready_for_downstream"] = False
    with pytest.raises(ValueError, match="not ready"):
        map_understanding(original)


def test_input_path_cannot_escape_data_root(tmp_path):
    settings = Settings(tmp_path, tmp_path / "out", tmp_path / "weights")
    with pytest.raises(ValueError):
        settings.input_path("../outside.pt")


def test_no_false_price_or_unknown_category_projection():
    item = {
        "product_id": "a",
        "category": "TOP",
        "owned": False,
        "price": None,
        "evidence": {"price_verified": False},
        "colors": [],
        "title": "shirt",
        "source_type": "PREPARED_CATALOGUE",
    }
    bundle = {"owned_items": [], "top_k_products": {"TOP": [item]}}
    candidates, blocked = shared_candidates(bundle, "SGD")
    assert not candidates and blocked[0]["id"] == "a"
    item.update(owned=True, category="UNKNOWN")
    assert not shared_candidates(bundle, "SGD")[0]
    item["category"] = "TOP"
    candidates, blocked = shared_candidates(bundle, "SGD")
    assert not blocked and ProductCandidate.model_validate(candidates[0]).purchase_cost_minor == 0


def test_verified_price_uses_native_currency_minor_units():
    item = {
        "product_id": "api",
        "category": "TOP",
        "owned": False,
        "price": {"amount": 20.25, "currency": "USD", "is_synthetic": False},
        "evidence": {"price_verified": True},
        "colors": [],
        "title": "shirt",
        "source_type": "ECOMMERCE_API",
    }
    bundle = {"owned_items": [], "top_k_products": {"TOP": [item]}}
    candidates, _ = shared_candidates(bundle, "SGD")
    assert candidates[0]["purchase_cost_minor"] == 2025 and candidates[0]["currency"] == "USD"
    item["price"]["amount"] = 20.251
    assert not shared_candidates(bundle, "SGD")[0]


@pytest.mark.skipif(not HEAVY, reason="requires products extra")
def test_shared_service_calls_preserved_backend_and_returns_category_groups(tmp_path):
    from travel_agent.products.retrieval.m5 import vector

    class Encoder:
        identity = "explicit-test-double"
        dimension = 512

        def encode_texts(self, texts):
            return [vector(self.identity, "text", [1] + [0] * 511) for _ in texts]

        def encode_images(self, images):
            return [vector(self.identity, "image", [1] + [0] * 511) for _ in images]

    request = json.loads((ROOT / "fixtures/phase1_request.json").read_text())
    # This test has no user wardrobe, external files, model weights or network.
    request["wardrobe_uploads"] = []
    result = run(
        {
            "source": "prepared",
            "request": request,
            "understanding": understanding(),
            "k_per_category": 3,
        },
        settings=Settings(ROOT, tmp_path, tmp_path),
        encoder=Encoder(),
    )
    assert result.status == "ok", result.reason
    assert {k: len(v) for k, v in result.data["bundle"]["top_k_products"].items()} == {
        "TOP": 3,
        "BOTTOM": 3,
        "FOOTWEAR": 3,
        "OUTERWEAR": 3,
    }
    assert not result.data[
        "candidates"
    ]  # Synthetic prices never masquerade as real purchase costs.
    assert len(result.data["handoff_blockers"]) == 12
    assert result.data["upstream_metadata"]["student1"]["data"]["ready_for_downstream"]


def test_base_import_does_not_load_models_or_network(tmp_path):
    import subprocess
    import sys

    script = (
        "import sys; import travel_agent.agent; "
        "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True, capture_output=True)


def test_tool_schema_rejects_unconfirmed_vector_mode():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ProductToolInput.model_validate({"reference_mode": "silently-trust"})
