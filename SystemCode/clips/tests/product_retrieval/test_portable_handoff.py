import copy
import json

import pytest
from PIL import Image

from travel_agent.products.notifications.lifecycle import DeletedAccount, Registry
from travel_agent.products.retrieval.ebay.handoff import write_handoff


def fixture_report(tmp_path):
    source = tmp_path / "original.png"
    Image.new("RGB", (4, 4), "blue").save(source)
    item = {
        "product_id": "example",
        "owned": False,
        "image_path": str(source),
        "embedding": {"model": "fixture", "values": [1.0]},
        "availability": "IN_STOCK",
        "evidence": {"availability_verified": True},
    }
    owned = copy.deepcopy(item)
    owned["owned"] = True
    owned["provenance"] = {"original": {"image_path": source.name}}
    return {
        "environment": "production",
        "lifecycle_generation": 0,
        "upstream_metadata": {
            "inputs": {"pt": {"path": str(tmp_path / "input.pt"), "sha256": "abc"}}
        },
        "bundle": {
            "top_k_products": {"TOP": [item]},
            "owned_items": [owned],
            "product_ranking": {
                "ranked_products": {"TOP": [{"item": copy.deepcopy(item), "score": 0.5}]}
            },
        },
    }


def test_portable_output_relocates_and_registers(tmp_path, monkeypatch):
    monkeypatch.setenv("EBAY_MANAGED_ROOT", str(tmp_path))
    report = fixture_report(tmp_path)
    original = copy.deepcopy(report)
    target = tmp_path / "runs/output/20261019"
    result = write_handoff(report, target)
    assert report == original
    data = json.loads(result.read_text())
    assert str(tmp_path) not in result.read_text()
    assert data["upstream_provenance"]["inputs"]["pt"] == {"filename": "input.pt", "sha256": "abc"}
    assert data["bundle"]["top_k_products"]["TOP"][0]["evidence"]["availability_verified"]
    moved = tmp_path / "another-machine"
    target.rename(moved)
    for item in [data["bundle"]["top_k_products"]["TOP"][0], data["bundle"]["owned_items"][0]]:
        with Image.open(moved / item["image_path"]) as image:
            image.verify()
    assert data["bundle"]["owned_items"][0]["provenance"]["original"]["image_path"].startswith(
        "wardrobe_images/"
    )
    registry = Registry(tmp_path, tmp_path / ".cache/ebay-lifecycle/registry.sqlite3")
    try:
        assert (
            registry.db.execute(
                "SELECT kind FROM artifacts WHERE path=?", ("runs/output/20261019",)
            ).fetchone()[0]
            == "portable_handoff"
        )
    finally:
        registry.close()


def test_portable_refuses_invalidated_result(tmp_path, monkeypatch):
    monkeypatch.setenv("EBAY_MANAGED_ROOT", str(tmp_path))
    report = fixture_report(tmp_path)
    report["lifecycle_generation"] = -1
    with pytest.raises(DeletedAccount):
        write_handoff(report, tmp_path / "runs/stale")
    assert not (tmp_path / "runs/stale").exists()


def test_portable_missing_image_does_not_create_package(tmp_path, monkeypatch):
    monkeypatch.setenv("EBAY_MANAGED_ROOT", str(tmp_path))
    report = fixture_report(tmp_path)
    (tmp_path / "original.png").unlink()
    with pytest.raises(ValueError):
        write_handoff(report, tmp_path / "runs/missing")
    assert not (tmp_path / "runs/missing").exists()
