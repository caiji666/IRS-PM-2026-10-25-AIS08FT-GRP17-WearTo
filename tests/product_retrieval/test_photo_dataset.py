import json
from pathlib import Path

import pytest

from travel_agent.products.retrieval.m5 import vector
from travel_agent.products.retrieval.photo_dataset import photo_index

ROOT = Path(
    __import__("os").environ.get("STUDENT3_DATA_ROOT", str(Path(__file__).resolve().parents[2]))
)


pytestmark = pytest.mark.skipif(
    not (ROOT / "data/photos20").is_dir(), reason="requires optional local photo dataset"
)


class ImageEncoder:
    identity = "photo-test-only"
    dimension = 3

    def encode_images(self, images):
        assert all(i.mode == "RGB" for i in images)
        return [vector(self.identity, "image", [1, 0, 0]) for _ in images]

    def encode_texts(self, texts):
        raise AssertionError("test dataset must not silently use text")


def test_real_dataset_counts_and_uses_images():
    normalized, index, paths = photo_index(ROOT / "data/photos20", ImageEncoder())
    assert len(paths) == 20 and len(index.items) == 20
    assert all(i.vector.modality == "image" for i in index.items)
    from collections import Counter

    assert Counter(i.product.category.value for i in index.items) == dict(
        TOP=5, BOTTOM=5, FOOTWEAR=5, OUTERWEAR=5
    )
    for i in index.items:
        assert i.product.source_type.value == "PREPARED_CATALOGUE"
        assert i.product.price is None and i.product.availability.value == "UNKNOWN"
        assert not i.product.evidence.image_verified
        assert i.product.provenance.original["photo_dataset_source"]["sha256"]


def test_corrupt_image_checksum_fails(tmp_path):
    import shutil

    shutil.copytree(ROOT / "data/photos20", tmp_path / "photos")
    image = next((tmp_path / "photos/images").iterdir())
    image.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        photo_index(tmp_path / "photos", ImageEncoder())


def test_manifest_path_escape_fails(tmp_path):
    import shutil

    shutil.copytree(ROOT / "data/photos20", tmp_path / "photos")
    manifest = tmp_path / "photos/manifest.json"
    r = json.loads(manifest.read_text())
    r["records"][0]["image_file"] = "../outside.jpg"
    manifest.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="outside"):
        photo_index(tmp_path / "photos", ImageEncoder())
