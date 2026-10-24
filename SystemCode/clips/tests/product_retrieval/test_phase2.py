from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from travel_agent.products.retrieval.m3 import process_catalogue
from travel_agent.products.retrieval.m5 import build_index, vector
from travel_agent.products.retrieval.m6 import RankedProductProvider
from travel_agent.products.retrieval.phase2 import UnifiedItem, combine, ingest, load_store
from travel_agent.products.retrieval.schemas import (
    ExpandedFashionContext,
    ProductRetrievalRequest,
    Trip,
)

ROOT = Path(
    __import__("os").environ.get("STUDENT3_DATA_ROOT", str(Path(__file__).resolve().parents[2]))
)


class Encoder:
    identity = "phase2-test-only"
    dimension = 3

    def encode_texts(self, texts):
        # Controlled signals for category, color, subcategory and style.
        out = []
        for t in texts:
            match = (
                t == "a product photo of a shirt, t-shirt, blouse or sweater"
                or t == "a product photo of a red garment"
                or t == "a product photo of t-shirt clothing"
                or t == "a product photo of casual clothing"
            )
            out.append(vector(self.identity, "text", [1, 0, 0] if match else [0, 1, 0]))
        return out

    def encode_images(self, images):
        return [vector(self.identity, "image", [1, 0, 0]) for _ in images]


@pytest.fixture
def imported(tmp_path):
    photo = tmp_path / "original.png"
    Image.new("RGBA", (30, 40), (255, 0, 0, 255)).save(photo)
    store = tmp_path / "items.json"
    images = tmp_path / "managed"
    result, issues = ingest([photo], Encoder(), store, images)
    return photo, store, images, result, issues


def test_import_structured_saved_and_reload(imported):
    photo, path, images, store, issues = imported
    assert not issues
    item = store.items[0]
    assert item.category.value == "TOP" and item.subcategory == "t-shirt"
    assert item.styles == ["casual"] and item.colors == ["red"]
    assert item.owned and not item.additional_purchase_required
    assert item.price is None and item.availability.value == "UNKNOWN"
    assert item.garment_embedding.modality == "image"
    assert item.embedding.values == item.garment_embedding.values
    assert item.provenance.original["phase2_upload"]["original_path"] == str(photo)
    assert load_store(path) == store
    with Image.open(item.image_path) as im:
        assert im.mode == "RGB" and not im.getexif()


def test_repeated_and_identical_images_no_duplicates(imported):
    photo, path, images, store, _ = imported
    duplicate = photo.with_name("same.png")
    duplicate.write_bytes(photo.read_bytes())
    again, issues = ingest([photo, duplicate], Encoder(), path, images)
    assert len(again.items) == 1 and not issues


def test_bad_files_preserve_store_and_report(imported):
    photo, path, images, store, _ = imported
    bad = photo.with_name("bad.jpg")
    bad.write_text("not an image")
    again, issues = ingest([photo.with_name("missing.png"), bad], Encoder(), path, images)
    assert len(issues) == 2 and again == store


def test_store_corruption_is_not_silently_replaced(imported):
    photo, path, images, _, _ = imported
    path.write_text("{bad")
    with pytest.raises(ValueError):
        ingest([photo], Encoder(), path, images)
    assert path.read_text() == "{bad"


def test_model_mismatch_rejected(imported):
    photo, path, images, store, _ = imported
    e = Encoder()
    e.identity = "other-model"
    with pytest.raises(ValueError):
        ingest([photo], e, path, images)


def test_abstention_keeps_image_embedding(imported):
    photo, path, images, _, _ = imported

    class Ambiguous(Encoder):
        def encode_texts(self, texts):
            return [vector(self.identity, "text", [1, 0, 0]) for _ in texts]

    result, _ = ingest([photo], Ambiguous(), path, images)
    item = result.items[0]
    assert item.category.value == "UNKNOWN" and item.subcategory == "UNKNOWN"
    assert item.colors == [] and item.styles == [] and item.style_status == "UNKNOWN"
    assert item.garment_embedding.modality == "image"


def test_owned_pool_not_truncated_and_unified_products(imported):
    photo, path, images, store, _ = imported
    e = Encoder()
    idx = build_index(process_catalogue(ROOT / "data/catalogue_phase1.json").products, e, ROOT)
    req = ProductRetrievalRequest(
        context=ExpandedFashionContext(trip=Trip(destination="Singapore"), query="shirt"), limit=1
    )
    ranking = RankedProductProvider(idx, e, ROOT).retrieve_grouped(req, 2)
    bundle = combine(store, ranking, e.identity, e.dimension)
    assert len(bundle.owned_items) == 1 and all(len(v) == 2 for v in bundle.top_k_products.values())
    assert all(
        not i.owned and i.garment_embedding.modality == "text"
        for v in bundle.top_k_products.values()
        for i in v
    )
    for i in bundle.owned_items + [i for v in bundle.top_k_products.values() for i in v]:
        UnifiedItem.model_validate_json(i.model_dump_json())
    with pytest.raises(ValueError):
        combine(store, ranking, "bad", 3)


@pytest.mark.parametrize(
    "changes",
    [
        {"owned": False},
        {"additional_purchase_required": True},
        {"ownership_source": None},
        {"subcategory": "shirt", "embedding": None},
        {"styles": ["casual"], "style_status": "UNKNOWN"},
    ],
)
def test_reject_inconsistent_owned_metadata(imported, changes):
    item = imported[3].items[0].model_dump(mode="json")
    item.update(changes)
    with pytest.raises(ValidationError):
        UnifiedItem.model_validate(item)


def test_original_m1_projection(imported):
    from travel_agent.products.retrieval.phase2.service import to_m1_garment
    from travel_agent.products.retrieval.schemas import Garment

    garment = to_m1_garment(imported[3].items[0])
    assert garment.source_type.value == "WARDROBE"
    Garment.model_validate_json(garment.model_dump_json())
    assert "styles" not in garment.model_dump()
