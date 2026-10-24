import json
from pathlib import Path

import pytest
import torch

from travel_agent.products.retrieval.m5 import vector
from travel_agent.products.retrieval.m5.encoder import MODEL_IDENTITY, TRANSFORMERS_VERSION
from travel_agent.products.retrieval.m6.provider import prepare_request
from travel_agent.products.retrieval.schemas import ProductRetrievalRequest
from travel_agent.products.retrieval.upstream import adapt_files

PROFILE = {
    "model_id": "patrickjohncyh/fashion-clip",
    "revision": "7e3ba62ce16b379a1ab479346b66f192e76f51b7",
    "preprocessing": "clip-rgb-exif-text77-v1",
    "transformers_version": TRANSFORMERS_VERSION,
    "dimension": 512,
    "normalization": "L2",
}


class Encoder:
    identity = MODEL_IDENTITY
    dimension = 512

    def encode_texts(self, texts):
        return [vector(self.identity, "text", [1] + [0] * 511) for _ in texts]


@pytest.fixture
def files(tmp_path):
    pt = tmp_path / "output.pt"
    text = tmp_path / "expanded.txt"
    context = tmp_path / "context.json"
    info = tmp_path / "models.json"
    tensor = torch.zeros(2, 512)
    tensor[0, 0] = 2
    tensor[1, 1] = 3
    payload = {
        "context_embedding": tensor[:1],
        "top_image_embeddings": tensor,
        "top_images": [
            {
                "image_id": "b",
                "source_url": "https://example.com/b",
                "image_url": "https://example.com/b.jpg",
                "description": "coat",
                "rank": 2,
                "photographer": "P",
            },
            {
                "image_id": "a",
                "source_url": "https://example.com/a",
                "description": "shoes",
                "rank": 1,
                "similarity_score": 0.5,
            },
        ],
    }
    torch.save(payload, pt)
    text.write_text("Seoul winter warm clothes")
    source = {
        "destination": "Seoul",
        "season": "winter",
        "weather_clothing_requirements": {"outerwear": ["winter coat"], "properties": ["warm"]},
        "activity_style_requirements": {"dinner": ["smart casual"]},
        "user_style": ["minimalist"],
        "extra_upstream": {"keep": True},
    }
    context.write_text(json.dumps(source))
    info.write_text(json.dumps({"context": PROFILE, "social_images": PROFILE}))
    return pt, text, context, info, payload, source


def test_unknown_identity_valid_request_but_ranking_blocked(files):
    pt, text, context, info, payload, source = files
    req, fallback, meta = adapt_files(pt, text, context)
    assert meta["status"] == "pending_model_identity"
    assert req.context_embedding.model.startswith("unverified-upstream:")
    with pytest.raises(ValueError, match="incompatible"):
        prepare_request(req, Encoder(), Path("."))
    ProductRetrievalRequest.model_validate_json(req.model_dump_json())
    assert len(req.context_embedding.values) == 512 and req.context_embedding.values[0] == 1
    assert meta["original_vector_norms"]["social_images"] == [2, 3]


def test_mapping_pairing_and_full_provenance(files):
    req, _, meta = adapt_files(*files[:3])
    assert req.context.trip.destination == "Seoul" and req.context.activities == ["dinner"]
    assert (
        req.context.style_preferences == ["minimalist"]
        and "winter coat" in req.context.weather_notes
    )
    assert req.context.query == "Seoul winter warm clothes"
    assert [r.reference_id for r in req.top_social_references] == ["b", "a"]
    assert req.top_social_references[0].embedding.values[0] == 1
    assert req.top_social_references[1].embedding.values[1] == 1
    assert meta["top_images"] == files[4]["top_images"] and meta["original_context"] == files[5]
    assert meta["inputs"]["pt"]["sha256"]


def test_declared_profiles_compatible_and_m6_reuses_vectors(files):
    req, _, meta = adapt_files(*files[:4])
    assert meta["status"] == "declared_compatible"
    prepared, warnings = prepare_request(req, Encoder(), Path("."))
    assert len(prepared.social_references) == 2 and not warnings
    assert prepared.context.model == MODEL_IDENTITY


def test_text_only_explicit_fallback_runs_without_image_vectors(files):
    _, req, _ = adapt_files(*files[:3])
    assert req.context_embedding is None
    prepared, warnings = prepare_request(req, Encoder(), Path("."))
    assert len(prepared.social_references) == 2 and len(warnings) == 2


def test_declared_different_models_remain_rejected(files):
    info = files[3]
    m = json.loads(info.read_text())
    m["social_images"]["revision"] = "different"
    info.write_text(json.dumps(m))
    req, _, meta = adapt_files(*files[:4])
    assert meta["status"] == "declared_incompatible"
    with pytest.raises(ValueError, match="incompatible"):
        prepare_request(req, Encoder(), Path("."))


@pytest.mark.parametrize(
    "change",
    ["context_shape", "image_count", "nan", "zero", "duplicate", "nonfloat", "missing_key"],
)
def test_bad_tensor_or_pairing_rejected(files, change):
    pt, text, context, info, payload, _ = files
    if change == "context_shape":
        payload["context_embedding"] = torch.ones(512)
    if change == "image_count":
        payload["top_image_embeddings"] = torch.ones(3, 512)
    if change == "nan":
        payload["top_image_embeddings"][0, 0] = float("nan")
    if change == "zero":
        payload["context_embedding"] = torch.zeros(1, 512)
    if change == "duplicate":
        payload["top_images"][1]["image_id"] = "b"
    if change == "nonfloat":
        payload["context_embedding"] = torch.ones(1, 512, dtype=torch.int64)
    if change == "missing_key":
        del payload["top_images"]
    torch.save(payload, pt)
    with pytest.raises(ValueError):
        adapt_files(pt, text, context)


def test_safe_load_does_not_execute_pickle_functions(files):
    torch.save({"dangerous": eval}, files[0])
    with pytest.raises(Exception):
        adapt_files(*files[:3])


def test_empty_or_bad_context_rejected(files):
    files[1].write_text("   ")
    with pytest.raises(ValueError):
        adapt_files(*files[:3])
    files[1].write_text("valid")
    files[2].write_text("{}")
    with pytest.raises(ValueError):
        adapt_files(*files[:3])


def test_profile_dimension_mismatch(files):
    m = json.loads(files[3].read_text())
    m["context"]["dimension"] = 768
    files[3].write_text(json.dumps(m))
    with pytest.raises(ValueError, match="dimension"):
        adapt_files(*files[:4])


def test_cli_outputs_and_input_unchanged(files, monkeypatch):
    from travel_agent.products.retrieval.upstream.__main__ import main

    before = [p.read_bytes() for p in files[:3]]
    output = files[0].parent / "adapted"
    monkeypatch.setattr(
        "sys.argv",
        [
            "upstream",
            "--pt",
            str(files[0]),
            "--text",
            str(files[1]),
            "--context",
            str(files[2]),
            "--output-dir",
            str(output),
        ],
    )
    assert main() == 0
    assert [p.read_bytes() for p in files[:3]] == before
    assert {p.name for p in output.iterdir()} == {
        "request.json",
        "request_text_only.json",
        "metadata.json",
    }
