"""Translate teammate artifacts without altering the M1-M6 contracts."""

import hashlib
import json
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field

from ..m5.encoder import MODEL_IDENTITY
from ..schemas import (
    Contract,
    Embedding,
    ExpandedFashionContext,
    ProductRetrievalRequest,
    SocialReference,
    Trip,
)


class ModelInfo(Contract):
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    preprocessing: str = Field(min_length=1)
    transformers_version: str = Field(min_length=1)
    dimension: int = Field(ge=1)
    normalization: Literal["L2"]

    @property
    def identity(self):
        return f"{self.model_id}@{self.revision}:{self.preprocessing}:transformers-{self.transformers_version}"


class UpstreamModels(Contract):
    # Separate profiles: text and images need not have been encoded identically.
    context: ModelInfo
    social_images: ModelInfo


def string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: nonempty string required")
    return value.strip()


def strings(value, label):
    if not isinstance(value, list):
        raise ValueError(f"{label}: list required")
    return [string(v, label) for v in value]


def tensor_rows(tensor, rows, label):
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise ValueError(f"{label}: floating tensor required")
    if tensor.ndim != 2 or tuple(tensor.shape) != (rows, 512):
        raise ValueError(f"{label}: expected ({rows},512)")
    t = tensor.detach().cpu().double()
    if not bool(torch.isfinite(t).all()):
        raise ValueError(f"{label}: nonfinite values")
    norms = t.norm(dim=1)
    if bool((norms < 1e-12).any()):
        raise ValueError(f"{label}: zero vector")
    return (t / norms[:, None]).tolist(), norms.tolist()


def adapt_files(pt_path, text_path, context_path, model_info_path=None):
    paths = {"pt": Path(pt_path), "text": Path(text_path), "context": Path(context_path)}
    for p in paths.values():
        if not p.is_file():
            raise ValueError(f"input file missing: {p}")
        if p.stat().st_size > 100_000_000:
            raise ValueError("input exceeds 100 MB")
    source = json.loads(paths["context"].read_text(encoding="utf-8-sig"))
    if not isinstance(source, dict):
        raise ValueError("context JSON must be an object")
    text = string(paths["text"].read_text(encoding="utf-8-sig"), "expanded text")
    destination = string(source.get("destination"), "destination")
    season = string(source.get("season"), "season")
    weather = source.get("weather_clothing_requirements")
    activities = source.get("activity_style_requirements")
    if not isinstance(weather, dict) or not isinstance(activities, dict):
        raise ValueError("weather and activity requirements must be objects")
    weather = {
        string(k, "weather key"): strings(v, "weather requirements") for k, v in weather.items()
    }
    activities = {
        string(k, "activity"): strings(v, "activity styles") for k, v in activities.items()
    }
    user_style = strings(source.get("user_style"), "user_style")
    notes = (
        "Season: "
        + season
        + ". Clothing requirements: "
        + "; ".join(k + ": " + ", ".join(v) for k, v in weather.items())
    )
    # Preserve activity-specific relationships in semantic text and full sidecar.
    notes += " . Activity requirements: " + "; ".join(
        k + ": " + ", ".join(v) for k, v in activities.items()
    )
    context = ExpandedFashionContext(
        trip=Trip(destination=destination),
        query=text,
        activities=list(activities),
        style_preferences=user_style,
        weather_notes=notes,
    )
    payload = torch.load(paths["pt"], map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("PT root must be a dictionary")
    for key in ["context_embedding", "top_images", "top_image_embeddings"]:
        if key not in payload:
            raise ValueError("PT missing " + key)
    images = payload["top_images"]
    if not isinstance(images, list) or not images:
        raise ValueError("top_images must be a nonempty list")
    if len(images) > 1000:
        raise ValueError("too many references")
    cv, cn = tensor_rows(payload["context_embedding"], 1, "context_embedding")
    iv, ins = tensor_rows(payload["top_image_embeddings"], len(images), "top_image_embeddings")
    digest = hashlib.sha256(paths["pt"].read_bytes()).hexdigest()
    unknown = "unverified-upstream:sha256:" + digest
    profiles = (
        UpstreamModels.model_validate_json(Path(model_info_path).read_text())
        if model_info_path
        else None
    )
    if profiles and (profiles.context.dimension != 512 or profiles.social_images.dimension != 512):
        raise ValueError("model-info dimension disagrees with supplied tensors")
    cm = profiles.context.identity if profiles else unknown
    sm = profiles.social_images.identity if profiles else unknown
    references = []
    ids = set()
    for row, values in zip(images, iv):
        if not isinstance(row, dict):
            raise ValueError("each image metadata entry must be an object")
        rid = string(row.get("image_id"), "image_id")
        if rid in ids:
            raise ValueError("duplicate reference ID: " + rid)
        ids.add(rid)
        # Metadata and vectors retain exact upstream list order, never re-sort here.
        references.append(
            SocialReference(
                reference_id=rid,
                url=row.get("source_url") or row.get("image_url"),
                description=row.get("description"),
                embedding=Embedding(model=sm, values=values),
            )
        )
    req = ProductRetrievalRequest(
        context=context,
        context_embedding=Embedding(model=cm, values=cv[0]),
        top_social_references=references,
        limit=20,
    )
    fallback = req.model_dump(mode="json")
    fallback["context_embedding"] = None
    for ref in fallback["top_social_references"]:
        ref["embedding"] = None
    fallback = ProductRetrievalRequest.model_validate(fallback)
    # Sidecar is deliberately not inserted into extra-forbid original M1 fields.
    compatible = cm == MODEL_IDENTITY and sm == MODEL_IDENTITY
    warnings = [
        "Model metadata is a supplied declaration, not independent proof of encoder configuration.",
        "Remote reference URLs are not downloaded; compatible image vectors can be used without local images.",
        "Text-only request re-encodes context plus mapped fields and descriptions, not the original social images.",
        "Weather/season requirements are semantic inputs; no new hard clothing-property filters are implemented.",
    ]
    if not profiles:
        warnings.insert(
            0,
            "Encoder identity unknown: imported-vector ranking must remain blocked until model metadata is confirmed.",
        )
    elif not compatible:
        warnings.insert(
            0,
            "Declared model profiles differ from local encoder: existing M6 must reject these imported vectors.",
        )
    # metadata must be JSON-compatible: fail clearly instead of dropping unsupported extra objects.
    json.dumps(images, allow_nan=False)
    metadata = {
        "schema_version": "upstream.adapter.1",
        "status": "declared_compatible"
        if compatible
        else ("declared_incompatible" if profiles else "pending_model_identity"),
        "inputs": {
            k: {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for k, p in paths.items()
        },
        "original_context": source,
        "expanded_fashion_context": text,
        "top_images": images,
        "reference_vector_order": [r.reference_id for r in references],
        "tensor_shapes": {
            "context_embedding": [1, 512],
            "top_image_embeddings": [len(images), 512],
        },
        "original_vector_norms": {"context": cn, "social_images": ins},
        "model_profiles": profiles.model_dump() if profiles else None,
        "local_model_identity": MODEL_IDENTITY,
        "imported_identities": {"context": cm, "social_images": sm},
        "model_info_path": str(Path(model_info_path).resolve()) if model_info_path else None,
        "warnings": warnings,
    }
    return req, fallback, metadata
