"""Real FashionCLIP encoder. No random/hash/mock embeddings in production."""

import hashlib
import os
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol

from .models import Vector, vector

MODEL_ID = "patrickjohncyh/fashion-clip"
MODEL_REVISION = "7e3ba62ce16b379a1ab479346b66f192e76f51b7"
try:
    TRANSFORMERS_VERSION = version("transformers")
except PackageNotFoundError:
    TRANSFORMERS_VERSION = "not-installed"
MODEL_IDENTITY = (
    f"{MODEL_ID}@{MODEL_REVISION}:clip-rgb-exif-text77-v1:transformers-{TRANSFORMERS_VERSION}"
)


class Encoder(Protocol):
    identity: str
    dimension: int

    def encode_texts(self, texts): ...
    def encode_images(self, images): ...


class FashionCLIPEncoder:
    identity = MODEL_IDENTITY
    dimension = 512

    def __init__(self, model_cache, offline=False, device="cpu", batch_size=16):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if device not in {"cpu", "mps"}:
            raise ValueError("device must be cpu or mps")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is not available; use CPU")
        options = dict(
            revision=MODEL_REVISION, cache_dir=str(model_cache), local_files_only=offline
        )
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=False, **options)
        self.model = (
            CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True, **options).to(device).eval()
        )
        self.dimension = self.model.config.projection_dim
        self.device, self.batch_size = device, batch_size

    def _encode(self, items, modality):
        import torch

        items = list(items)
        result = []
        with torch.inference_mode():
            for start in range(0, len(items), self.batch_size):
                batch = items[start : start + self.batch_size]
                if modality == "text":
                    if any(not isinstance(x, str) or not x.strip() for x in batch):
                        raise ValueError("text encoding requires nonempty strings")
                    inputs = self.processor(
                        text=batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.model.config.text_config.max_position_embeddings,
                    )
                    features = self.model.get_text_features(**inputs.to(self.device))
                else:
                    inputs = self.processor(images=batch, return_tensors="pt")
                    features = self.model.get_image_features(**inputs.to(self.device))
                for row in features.cpu().float().numpy():
                    result.append(vector(self.identity, modality, row))
        return result

    def encode_texts(self, texts):
        return self._encode(texts, "text")

    def encode_images(self, images):
        return self._encode(images, "image")


class CachedEncoder:
    """Content + modality + complete model identity cache, atomic JSON writes."""

    def __init__(self, encoder: Encoder, cache_dir):
        self.encoder = encoder
        self.identity, self.dimension = encoder.identity, encoder.dimension
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _encode(self, items, modality):
        items = list(items)
        paths, results, pending, positions = [], [None] * len(items), [], []
        for i, item in enumerate(items):
            if modality == "text":
                if not isinstance(item, str) or not item.strip():
                    raise ValueError("text must be nonempty")
                content = item.encode("utf-8")
            else:
                content = str(item.size).encode() + item.mode.encode() + item.tobytes()
            key = hashlib.sha256(self.identity.encode() + modality.encode() + content).hexdigest()
            path = self.cache_dir / f"{key}.json"
            paths.append(path)
            try:
                cached = Vector.model_validate_json(path.read_text())
                if (
                    cached.model != self.identity
                    or cached.modality != modality
                    or len(cached.values) != self.dimension
                ):
                    raise ValueError("incompatible cache")
                results[i] = cached
                self.hits += 1
            except (OSError, ValueError):
                pending.append(item)
                positions.append(i)
                self.misses += 1
        generated = (
            (
                self.encoder.encode_texts(pending)
                if modality == "text"
                else self.encoder.encode_images(pending)
            )
            if pending
            else []
        )
        if len(generated) != len(pending):
            raise ValueError("encoder returned incorrect batch count")
        for i, value in zip(positions, generated):
            value = Vector.model_validate(value)
            if (
                value.model != self.identity
                or value.modality != modality
                or len(value.values) != self.dimension
            ):
                raise ValueError("encoder returned incompatible vector")
            results[i] = value
            # Same-directory replacement avoids partially written entries.
            fd, name = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as output:
                    output.write(value.model_dump_json())
                os.replace(name, paths[i])
            finally:
                if os.path.exists(name):
                    os.unlink(name)
        return results

    def encode_texts(self, texts):
        return self._encode(texts, "text")

    def encode_images(self, images):
        return self._encode(images, "image")
