"""Private, single-device profile and wardrobe views for the loopback UI."""

import io
import json
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Annotated

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
PORTRAIT_REF = "current"


def portrait_path(root):
    return Path(root) / "runtime/uploads/portraits" / f"{PORTRAIT_REF}.png"


def portrait_view(root):
    path = portrait_path(root)
    return {
        "uploaded": path.is_file(),
        "portrait_ref": PORTRAIT_REF if path.is_file() else None,
        "image": "/portrait-media/current.png" if path.is_file() else None,
        "consent": path.is_file(),
    }


def save_portrait(root, content):
    path = portrait_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".png")
    os.close(fd)
    try:
        decode_upload(content, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return portrait_view(root)


def delete_portrait(root):
    portrait_path(root).unlink(missing_ok=True)
    return portrait_view(root)


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)
    name: Annotated[str, StringConstraints(max_length=80)] = ""
    age: int | None = Field(default=None, ge=1, le=120, strict=True)
    height_cm: float | None = Field(default=None, ge=30, le=250, strict=True)
    weight_kg: float | None = Field(default=None, ge=1, le=400, strict=True)


def atomic_record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def profile_view(root):
    path = root / "runs/web_profile/profile.json"
    return (
        Profile.model_validate_json(path.read_text()).model_dump()
        if path.exists()
        else (Profile().model_dump())
    )


def save_profile(root, data):
    value = Profile.model_validate(data).model_dump()
    atomic_record(root / "runs/web_profile/profile.json", value)
    return value


def wardrobe_view(root):
    from travel_agent.products.retrieval.phase2.service import load_store

    store = load_store(root / "runs/wardrobe/items.json")
    labels_path = root / "runs/web_profile/wardrobe_labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    result = []
    for item in store.items:
        digest = item.product_id.removeprefix("wardrobe:")
        # Only images managed by the existing Phase 2 service may be served.
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            continue
        image = root / "runs/wardrobe/images" / f"{digest}.png"
        result.append(
            {
                "id": item.product_id,
                "title": labels.get(item.product_id, "My wardrobe piece"),
                "image": f"/wardrobe-media/{digest}.png" if image.is_file() else None,
                "category": item.category.value,
                "subcategory": item.subcategory,
                "colors": item.colors,
                "styles": item.styles,
                "owned": True,
            }
        )
    return {"items": result, "count": len(result)}


def decode_upload(content, target):
    """Decode bounded raster bytes, normalize orientation and strip photo metadata."""
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose an image smaller than 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Please upload a JPG, PNG or WebP image.")
                if source.width * source.height > 20_000_000:
                    raise ValueError("Please choose an image under 20 megapixels.")
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.save(target, format="PNG")
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError("This image could not be read. Try a different photo.") from exc


def ingest_upload(root, directory):
    from travel_agent.products.retrieval.m5 import CachedEncoder, FashionCLIPEncoder
    from travel_agent.products.retrieval.phase2.service import ingest, load_store

    upload = json.loads((directory / "upload.json").read_text())
    path = directory / "upload.png"
    store_path = root / "runs/wardrobe/items.json"
    before = {i.product_id for i in load_store(store_path).items}
    encoder = CachedEncoder(
        FashionCLIPEncoder(root / ".cache/huggingface", offline=True),
        root / ".cache/embeddings",
    )
    store, issues = ingest([path], encoder, store_path, root / "runs/wardrobe/images")
    imported = [
        i
        for i in store.items
        if i.provenance.original.get("phase2_upload", {}).get("original_path")
        == str(path.resolve())
    ]
    if issues or not imported:
        raise ValueError("The photo could not be processed. Try a clear single-garment image.")
    labels_path = root / "runs/web_profile/wardrobe_labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    for item in imported:
        labels.setdefault(item.product_id, upload["title"])
    atomic_record(labels_path, labels)
    result = {"count": len(store.items), "added": len({i.product_id for i in store.items} - before)}
    atomic_record(directory / "wardrobe_result.json", result)
    return result
