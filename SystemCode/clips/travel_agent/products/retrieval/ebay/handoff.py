"""Portable downstream bundle output; upstream paths are audit data, not dependencies."""

import copy
import hashlib
import json
import os
import shutil
from contextlib import nullcontext
from pathlib import Path

from PIL import Image


def write_handoff(report, target):
    from ...notifications.lifecycle import DeletedAccount, Registry, lifecycle_lock

    target = Path(target).resolve()
    production = report.get("environment") == "production"
    root = Path(os.environ["EBAY_MANAGED_ROOT"]).resolve() if production else None
    with lifecycle_lock(root) if production else nullcontext():
        registry = (
            Registry(root, root / ".cache/ebay-lifecycle/registry.sqlite3") if production else None
        )
        try:
            if registry and report.get("lifecycle_generation") != registry.generation():
                raise DeletedAccount("result invalidated; rerun retrieval")
            if target.exists():
                raise FileExistsError("handoff output already exists")
            output = {
                key: copy.deepcopy(report[key])
                for key in (
                    "schema_version",
                    "environment",
                    "marketplace",
                    "status",
                    "fetched_at",
                    "scoring",
                    "reference_mode",
                    "bundle",
                    "summary",
                )
                if key in report
            }
            output["upstream_provenance"] = {
                "inputs": {
                    key: {"filename": Path(value["path"]).name, "sha256": value["sha256"]}
                    for key, value in report.get("upstream_metadata", {}).get("inputs", {}).items()
                },
                "note": "Filenames and hashes are audit-only; upstream files are not needed to consume this bundle.",
            }
            sources = {}
            copies = {}

            def collect(value):
                if isinstance(value, dict):
                    for child in value.values():
                        collect(child)
                elif isinstance(value, list):
                    for child in value:
                        collect(child)
                elif isinstance(value, str) and Path(value).is_absolute():
                    path = Path(value)
                    if path.is_file() and path.suffix.lower() in {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".webp",
                        ".bmp",
                    }:
                        sources.setdefault(path.name, set()).add(path)

            collect(output)

            def rewrite(value, owned=False):
                if isinstance(value, dict):
                    owned = value.get("owned", owned)
                    for key, child in list(value.items()):
                        path = None
                        if isinstance(child, str) and Path(child).is_absolute():
                            candidate = Path(child)
                            if candidate in sources.get(candidate.name, set()):
                                path = candidate
                            elif key == "image_path":
                                raise ValueError("missing portable image source")
                            else:
                                value[key] = None
                        elif key == "image_path" and isinstance(child, str) and child:
                            matches = sources.get(Path(child).name, set())
                            if len(matches) != 1:
                                raise ValueError("unresolved portable image reference")
                            path = next(iter(matches))
                        if path:
                            with Image.open(path) as image:
                                image.verify()
                            folder = "wardrobe_images" if owned else "product_images"
                            name = (
                                hashlib.sha256(path.read_bytes()).hexdigest() + path.suffix.lower()
                            )
                            relative = f"{folder}/{name}"
                            copies[relative] = path
                            value[key] = relative
                        elif not isinstance(child, str):
                            rewrite(child, owned)
                elif isinstance(value, list):
                    for child in value:
                        rewrite(child, owned)

            rewrite(output)
            if registry:
                registry.register(target, "portable_handoff")
            target.mkdir(parents=True)
            for folder in ("product_images", "wardrobe_images"):
                (target / folder).mkdir()
            for relative, source in copies.items():
                shutil.copy2(source, target / relative)
            (target / "result.json").write_text(
                json.dumps(output, ensure_ascii=False, indent=2) + "\n"
            )
            (target / "README.md").write_text(
                "# Product handoff\n\n"
                "Read result.json: bundle.top_k_products and bundle.owned_items are the candidate pool. "
                "bundle.product_ranking.ranked_products includes item, score, context_similarity and social_similarity. "
                "Resolve image paths against THIS folder: handoff_dir / item['image_path']. "
                "FashionCLIP embeddings, prices, API inventory evidence and provenance are included. "
                "Upstream PT/TXT/JSON files are not needed. Raw diagnostics are stored separately.\n\n"
                "Inventory is an API estimate at fetched_at, not a checkout guarantee. "
                "This folder is registered for cleanup on the generating Mac; recipient copies require "
                "their own cleanup integration or coordinated deletion. Do not commit runtime/user data to Git.\n"
            )
            return target / "result.json"
        finally:
            if registry:
                registry.close()
