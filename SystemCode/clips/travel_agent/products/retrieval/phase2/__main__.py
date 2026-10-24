import argparse
import sys
from pathlib import Path

from ..m3 import process_catalogue
from ..m5 import CachedEncoder, FashionCLIPEncoder, build_index
from ..m6 import RankedProductProvider
from ..schemas import ProductRetrievalRequest
from .models import WardrobeStore
from .service import atomic_json, combine, ingest, load_store


def main():
    p = argparse.ArgumentParser(description="Phase 2 local wardrobe import and unified output")
    p.add_argument("--model-cache", type=Path, default=Path(".cache/huggingface"))
    p.add_argument("--embedding-cache", type=Path, default=Path(".cache/embeddings"))
    p.add_argument("--offline", action="store_true")
    p.add_argument("--store", type=Path, default=Path("runs/wardrobe/items.json"))
    p.add_argument("--image-dir", type=Path, default=Path("runs/wardrobe/images"))
    commands = p.add_subparsers(dest="command", required=True)
    for name in ["ingest", "bundle"]:
        c = commands.add_parser(name)
        c.add_argument("--images", nargs="*", default=[], type=Path)
        c.add_argument(
            "--pick", action="store_true", help="open native file picker for single-garment photos"
        )
        c.add_argument("--output", type=Path, required=True)
        if name == "bundle":
            c.add_argument("--catalogue", type=Path, default=Path("data/catalogue_phase1.json"))
            c.add_argument("--request", type=Path, default=Path("fixtures/phase1_request.json"))
            c.add_argument("--k-per-category", type=int, default=5)
    a = p.parse_args()
    try:
        paths = list(a.images)
        if a.pick:
            import tkinter as tk
            from tkinter.filedialog import askopenfilenames

            window = tk.Tk()
            window.withdraw()
            try:
                paths.extend(
                    Path(x)
                    for x in askopenfilenames(
                        title="选择单件衣服照片",
                        filetypes=[("Garment photos", "*.png *.jpg *.jpeg *.webp")],
                    )
                )
            finally:
                window.destroy()
        if a.command == "ingest" and not paths:
            raise ValueError("provide --images or select photos with --pick")
        # Prevent user paths from overwriting source images or the persistent store.
        if a.output.resolve() == a.store.resolve() or {a.output.resolve(), a.store.resolve()} & {
            x.expanduser().resolve() for x in paths
        }:
            raise ValueError("output must differ from store and input images")
        encoder = CachedEncoder(
            FashionCLIPEncoder(a.model_cache, offline=a.offline), a.embedding_cache
        )
        store, issues = (
            ingest(paths, encoder, a.store, a.image_dir) if paths else (load_store(a.store), [])
        )
        if a.command == "bundle":
            req = ProductRetrievalRequest.model_validate_json(a.request.read_text())
            if req.wardrobe_uploads or req.wardrobe_garments:
                raise ValueError(
                    "Phase 2 uses persisted owned items; import photos via --images/--pick, do not also supply wardrobe request fields"
                )
            normalized = process_catalogue(a.catalogue)
            index = build_index(normalized.products, encoder, Path("."))
            ranked = RankedProductProvider(
                index, encoder, Path("."), processing_issues=normalized.issues
            ).retrieve_grouped(req, a.k_per_category)
            result = combine(store, ranked, encoder.identity, encoder.dimension, issues)
            result.warnings.extend(
                f"{i.stage} record {i.record_index}: {i.reason}" for i in normalized.issues
            )
            atomic_json(a.output, result)
        else:
            from typing import List

            from pydantic import Field

            from ..m4.wardrobe import WardrobeIssue

            # Same atomic persistence mechanism for the items; diagnostic report is separate.
            from ..schemas import Contract

            class ImportReport(Contract):
                wardrobe: WardrobeStore
                issues: List[WardrobeIssue] = Field(default_factory=list)

            atomic_json(a.output, ImportReport(wardrobe=store, issues=issues))
        print(f"Saved {a.output}; owned items={len(store.items)}, image issues={len(issues)}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
