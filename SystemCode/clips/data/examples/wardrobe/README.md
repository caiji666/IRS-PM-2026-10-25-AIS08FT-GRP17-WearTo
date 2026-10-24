# Shared wardrobe baseline

These 21 PNG images are the approved shared integration baseline: 20 images from the user-supplied wardrobe archive and one existing wardrobe example. Repository upload was explicitly approved on 2026-10-21. This permission does not assert a separate public redistribution license. They represent owned-item test inputs, not purchasable listings or verified inventory.

`manifest.json` lists relative paths, checksums and dimensions. Filenames are descriptive only; category, color and style must still be inferred by the wardrobe pipeline, including UNKNOWN when uncertain. The images contain no embedded metadata. No user profile, credentials, absolute local paths or model vectors are included.

## Import after cloning

From the repository root, after installing product dependencies and caching FashionCLIP as described in the product runbook:

```bash
.venv/bin/python -m travel_agent.products.retrieval.phase2 --offline \
  ingest --images data/examples/wardrobe/images/*.png \
  --output runs/wardrobe/baseline_import.json
```

This generates FashionCLIP embeddings and structured wardrobe items in `runs/wardrobe/items.json`, processed images in `runs/wardrobe/images/`, and import diagnostics in `runs/wardrobe/baseline_import.json`. Review diagnostics for any failed images. Existing compatible items are retained and duplicate images are deduplicated. This is an explicit import, not an automatic replacement of a user's wardrobe.

The profile page and live Student 1–4 runner read that same wardrobe store. Restart the local web server if needed and refresh the profile page. Subsequent private uploads and generated outputs remain Git-ignored; this shared directory is not an upload destination.
