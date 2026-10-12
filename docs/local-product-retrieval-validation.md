# Resource-complete local product retrieval checkout

The current local checkout now contains its own upstream input files, historical product photos, reference raster images, personal wardrobe raster, pinned FashionCLIP snapshot and private Sandbox credential file. The original standalone project is preserved but is not required by the commands below.

All commands run from this repository's root. The dependency environment is already installed on the current Mac. For another machine, follow [the fresh-checkout runbook](product-retrieval-runbook.md) to install dependencies and obtain excluded resources.

## Resource locations

| Resource | Repository-local location |
|---|---|
| Original PT/TXT/JSON inputs | `data/team_input/` |
| Ten downloaded reference images in PT order | `data/team_input/reference_images/000.jpg` through `009.jpg` |
| Historical 20-photo catalogue/manifest/images | `data/photos20/` |
| One actual user wardrobe photograph | `data/raw/wardrobe/owned-001.png` |
| Pinned FashionCLIP snapshot | `.cache/huggingface/` |
| Private Sandbox credentials | `.secrets/ebay-sandbox.json` |
| Persisted imported wardrobe | `runs/wardrobe/` |
| Shared tool cache/identity-specific wardrobe/API images | `artifacts/product-retrieval-runs/` |
| Outputs for this independent validation | `artifacts/independent-run/` |

Resources were copied as independent ordinary files, not symlinks into the original project. Input/data/model checksums were verified. The credential file has mode 0600 and its directory mode 0700. These resources remain ignored by Git; they are present on this Mac but will not be published with a source push. No credential value is included in documentation or validation summaries.

## 1. Configure the current checkout

In the VS Code terminal opened at the repository root:

```bash
export STUDENT3_DATA_ROOT="$PWD"
export STUDENT3_MODEL_CACHE="$PWD/.cache/huggingface"
export STUDENT3_WORK_ROOT="$PWD/artifacts/product-retrieval-runs"
export STUDENT3_EBAY_CREDENTIALS="$PWD/.secrets/ebay-sandbox.json"
mkdir -p artifacts/independent-run

.venv/bin/python -c "import sys, torch, transformers; print(sys.version); print(torch.__version__); print(transformers.__version__)"
```

Expected stack: Python 3.11.17, Torch 2.8.0 and Transformers 4.57.6. Run these exports in each new terminal to replace any old resource-root settings. All values above resolve inside this repository.

## 2. Load and normalize prepared products

```bash
.venv/bin/python -m travel_agent.products.retrieval.cli \
  --catalogue data/catalogue.json --request fixtures/request.json \
  --output artifacts/independent-run/catalogue-response.json

.venv/bin/python -m travel_agent.products.retrieval.m3 \
  --catalogue data/catalogue_phase1.json \
  --output artifacts/independent-run/normalized-catalogue.json
```

The first output has five original prepared catalogue matches. The second has 34 normalized products and two reported category issues; uncertain categories remain UNKNOWN. These are fictional catalogue records, not live purchasing offers.

## 3. Verify the local model and create a prepared-product index

```bash
.venv/bin/python -m travel_agent.products.retrieval.m5.download \
  --model-cache "$STUDENT3_MODEL_CACHE"

.venv/bin/python -m travel_agent.products.retrieval.retrieval_cli \
  --model-cache "$STUDENT3_MODEL_CACHE" --offline \
  index --normalized artifacts/independent-run/normalized-catalogue.json \
  --output artifacts/independent-run/product-index.json

.venv/bin/python -m travel_agent.products.retrieval.retrieval_cli \
  --model-cache "$STUDENT3_MODEL_CACHE" --offline \
  retrieve --index artifacts/independent-run/product-index.json \
  --request fixtures/phase1_request.json --config fixtures/m6_config.json \
  --per-category --k-per-category 5 \
  --output artifacts/independent-run/category-top-k.json \
  --legacy-output artifacts/independent-run/legacy-retrieval-response.json
```

The model verifier checks the pinned snapshot/weight hash. The copied files are already complete, so no model download was needed in this validation. If a required file is missing or invalid on a future run, this downloader uses the network to restore it.

`product-index.json` contains real FashionCLIP **text** vectors for the fictional prepared products; the catalogue does not have verified retail raster images. `category-top-k.json` contains independent category rankings and context/social scores. `legacy-retrieval-response.json` is the projection into the existing M1 response interface. Section 7 uses the actual historical product rasters instead.

The original combined retrieval demo remains available:

```bash
.venv/bin/python -m travel_agent.products.retrieval.retrieval_cli \
  --model-cache "$STUDENT3_MODEL_CACHE" --offline \
  demo --catalogue data/catalogue_phase1.json \
  --request fixtures/phase1_request.json \
  --output artifacts/independent-run/retrieval-demo.json
```

## 4. Generate the structured understanding example

```bash
.venv/bin/python -m travel_agent.understanding.demo \
  --output artifacts/independent-run/understanding-output.json
```

This produces a ready Student 1 `ModuleResult` with trip facts, expanded clothing context, budget and provenance. The example deliberately uses synthetic weather and draft knowledge. It tests the interface, not live weather or real Qwen inference. Real upstream results can replace this file if they have `status=ok` and `ready_for_downstream=true`.

## 5. Read and validate the original teammate files

```bash
.venv/bin/python -m travel_agent.products.retrieval.upstream \
  --pt data/team_input/phase_2_output.pt \
  --text data/team_input/expanded_fashion_context.txt \
  --context data/team_input/context.json \
  --output-dir artifacts/independent-run/upstream
```

This writes adapted `request.json`, `request_text_only.json` and `metadata.json`. The input has one 512-dimensional context vector and ten reference-image rows. Model identity remains unverified, so direct reuse of the imported vectors is blocked. The shared workflow instead re-encodes the ten repository-local reference images with our own model, while retaining metadata/provenance. No silent model relabelling occurs.

## 6. Understand and persist the actual personal wardrobe image

```bash
.venv/bin/python -m travel_agent.products.retrieval.phase2 \
  --model-cache "$STUDENT3_MODEL_CACHE" --offline \
  --store runs/wardrobe/items.json --image-dir runs/wardrobe/images \
  ingest --images data/raw/wardrobe/owned-001.png \
  --output artifacts/independent-run/wardrobe-import.json

.venv/bin/python -m travel_agent.products.retrieval.phase2 \
  --model-cache "$STUDENT3_MODEL_CACHE" --offline \
  --store runs/wardrobe/items.json --image-dir runs/wardrobe/images \
  bundle --catalogue data/catalogue_phase1.json \
  --request fixtures/phase1_request.json \
  --output artifacts/independent-run/wardrobe-prepared-bundle.json
```

The import preprocesses the raster, predicts attributes with abstention and saves a 512-dimensional image vector. It persisted one owned item with zero image issues. Uncertain attributes, including this item's category, remain UNKNOWN. Re-running with the same pixels updates the same item rather than duplicating it.

`wardrobe-prepared-bundle.json` combines the owned pool with prepared catalogue Top-K. It is distinct from the image-based historical-photo workflow below.

## 7. Run the shared tool with reference and product images

```bash
.venv/bin/python -m travel_agent.products.demo \
  --input examples/products/upstream-photos.json \
  --understanding artifacts/independent-run/understanding-output.json \
  --wardrobe-from runs/wardrobe/items.json \
  --output artifacts/independent-run/photo-products-output.json
```

This is the main offline integration workflow: original teammate metadata → ten local reference rasters → local FashionCLIP image encoding → twenty historical candidate photo embeddings → context/social MAX scoring → four independent Top-5 groups → owned pool plus candidate output.

Verified terminal result:

```json
{"status":"ok","reason":null,"returned_per_category":{"TOP":5,"BOTTOM":5,"FOOTWEAR":5,"OUTERWEAR":5}}
```

The JSON contains 20 candidate items and one actual owned item. References are not treated as owned clothes. Historical candidates have no real price/stock information, and the owned item's category is UNKNOWN; therefore the strict priced `data.candidates` list is empty, while all 21 items remain in `data.bundle`. This is usable prototype outfit input, not a validated purchasing recommendation.

## 8. Optional all-in-one historical-photo smoke

```bash
.venv/bin/python -m travel_agent.products.retrieval.photo_dataset \
  --offline --model-cache "$STUDENT3_MODEL_CACHE" \
  --dataset data/photos20 --request fixtures/phase1_request.json \
  --wardrobe-dir artifacts/independent-run/simulated-photo-wardrobe \
  --output artifacts/independent-run/photos20-smoke.json
```

This separately tests all 20 photos as both candidates and **simulated owned uploads**. It produced 20 image vectors, 20 simulated owned items and zero image issues. It does not use or change the actual personal wardrobe pool. Its Top-5 includes all five eligible candidates per class, so it is an execution smoke, not a ranking-quality study.

## 9. Authenticate and call the actual Sandbox API

```bash
.venv/bin/python -m travel_agent.products.retrieval.ebay \
  --credentials .secrets/ebay-sandbox.json auth-check

.venv/bin/python -m travel_agent.products.demo \
  --input examples/products/upstream-ebay.json \
  --understanding artifacts/independent-run/understanding-output.json \
  --wardrobe-from runs/wardrobe/items.json \
  --output artifacts/independent-run/ebay-tool-output.json
```

Authentication succeeded using the new checkout's own credential file. The shared tool made ten successful Browse image-search requests with zero API issues, but returned no candidates. The resulting status is `no_candidates`, with four empty product groups and the one actual owned item retained. No historical candidates are substituted as eBay responses. Downloaded API reference rasters are under `artifacts/product-retrieval-runs/ebay/reference_images/`.

Nonempty Sandbox/Production product retrieval has not been demonstrated by this live run. Candidate-photo download/encoding/ranking is covered by mock API transport tests with real FashionCLIP. Production access, live stock and final shopping feasibility remain unverified.

## 10. Run all regression and formatting checks

```bash
ITSA_REAL_MODEL_TEST=1 .venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

The complete suite passed 231 tests. Ruff and formatting checks passed. These tests do not call eBay; section 9 is the separate live check.

## Output handoff and independence verification

Use `artifacts/independent-run/photo-products-output.json` for the current image-based prototype handoff. Its `data.bundle` contains owned items, category Top-K, vectors, scores and provenance; `data.handoff_blockers` records missing purchase-interface requirements. Supply referenced image files along with JSON when a downstream teammate needs the pictures on another machine.

Use `artifacts/independent-run/ebay-tool-output.json` to inspect the actual Sandbox result and raw search summaries. It currently has no candidate products. Retrieval never marks final budget/outfit feasibility as passed.

During validation, a temporary Python audit hook denied file access to the old standalone resource root. The denial was explicitly tested before running the workflows. All primary new outputs contain only new-checkout resource paths, and every referenced owned/product image exists inside this repository. Model/data resources are ordinary copied files with no links into the old project. The temporary guard is a validation tool outside this repository; users do not need it to run the project.
