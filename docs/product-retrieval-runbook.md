# Product retrieval integration

The `search_products` LangChain tool calls `products.service.run`, which validates an explicit payload and lazily invokes the preserved `travel_agent.products.retrieval` backend. It returns a shared `ModuleResult`. No model loading, model download or network calls occur during import. The single main agent already registers this tool; no separate agent is introduced.

## Environment

Use Python 3.11 and `uv sync --locked --extra products`. The tested stack is Transformers 4.57.6 and Torch 2.8.0. The requested Transformers 5.19.0 was unavailable on official PyPI at verification; it has not been installed or claimed compatible. The shared `qwen` range stays unchanged (`>=4.51,<5`). Shared dependencies and the generated lockfile need team review before publishing.

The optional products extra includes Pandas, Pillow, NumPy, Torch, Transformers and requests. Base CI stays offline without these heavy optional dependencies; products tests run with the extra. Legacy dataset/real-model tests additionally require the local dataset/model cache and explicit `ITSA_REAL_MODEL_TEST=1`.

## Local data and credentials

For this Mac, use the existing standalone project as a local resource root:

```bash
export STUDENT3_DATA_ROOT="$HOME/Desktop/IntelligentTravelStyleAssistant"
export STUDENT3_MODEL_CACHE="$STUDENT3_DATA_ROOT/.cache/huggingface"
export STUDENT3_WORK_ROOT="$PWD/artifacts/product-retrieval-runs"
```

Paths supplied through tool inputs must remain under the resource root, including resolved symlinks. Secrets are configured through `STUDENT3_EBAY_CREDENTIALS`, never inside the LLM payload. Do not commit PT files, user photographs, full datasets, credentials, model weights, caches or generated runtime output. The original local data, model files and credentials remain in the standalone project. Other machines must obtain their own upstream artifacts, acquire the photo dataset from its original manifest instructions, and download the pinned FashionCLIP revision with `travel_agent.products.retrieval.m5.download`.

The three original teammate files remain the supported Student 2 input:

- `data/team_input/phase_2_output.pt`: safely loaded with `torch.load(weights_only=True, map_location="cpu")`; validates finite floating `(1,512)` context and `(N,512)` image tensors and metadata pairing order.
- `data/team_input/expanded_fashion_context.txt`.
- `data/team_input/context.json`.

`reference_mode=recompute` preserves PT metadata/provenance but uses our own encoder on reference raster files cached at `data/team_input/reference_images/000.jpg` or `runs/ebay_sandbox/reference_images/000.jpg`, in exact PT row order. Missing cached rasters fall back to descriptions with explicit warnings. No raster download is hidden in the offline path. The eBay path downloads reference images itself. Imported vectors require an explicit compatible model-info file; unknown vectors are never relabelled. Upstream transformer version alone is insufficient proof of compatible model weights/preprocessing.

## Run Student 1 and then Student 3

```bash
.venv/bin/python -m travel_agent.understanding.demo \
  --output artifacts/student1.json

.venv/bin/python -m travel_agent.products.demo \
  --input examples/products/upstream-photos.json \
  --understanding artifacts/student1.json \
  --wardrobe-from runs/wardrobe/items.json \
  --output artifacts/products-output.json
```

Student 1's default example explicitly uses synthetic weather and draft knowledge. It is an integration fixture, not live travel reasoning. Only `status=ok` and `ready_for_downstream=true` are accepted. Trip dates, activities, style and expanded context map into the unchanged Student 3 request. The complete original `ModuleResult`, session state, reasoning traces and hard constraints remain in a sidecar. Its outfit budget is not silently mapped to a per-item ceiling. Exclusions in the offline path support exact category names and title substrings only; complete hard-constraint validation remains Student 5's responsibility.

The products tool uses the 20 historical retail photos as candidate products, separately from the saved user wardrobe. Reference photographs are never treated as owned garments. Existing wardrobe rasters are re-ingested into a store isolated by runtime encoder identity; the original wardrobe JSON and original environment are retained. Explicit abstention preserves UNKNOWN attributes.

The old standalone `student3.photo_dataset` smoke still supports reusing the 20 photos as **simulated** owned uploads. The new shared demo does not do that.

## Sandbox

```bash
export STUDENT3_EBAY_CREDENTIALS="$STUDENT3_DATA_ROOT/.secrets/ebay-sandbox.json"
.venv/bin/python -m travel_agent.products.demo \
  --input examples/products/upstream-ebay.json \
  --understanding artifacts/student1.json \
  --wardrobe-from runs/wardrobe/items.json \
  --output artifacts/ebay-output.json
```

This invokes actual OAuth and Browse image search, downloads candidate pictures when available, encodes them and returns category Top-K. Empty Sandbox responses return `no_candidates`; no prepared products are substituted as API results. Sandbox products/prices are synthetic, stock stays unknown, and Production access/real inventory are not claimed. Errors remain explicit. Existing standalone eBay CLI commands are also retained.

## Output and handoff

`artifacts/products-output.json` is the shared ModuleResult:

- `data.bundle.owned_items`: complete owned pool, including UNKNOWN predictions, structured attributes and FashionCLIP vectors.
- `data.bundle.top_k_products`: independently ranked TOP/BOTTOM/FOOTWEAR/OUTERWEAR lists.
- `data.bundle.product_ranking`: context/social scores, weights and MAX social aggregation.
- `data.upstream_metadata`: full Student 1/2 provenance and original requirements.
- `data.candidates`: projection into unchanged shared `ProductCandidate` only for representable known categories and verified purchase prices, or owned items with known currency and zero additional purchase cost.
- `data.handoff_blockers`: items that cannot be projected (unknown category/currency, unverified price or excessive currency precision). These items remain in the bundle. Historical photos without real prices cannot become purchasable zero-cost candidates.

Student 4 can use the bundle for semantic prototype outfit work and consume representable `candidates` once prices are available. Student 5 must validate total budget, exchange/shipping/tax policies, stock and all other constraints before recommending a purchasable outfit. This output always leaves `feasibility_status=unverified`. Social retrieval, outfit generation, final validation, persistent sessions and a live LLM agent remain unfinished/unverified in this repository.

## Checks

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
STUDENT3_DATA_ROOT="$HOME/Desktop/IntelligentTravelStyleAssistant" \
  ITSA_REAL_MODEL_TEST=1 .venv/bin/python -m pytest -q
```

Copied legacy code/tests are formatted for the shared repository; this does not rewrite the standalone M1/M2 schemas or loader. The standalone encoder identity now records the installed Transformers version, and its processor explicitly uses the saved slow configuration. An optional request override was added to the eBay pipeline for Student 1 context; existing callers and defaults remain supported. Standalone and shared adapters must be kept in sync if backend logic changes.
