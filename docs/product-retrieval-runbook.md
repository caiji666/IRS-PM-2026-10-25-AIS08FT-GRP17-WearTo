# Product retrieval runbook

This guide runs the product retrieval and wardrobe pipeline from a checkout of **this repository**. It does not require the original author's Desktop folder, an existing wardrobe store, or an existing model cache. The backend package is `travel_agent.products.retrieval`; the shared LangChain entry point is `travel_agent.products.demo` / `search_products`.

## 1. Obtain the version containing this module

**Publication status:** these changes currently exist only on the author's local `feat/student3-integration` branch. They have not been pushed. The clone command below becomes usable after that branch is published and you have access to the private repository. Downloading the current GitHub `main` branch will not provide this implementation until it is merged.

After publication:

```bash
git clone --branch feat/student3-integration \
  https://github.com/caiji666/travel-style-agent.git
cd travel-style-agent
```

If already cloned, save your own changes first, then fetch and check out the published feature branch. After the feature is merged, a normal clone of `main` is sufficient. For a ZIP download, select the appropriate branch on GitHub, extract it, and open the extracted repository root; Git commands are unnecessary for a ZIP checkout.

Check that these paths exist:

```text
pyproject.toml
uv.lock
travel_agent/products/retrieval/
examples/products/upstream-photos.json
docs/product-retrieval-runbook.md
```

## 2. Install the environment and set repository-local paths

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/) first. The commands below use Bash/Zsh syntax on macOS or Linux; Windows users can use WSL or translate directory/environment commands to PowerShell. `uv run` avoids depending on a platform-specific `.venv/bin/python` path.

Run every command below from the repository root:

```bash
uv python install 3.11
uv sync --locked --extra products

export STUDENT3_DATA_ROOT="$PWD"
export STUDENT3_MODEL_CACHE="$PWD/.cache/huggingface"
export STUDENT3_WORK_ROOT="$PWD/artifacts/product-retrieval-runs"

mkdir -p artifacts data/team_input data/raw/wardrobe

uv run --locked --extra products python -c \
  "import sys, torch, transformers; print(sys.version); print('torch:', torch.__version__); print('transformers:', transformers.__version__)"
```

The tested stack is Python 3.11, Transformers 4.57.6 and Torch 2.8.0. Transformers 5.19.0 was unavailable on official PyPI when checked; this guide does not install or claim compatibility with that version. Optional Qwen dependencies retain their existing shared range. Do not install the original standalone project's Transformers 4.46.3 requirements into this environment.

The existing `STUDENT3_*` environment variable names remain for compatibility. Their values now point to **your checkout**, not another person's computer. Set them again in a new terminal. All tool input paths must resolve under `STUDENT3_DATA_ROOT`, including symlinks; model and output locations are separately configured.

## 3. Understand what GitHub includes

| Included in Git | Obtain separately |
|---|---|
| Source, tests, request fixtures and fictional prepared catalogues | The three upstream PT/TXT/JSON files |
| Small synthetic image test fixture | Social reference raster images |
| Example tool payloads | The optional 20-photo historical dataset |
| Dependency lockfile and this guide | Personal wardrobe photographs |
| | Pinned FashionCLIP weights and your own eBay credentials |

PT files, complete datasets, personal photographs, credentials, weights, caches and generated outputs are intentionally ignored by Git. A fresh clone does not include the author's private inputs, saved wardrobe or Sandbox credentials. Do not commit those files after preparing them.

You can run sections 4 and 5 without teammate files, wardrobe photographs or eBay access. Sections 6–9 progressively add these resources.

The current author checkout has now been populated with independent local copies of these resources. See [the resource-complete local validation and command guide](local-product-retrieval-validation.md). This local preparation does not add private resources to Git or make them available in another person's clone.

## 4. Validate the catalogue and normalize products without a model

```bash
uv run --locked --extra products python -m travel_agent.products.retrieval.cli \
  --catalogue data/catalogue.json \
  --request fixtures/request.json \
  --output artifacts/catalogue-response.json

uv run --locked --extra products python -m travel_agent.products.retrieval.m3 \
  --catalogue data/catalogue_phase1.json \
  --output artifacts/normalized-catalogue.json

uv run --locked --extra products python -m travel_agent.products.retrieval.m3 \
  --input fixtures/m3_input.json \
  --output artifacts/normalization-example.json
```

Outputs:

- `catalogue-response.json`: validated M1/M2 request and prepared catalogue retrieval response.
- `normalized-catalogue.json`: cleaned, categorized and standardized products with issues/provenance.
- `normalization-example.json`: cleaning of intentionally imperfect example records.

The prepared catalogues contain fictional demonstration products. Their prices are synthetic and stock is unknown; they are not real purchasing offers. No exchange rates or missing stock information are invented. These commands do not load FashionCLIP or contact eBay.

## 5. Download FashionCLIP and run the included end-to-end example

The first command needs Internet access and `curl`. The downloader verifies a pinned revision and the weight SHA-256; allow roughly 605 MB for model weights plus small configuration/tokenizer files. Subsequent `--offline` commands require that completed download and do not download weights.

```bash
uv run --locked --extra products python -m travel_agent.products.retrieval.m5.download \
  --model-cache "$STUDENT3_MODEL_CACHE"

uv run --locked --extra products python -m travel_agent.products.retrieval.retrieval_cli \
  --model-cache "$STUDENT3_MODEL_CACHE" --offline \
  demo --catalogue data/catalogue_phase1.json \
  --request fixtures/phase1_request.json \
  --output artifacts/retrieval-demo.json
```

`retrieval-demo.json` includes normalization, the embedding index, global and per-category rankings, and the legacy M1 response. Prepared products use their text because no verified retail raster images are bundled. Reference descriptions are explicit text fallbacks. This validates real FashionCLIP text encoding and retrieval, not visual product-search accuracy.

To run index creation and retrieval separately:

```bash
uv run --locked --extra products python -m travel_agent.products.retrieval.retrieval_cli \
  --model-cache "$STUDENT3_MODEL_CACHE" --offline \
  index --normalized artifacts/normalized-catalogue.json \
  --output artifacts/product-index.json

uv run --locked --extra products python -m travel_agent.products.retrieval.retrieval_cli \
  --model-cache "$STUDENT3_MODEL_CACHE" --offline \
  retrieve --index artifacts/product-index.json \
  --request fixtures/phase1_request.json \
  --config fixtures/m6_config.json --per-category --k-per-category 5 \
  --output artifacts/category-top-k.json \
  --legacy-output artifacts/legacy-retrieval-response.json
```

`category-top-k.json` returns TOP/BOTTOM/FOOTWEAR/OUTERWEAR independently. Social similarity uses MAX across references; scores are semantic relevance, not outfit compatibility or purchasing validation.

### Call the shared LangChain tool using only included fixtures

```bash
uv run --locked --extra products python -m travel_agent.understanding.demo \
  --output artifacts/understanding-output.json

uv run --locked --extra products python - <<'PY'
import json
from pathlib import Path

payload = {
    "source": "prepared",
    "request": json.loads(Path("fixtures/phase1_request.json").read_text()),
    "k_per_category": 5,
}
Path("artifacts/prepared-tool-input.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

uv run --locked --extra products python -m travel_agent.products.demo \
  --input artifacts/prepared-tool-input.json \
  --understanding artifacts/understanding-output.json \
  --output artifacts/prepared-tool-output.json
```

This calls the actual `search_products` LangChain tool. It does not require a chat-model API key or a separate agent. The output is a shared `ModuleResult` containing the owned pool (initially empty), category Top-K, scores, vectors and provenance.

The understanding example uses explicitly synthetic weather and draft knowledge. Only `status=ok` and `ready_for_downstream=true` are accepted. Its complete original output, session state and hard constraints remain in a sidecar; the trip budget is not silently converted to a single-item price ceiling. Complete constraint validation remains downstream.

## 6. Import personal wardrobe images

Copy your own single-garment photograph to `data/raw/wardrobe/my-shirt.jpg`. Replace that filename below if necessary. No personal photograph is included in the repository.

```bash
uv run --locked --extra products python -m travel_agent.products.retrieval.phase2 \
  --model-cache "$STUDENT3_MODEL_CACHE" --offline \
  --store runs/wardrobe/items.json --image-dir runs/wardrobe/images \
  ingest --images data/raw/wardrobe/my-shirt.jpg \
  --output artifacts/wardrobe-import.json

uv run --locked --extra products python -m travel_agent.products.retrieval.phase2 \
  --model-cache "$STUDENT3_MODEL_CACHE" --offline \
  --store runs/wardrobe/items.json --image-dir runs/wardrobe/images \
  bundle --catalogue data/catalogue_phase1.json \
  --request fixtures/phase1_request.json \
  --output artifacts/wardrobe-bundle.json

uv run --locked --extra products python -m travel_agent.products.demo \
  --input artifacts/prepared-tool-input.json \
  --understanding artifacts/understanding-output.json \
  --wardrobe-from runs/wardrobe/items.json \
  --output artifacts/prepared-tool-with-wardrobe.json
```

The import decodes/orients the image, predicts category/subcategory/color/style with abstention, creates a 512-dimensional FashionCLIP image vector and persists an owned item. UNKNOWN is retained when predictions are uncertain. `runs/wardrobe/items.json` stores the structured wardrobe; `runs/wardrobe/images/` stores managed images. The shared tool re-ingests these rasters into a runtime-identity-specific store under `artifacts/product-retrieval-runs/wardrobe/`.

Omit `--wardrobe-from` when you do not have a wardrobe. A reference image never becomes an owned garment automatically. Use a new store or re-ingest original pictures after changing model identity; never relabel old embeddings.

A native picker is optional: replace `--images ...` with `--pick`. It requires Tk support, which some Python installations lack; explicit image paths work without the picker.

To check raw image understanding with included synthetic/error fixtures:

```bash
uv run --locked --extra products python -m travel_agent.products.retrieval.retrieval_cli \
  --model-cache "$STUDENT3_MODEL_CACHE" --offline \
  wardrobe --uploads fixtures/m4_uploads.json \
  --output artifacts/wardrobe-fixture-report.json
```

That fixture deliberately contains one synthetic image and one missing file. A reported missing-image issue is expected; this is not a clothing-recognition accuracy test.

## 7. Add the original teammate PT/TXT/JSON input

Obtain the following three files from the upstream contributor through the team's agreed private channel, and place them here:

```text
data/team_input/phase_2_output.pt
data/team_input/expanded_fashion_context.txt
data/team_input/context.json
```

Keep the original filenames to use the provided example payloads. The adapter safely loads PT with `torch.load(weights_only=True, map_location="cpu")`, validates finite floating `(1,512)` context and `(N,512)` reference tensors, and pairs rows with metadata in their original order. The current provided file has 10 reference images. Only load artifacts from the trusted upstream contributor; safe-loading restrictions do not authenticate a file's origin.

```bash
uv run --locked --extra products python -m travel_agent.products.retrieval.upstream \
  --pt data/team_input/phase_2_output.pt \
  --text data/team_input/expanded_fashion_context.txt \
  --context data/team_input/context.json \
  --output-dir artifacts/upstream
```

Outputs are `request.json`, `request_text_only.json` and `metadata.json`. Without complete model metadata, the status is `pending_model_identity`: the original imported vectors remain blocked from direct ranking. This does not prevent reading the files or recomputing embeddings locally.

### Obtain reference raster images for visual comparison

The offline shared tool does not download URLs. Download the provided references first, or obtain the corresponding cached images from the upstream contributor. Number them `000.jpg`, `001.jpg`, etc. in exact PT metadata order under `data/team_input/reference_images/`.

For the current Unsplash inputs, this command downloads and preprocesses them with the existing bounded image downloader. It requires Internet access:

```bash
uv run --locked --extra products python - <<'PY'
import json
from pathlib import Path

from travel_agent.products.retrieval.ebay.client import download_photo

metadata = json.loads(Path("artifacts/upstream/metadata.json").read_text())
for index, row in enumerate(metadata["top_images"]):
    target = Path("data/team_input/reference_images") / f"{index:03d}.jpg"
    download_photo(row["image_url"], target, {"images.unsplash.com"})
    print(f"Saved reference {index}: {target}")
PY
```

This snippet supports the current `images.unsplash.com` URLs. Other hosts require a reviewed downloader configuration; do not silently disable URL/redirect checks. Without usable local reference rasters, the offline adapter explicitly falls back to descriptions where available; that run is not image-to-image comparison. The eBay pipeline downloads reference rasters itself.

### Run teammate inputs against the included prepared catalogue

```bash
uv run --locked --extra products python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("examples/products/upstream-photos.json").read_text())
payload["source"] = "prepared"
Path("artifacts/upstream-prepared-input.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

uv run --locked --extra products python -m travel_agent.products.demo \
  --input artifacts/upstream-prepared-input.json \
  --understanding artifacts/understanding-output.json \
  --output artifacts/upstream-prepared-output.json
```

Add `--wardrobe-from runs/wardrobe/items.json` if section 6 created a wardrobe. This uses teammate references and the understanding context, but candidate products still have text embeddings from the fictional catalogue. Section 8 adds actual candidate raster photos; section 9 requests candidates from eBay.

The default `reference_mode=recompute` preserves PT metadata but re-encodes reference images with the local encoder. Direct vector reuse requires confirmed model ID, revision, preprocessing, installed library version, dimension and normalization. An example metadata structure is in `fixtures/upstream_model_info_local_example.json`; it is not proof of the upstream configuration and is not a ready-made compatibility certificate for this environment. After verification, supply `upstream.model_info` in a tool payload or `--model-info` to the adapter and select `reference_mode=imported`. Unknown/incompatible vectors are never silently relabelled.

## 8. Optional historical 20-photo candidate dataset

This dataset is **not in Git**. Obtain the prepared `Photos20-dataset.zip` from the project maintainer, or obtain/recreate its catalogue, manifest, source/license metadata and matching images from the original acquisition records. Do not assume a GitHub clone supplies these files. The current archive already contains a `data/photos20/` prefix:

```bash
# Put the privately obtained archive at data/raw/Photos20-dataset.zip first.
unzip data/raw/Photos20-dataset.zip 'data/photos20/*' -d .
```

Verify the extracted layout:

```text
data/photos20/catalogue.json
data/photos20/manifest.json
data/photos20/source-license-metadata.json
data/photos20/images/...
```

Use the existing manifest and matching images as a set; do not substitute arbitrary photos without updating provenance. Runtime checks reject missing files, mismatched catalogue/manifest IDs, path escapes and checksum mismatches. Follow the source/license information; historical retail photographs are not live listings or a guarantee of redistribution rights.

With sections 5 and 7 complete:

```bash
uv run --locked --extra products python -m travel_agent.products.demo \
  --input examples/products/upstream-photos.json \
  --understanding artifacts/understanding-output.json \
  --output artifacts/photo-products-output.json
```

Add `--wardrobe-from runs/wardrobe/items.json` to include your wardrobe. `photo-products-output.json` contains actual candidate image embeddings, reference image embeddings when downloaded, and independent category Top-K. There are five candidates per category in this dataset, so Top-5 returns all eligible category members; this is an execution test, not a ranking-quality evaluation. Prices and stock remain unknown.

The independent all-in-one dataset smoke is also available:

```bash
uv run --locked --extra products python -m travel_agent.products.retrieval.photo_dataset \
  --offline --model-cache "$STUDENT3_MODEL_CACHE" \
  --dataset data/photos20 --request fixtures/phase1_request.json \
  --wardrobe-dir artifacts/simulated-photo-wardrobe \
  --output artifacts/photos20-smoke.json
```

This last command intentionally reuses all 20 photos as **simulated owned uploads**. Its `owned_items` are not your personal wardrobe. The shared tool command above keeps candidate photos separate from personal ownership.

## 9. Configure your own eBay Sandbox credentials

You need your own eBay Developers Sandbox keyset. No author credentials are supplied. Use the Sandbox **App ID / Client ID** and **Cert ID / Client Secret**; the Dev ID is not the OAuth client secret. Keep secrets out of tool payloads, screenshots, chat, Git and command history.

If Tk is available, open the existing local masked form:

```bash
uv run --locked --extra products python -m travel_agent.products.retrieval.ebay \
  --credentials .secrets/ebay-sandbox.json configure
```

Select `sandbox`, enter your own credentials and save. If Tk is unavailable, create `.secrets/ebay-sandbox.json` in a local editor with the structure below, replace the placeholder values, and restrict file permissions. Do not run with placeholders:

```json
{
  "environment": "sandbox",
  "client_id": "YOUR_SANDBOX_APP_ID",
  "client_secret": "YOUR_SANDBOX_CERT_ID"
}
```

```bash
chmod 600 .secrets/ebay-sandbox.json
export STUDENT3_EBAY_CREDENTIALS="$PWD/.secrets/ebay-sandbox.json"

uv run --locked --extra products python -m travel_agent.products.retrieval.ebay \
  --credentials .secrets/ebay-sandbox.json auth-check
```

Successful authentication prints a confirmation without displaying or saving the token. This step needs Internet access but not FashionCLIP or teammate files. For the full workflow, first complete the model download and section 7's upstream file preparation:

```bash
uv run --locked --extra products python -m travel_agent.products.demo \
  --input examples/products/upstream-ebay.json \
  --understanding artifacts/understanding-output.json \
  --output artifacts/ebay-tool-output.json
```

Optionally add `--wardrobe-from runs/wardrobe/items.json`. This invokes OAuth, downloads reference images, performs Browse image search, downloads candidate images when available, encodes them and returns category Top-K alongside the owned pool. The default shared workflow uses EBAY_US and up to 10 references / 40 unique candidates.

For direct backend access without the shared understanding overlay:

```bash
uv run --locked --extra products python -m travel_agent.products.retrieval.ebay \
  --credentials .secrets/ebay-sandbox.json --marketplace EBAY_US \
  run --pt data/team_input/phase_2_output.pt \
  --text data/team_input/expanded_fashion_context.txt \
  --context data/team_input/context.json \
  --model-cache "$STUDENT3_MODEL_CACHE" \
  --max-references 10 --limit 20 --max-products 40 --k-per-category 5 \
  --wardrobe-store runs/wardrobe/items.json \
  --output-dir runs/ebay_sandbox
```

The direct CLI writes `runs/ebay_sandbox/result.json`; reference/product images are saved in that directory. Its exit codes are 0 for nonempty completion, 3 for no usable API products, and 2 for errors. The shared tool instead returns `no_candidates` for an empty candidate pool and its demo exits 0 for that explicit result; inspect the JSON status, not just the shell exit code.

The last live Sandbox verification returned ten successful searches with zero candidates. Empty results are possible and do not justify inventing products or substituting the historical dataset as API output. Sandbox data/prices are synthetic and stock is unknown. Production access and nonempty live product retrieval remain separately unverified. The back half has been tested with mock API responses and real FashionCLIP.

## 10. Read the output and hand it to the outfit module

Shared tool outputs are `ModuleResult` JSON:

| Field | Purpose |
|---|---|
| `status` / `reason` | Explicit `ok`, `needs_input`, `no_candidates` or `error` outcome |
| `data.bundle.owned_items` | Complete owned pool, attributes, image paths and vectors, including UNKNOWN |
| `data.bundle.top_k_products` | Independent category Top-K with structured items and vectors |
| `data.bundle.product_ranking` | Context/social similarities, MAX aggregation, weights and scores |
| `data.upstream_metadata` | Original upstream information, input hashes, requirements and source traces |
| `data.request_used` | Exact adapted retrieval request |
| `data.candidates` | Items representable in the unchanged shared `ProductCandidate` interface |
| `data.handoff_blockers` | Items that cannot yet be projected and their reasons |
| `data.feasibility_status` | Always `unverified` at retrieval time |

The outfit module can use `bundle` for prototype combination work. Only known-category owned items with a known currency and zero additional cost, or products with verified purchase prices and supported currency precision, enter `candidates`. Historical photos without real prices do not become zero-cost purchasable products. All nonprojectable items remain in the bundle. Final budget, stock, shipping/tax, currency and other constraint validation remain downstream.

Output JSON contains local image paths; it does not embed the image bytes. When another machine needs the images, supply the referenced files through an approved channel and agree on path mapping. Vector/model compatibility must still be preserved.

## 11. Run checks at the appropriate resource level

With the products environment installed, the default suite requires no model, upstream files or credentials:

```bash
uv run --locked --extra products ruff check .
uv run --locked --extra products ruff format --check .
uv run --locked --extra products pytest -q
```

A clean checkout was verified with 225 passed and 6 skipped tests. Without `data/photos20`, its three dataset tests are skipped. The three real-model tests are opt-in. Do not enable every real-model test before obtaining the optional photo dataset: one eBay transport smoke also uses a historical photo.

After downloading FashionCLIP, run the model tests that only use included fixtures:

```bash
ITSA_REAL_MODEL_TEST=1 uv run --locked --extra products pytest -q \
  tests/product_retrieval/test_m456.py \
  tests/product_retrieval/test_phase1_completion.py
```

After also preparing the 20-photo dataset, run the complete suite:

```bash
ITSA_REAL_MODEL_TEST=1 uv run --locked --extra products pytest -q
```

The last full local verification passed 231 tests. That count describes the current test version with all optional resources, not a guarantee that every future checkout has the same test count. These tests do not call the live eBay API; section 9 is the separate live verification.

The repository's base CI uses `uv sync --locked` without heavy extras. Avoid running a base-only sync in your active products environment unless you intend to remove optional packages. Use a separate environment for a base-CI check:

```bash
UV_PROJECT_ENVIRONMENT=.venv-ci uv sync --locked
UV_PROJECT_ENVIRONMENT=.venv-ci uv run --locked pytest -q
```

## Troubleshooting and scope

- Missing upstream/dataset files: obtain the excluded resources and verify the paths in sections 7–8. A clone alone does not include them.
- Missing cached model while offline: finish the pinned download in section 5 first; `--offline` does not mean “use fake embeddings.”
- Missing reference rasters: the offline tool may use explicit description fallbacks; download the actual reference images to test visual similarity.
- Incompatible imported vectors or wardrobe store: confirm model/preprocessing metadata or re-encode original rasters into a new store; do not edit model labels to bypass compatibility checks.
- Tk errors: use explicit `--images` and the local-editor credentials fallback.
- Sandbox authentication errors: check environment/keyset and client-secret fields. Do not paste tokens/secrets into issues.
- `no_candidates`: inspect the recorded API responses/issues; do not interpret it as a successful nonempty product recommendation.
- Non-ready understanding output: finish upstream clarification/human review; the product adapter does not bypass `ready_for_downstream`.

This guide verifies structured understanding output plus upstream reference files through product retrieval and wardrobe output. Online social retrieval, final outfit generation/validation, real Qwen inference, a live chat-model agent and persistent sessions are not claimed completed by these commands. No model/network calls occur merely by importing the shared tools.
