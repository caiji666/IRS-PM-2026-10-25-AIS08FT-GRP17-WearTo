# Dynamic outfit compatibility backend (Student 4)

This module consumes the **current** Student 3 result. It never falls back to the repository's historical `outfit-compatibility/result.json`. The website remains a Student 1–3 interface; the optional backend runner adds Student 4 without changing frontend behavior.

## Environment and normalized model

Use Python 3.11 and the existing pinned product encoder. Add the `outfits` extra to the existing product/Qwen/inspiration dependencies:

```bash
.tools/uv pip install --python .venv/bin/python -e '.[products,inspiration,qwen,outfits]'
.venv/bin/python -m travel_agent.outfits.train --reencode
```

Alternatively use a configured Python environment's pip. FashionCLIP must already be cached as described in the product runbook. Training reads the existing Polyvore CSV, category map and image archive. It re-encodes every training photo with the **same pinned FashionCLIP identity and per-garment L2 normalization** as Student 3, then trains StandardScaler + MLP (16 hidden units, alpha 0.1, early stopping, seed 42). Category order is OUTERWEAR, TOP, BOTTOM, FOOTWEAR; missing roles are zero-filled. The input has 2048 features.

Generated files are local and Git-ignored:

- `artifacts/outfits/normalized-v1/normalized_embeddings.npz`
- `embedding_manifest.json` (archive/vector checksums and encoder identity)
- `model.joblib`
- `manifest.json` (model hash/version, libraries, training hashes and evaluation)

After the first encoding pass, omit `--reencode` to retrain from the checksum-validated normalized cache. The old `.pkl` and old raw vectors remain intact and are not used by this service. `OUTFITS_MODEL_DIR` is an operator-side environment setting for an alternative trusted local model directory; JSON payloads cannot select pickle files. Loading checks the manifest, checksum, category order, dimension and scikit-learn version. Never load arbitrary user-supplied model files.

The local re-encoded model's validation accuracy is 56.86% and test accuracy 60.50%. The outfit-row split has 384 shared product IDs between train/test. These are prototype metrics, not a product-disjoint evaluation or a quality guarantee on eBay/wardrobe data. Improving evaluation remains separate work.

## Public entry and input

`travel_agent.outfits.service.run(payload) -> ModuleResult` is the formal entry used by `build_and_score_outfits`. The old `compatibility_service.py` is only an import facade for that same function. It no longer implicitly reads a fixed sample.

Provide either:

- `products`: the actual Student 3 data/report containing `bundle`; or
- `product_result_path`: the current portable `result.json`, resolved relative to its own folder.

Also pass `understanding` (Student 1 data or its ModuleResult JSON) and, for a portable report, `ranking_request` from the corresponding diagnostic `request_used`. An in-memory diagnostic report already carries those scoring signals in `request_used`. This is needed to score owned-item relevance honestly; no zero/random score is substituted. `base_dir` resolves relative images for in-memory inputs. `batch_size` defaults to 256. `max_combinations` defaults to 200000: excessive inputs fail explicitly rather than silently truncating combinations.

Owned relevance uses the existing M6 weighted context similarity + **maximum** social-reference similarity. This does not change the eBay Top-K selection or rank owned clothes against those Top-K products.

## Combinations and Student 5 projection

The module unions known-category owned clothes and eBay Top-K products **at outfit construction time**. TOP, BOTTOM and FOOTWEAR are mandatory; OUTERWEAR is optional. At most one item fills each role. UNKNOWN/unsupported categories remain in the catalog with explicit exclusion reasons; they are not guessed. All legal combinations are evaluated in batches, sorted by class-1 probability. Outfit IDs hash the role/product-ID mapping and do not change merely because ranks or input order change.

Successful `ModuleResult.data` contains:

- `status`, `source`, `model_version`, `model_metadata`.
- `all_scored_combinations`: **every** scored outfit with its stable ID, role-to-product-ID mapping and compatibility score.
- `product_catalog`: complete original product/wardrobe fields and vectors, plus relevance scores. This avoids duplicating 512-dimensional vectors in every outfit.
- `outfits`: all combinations whose items can be projected safely to the current Student 5 Product contract, each with `outfit_id`, `compatibility_score`, `items`.
- `excluded_items`, `projection_issues`, `summary`, and explicit projection policies.

Pass `data` to the existing Student 5 adapter/service:

```python
from travel_agent.outfits.service import run as build_outfits
from travel_agent.recommendation.service import run as recommend

fourth = build_outfits(payload)
if fourth.status == 'ok':
    fifth = recommend({
        'action': 'recommend',
        'request': student5_request,
        'student4': fourth.data,
    })
```

Important semantics:

- Original prices, currencies, UNKNOWN stock and evidence remain in `product_catalog`.
- Purchase items retain native currency. No FX conversion occurs. Price/stock must be verified to map to Student 5's typed fields. Unverified purchases are still scored in `all_scored_combinations`; affected combinations are omitted only from `outfits`, with explicit projection issues. They are never silently declared in stock/out of stock.
- An owned item's projected `price=0` means **zero additional purchase cost**, not an inferred historical price. Its original `price=null` is preserved. The projected stock boolean is an ignored false placeholder: Student 5 exempts `is_user_owned=true` from merchant-stock checks. Raw wardrobe availability remains UNKNOWN. Zero additional purchase costs are expressed in USD. Original purchase prices stay in USD; the original user budget retains its currency. Student 5 owns currency conversion.
- An owned item's zero cost is excluded by Student 5's purchase-total calculation. No purchase link is invented.
- Size reliability is carried only from verified size evidence; no size is inferred from profile measurements.
- Missing required roles produce `no_candidates`. Invalid/missing inputs or model prerequisites produce `needs_input`. No model-compatibility or projection check is bypassed.
- The complete artifact can be large; use backend/file handoffs rather than inserting all vectors/combinations into an LLM conversation. Image paths in the typed Student 5 projection are resolved local paths; run the adapter on the recipient's copy of the Student 3 bundle before using it on another computer.

## Run existing handoff through Student 4

Replace the timestamp with a corresponding run's paths:

```bash
.venv/bin/python -m travel_agent.outfits \
  --products runs/product_retrieval/<timestamp>/result.json \
  --ranking-report runs/product_retrieval_diagnostics/<timestamp>/run_report.json \
  --understanding runs/live_pipeline/<timestamp>/understanding.json \
  --output runs/outfit_generation/<timestamp>/result.json
```

Add `--verify-recommendation` to save an additional `_student5.json` result using the **unchanged** original request. A non-ready Student 5 result returns exit code 4 and is preserved for inspection. Different purchase/budget currencies are an explicit CurrencyMismatchError, not a successful budget check.

## Fresh real Student 1 -> 4 run

Use an explicitly reviewed message/location payload as documented in the local UI/product runbooks:

```bash
.venv/bin/python -m scripts.run_live_product_pipeline \
  --input /path/to/reviewed_input.json --with-outfits
```

This runs real local Qwen (for message inputs), live weather, live Unsplash, live eBay Production, writes the timestamped portable Student 3 handoff, then dynamically calls Student 4 with **that same run's** file and scoring signals. Student 4 writes `runs/outfit_generation/<timestamp>/result.json` and `student5_input.json` (the original request plus all typed outfits); the live pipeline record includes its status, path and summary. Omitting `--with-outfits` preserves the existing Student 1–3 runner/UI behavior. No deletion worker or AI preview is invoked.

## Local verification

Fresh real run `20261021_213305_425160+0800` completed through Student 4. It indexed 80 live eBay candidates and returned 20 Top-K items plus 21 owned items. Thirteen owned items have supported categories; eight UNKNOWN items were retained outside outfit-role assignment. Category pool sizes were OUTERWEAR=6, TOP=7, BOTTOM=12, FOOTWEAR=8, yielding **4704 scored and Student5-ready combinations**, including optional-coat combinations. This is complete enumeration, not Top-3 filtering.

The original 500 SGD request correctly fails Student 5's currency check against USD purchases. A separate, explicitly labeled **500 USD controlled validation fixture**, without changing the original request or any prices, demonstrated successful constraint filtering and three final recommendations. It is not evidence that the original SGD budget passes. New regression tests cover dynamic inputs, stable IDs, zero padding, owned participation, UNKNOWN handling, exact social-MAX relevance, no silent truncation and the Student 5 handoff.

Production-derived outfit artifacts are registered with the existing local lifecycle registry when saved. This does not run a deletion worker or clear any queue. Model artifacts, credentials, runtime inputs and generated results are not committed to Git.
