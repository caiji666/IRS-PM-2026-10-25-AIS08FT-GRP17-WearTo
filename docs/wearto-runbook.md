# WearTo: complete local website runbook

This is the primary guide for the current **Student 1–5 website**, including local
Qwen dialogue, weather context, Unsplash references, eBay Production products,
wardrobe-aware outfit scoring, final Top-3 recommendations and selected-outfit AI
preview. Run commands from the repository root. The UI, API server and model jobs
are Python-based; the browser uses HTML/CSS/JavaScript. No npm build or separate
frontend server is required.

## A. Start an already configured checkout

On the original Mac, where the environment, models and credentials are prepared:

```bash
cd /Users/yoyozhang/Desktop/travel-style-agent
.venv/bin/python -m travel_agent.web --port 8765
```

Open **http://127.0.0.1:8765/**. Keep the terminal running; use Ctrl+C to stop.
If port 8765 is already occupied, stop the previous instance before restarting.
Reload the browser once after a code update. Restarting the server clears in-memory
recommendation sessions, but keeps stored wardrobe items and result files.

The steps below describe first-time setup for a different checkout or computer.
They do not require the original `IntelligentTravelStyleAssistant` directory.

## B. Clone and install all website dependencies

Install Git and uv, then:

```bash
git clone https://github.com/caiji666/travel-style-agent.git
cd travel-style-agent
uv python install 3.11
uv sync --locked \
  --extra products --extra inspiration --extra qwen --extra outfits \
  --extra ebay-notifications --extra preview
```

Keep these extras installed together. A later plain `uv sync` can remove optional
packages from this environment. The following `uv run --no-sync` commands use the
prepared environment without pruning it. Allow several GB of local disk space
for downloaded models and generated data. A private repository requires GitHub access.

## C. Download the models and prepare the outfit scorer

The website's workers run models offline, so complete both downloads first:

```bash
uv run --no-sync python -m travel_agent.products.retrieval.m5.download \
  --model-cache .cache/huggingface

uv run --no-sync python - <<'PY'
from huggingface_hub import snapshot_download
from travel_agent.understanding.extraction import MODEL_ID, MODEL_REVISION
snapshot_download(
    repo_id=MODEL_ID,
    revision=MODEL_REVISION,
    cache_dir=".cache/huggingface",
)
PY

uv run --no-sync python -m travel_agent.outfits.train --reencode
```

The downloads require network access. Qwen uses local inference and needs no
DashScope or OpenAI text-model API key. The training command re-encodes the supplied
Polyvore archive and creates `artifacts/outfits/normalized-v1/model.joblib`, feature
cache and manifests. It can take several minutes or longer on CPU. The source CSV,
category map and image archive are in `outfit-compatibility/`; the old historical
pickle is not a substitute for the aligned model. Repeat training only when needed;
normal website startup reuses the generated model. See [model details](outfit-integration.md).

## D. Configure private API credentials

Create the private configuration directory:

```bash
mkdir -p .secrets
chmod 700 .secrets
```

Create `.secrets/unsplash.env` in a local editor:

```dotenv
UNSPLASH_ACCESS_KEY=YOUR_UNSPLASH_ACCESS_KEY
```

Create `.secrets/ebay-production.json`:

```json
{
  "environment": "production",
  "client_id": "YOUR_PRODUCTION_APP_ID",
  "client_secret": "YOUR_PRODUCTION_CERT_ID"
}
```

`client_id` is eBay App ID / Client ID; `client_secret` is Cert ID / Client Secret,
not Dev ID. The live website requires an enabled **Production** keyset; a Sandbox
key cannot substitute for it. See [Production activation and lifecycle service](ebay-production-activation.md)
if activation is still required. The notification receiver is deployed separately;
the local website does not automatically operate its cleanup worker.

For real AI preview, create `.env` from `.env.example` if it does not exist, then
fill these entries in a local editor:

```dotenv
PREVIEW_PROVIDER=openai
PREVIEW_API_KEY=YOUR_OPENAI_IMAGE_API_KEY
PREVIEW_MODEL=gpt-image-2.5-sunburst
PREVIEW_ALLOWED_IMAGE_HOSTS=i.ebayimg.com,images.unsplash.com
```

Use a model enabled for your OpenAI project. Image generation is a paid external
operation, run only after confirming one outfit. Without a usable provider the
system may return a product-card fallback or a preview error; this is not a real
AI-generated image. Do not put credentials in JavaScript, commits, screenshots or
shared result packages. `.env` and `.secrets/` are Git-ignored. Restrict access:

```bash
chmod 600 .secrets/unsplash.env .secrets/ebay-production.json .env
uv run --no-sync python -m travel_agent.products.retrieval.ebay \
  --credentials .secrets/ebay-production.json auth-check
```

The separate Unsplash file is required by the live runner even if the same key is
also present in `.env`. The web server loads `.env` at startup; restart it after
changing preview settings. Weather uses Open-Meteo without an API key. Student 5
also needs access to the ECB rate feed, or an acceptable existing cache, for currency
normalisation. Source prices remain unchanged in Student 3/4; Student 5 performs FX.

## E. Optionally import the shared wardrobe

To reproduce the shared 21-photo baseline:

```bash
uv run --no-sync python -m travel_agent.products.retrieval.phase2 --offline \
  ingest --images data/examples/wardrobe/images/*.png \
  --output runs/wardrobe/baseline_import.json
```

This creates the local wardrobe store and FashionCLIP embeddings under
`runs/wardrobe/`. Alternatively upload your own single-garment images on `/profile`.
Wardrobe import is optional; an empty wardrobe still permits product-only outfits.
The approved shared baseline is tracked; future private photos and profiles are not.

## F. Launch the website and test the complete flow

```bash
uv run --no-sync python -m travel_agent.web --port 8765
```

Open http://127.0.0.1:8765/ and:

1. Click **New journey**. Enter a city/country, travel dates, budget and currency.
   You can provide these in separate messages. Use current/future dates, for example
   `2026-10-25` then `2026-10-28` when testing on 22 October 2026.
2. Supply activities/style, select optional bubbles or choose **Skip / Skip all**.
3. Review the extracted facts, geocoding location and optional dress code. Correct
   any mistakes before **Confirm my trip & find pieces**. No social/product search
   runs merely because a fact or optional choice was entered.
4. The backend runs weather/context reasoning → fresh Unsplash/FashionCLIP references
   → eBay image and keyword retrieval → category Top-K → product/wardrobe outfit
   combinations and compatibility scores → Student 5 currency/constraint filtering
   and final ranking. The browser displays up to three outfits with images.
5. Inspiration pictures and the destination heading update with the corresponding
   result. The slideshow supports autoplay, pause and manual arrows.
6. Select an outfit tab and click **Confirm this outfit**, then **Generate AI preview**.
   Without a portrait, a successful provider returns a flat-lay composition. A managed
   portrait can be uploaded on Profile for the portrait workflow; this is not a sizing
   or exact-fit guarantee. The recorded live acceptance test used flat-lay mode.
7. Use **Adjust requirements** to refine an existing recommendation. Budget changes
   can reuse candidates; destination/date/style changes require confirmation and
   rerunning affected stages. Use **New journey** for an independent trip.

A validated run can still have fewer than three feasible outfits or no candidates.
The system must report shortages/conflicts rather than invent stock or relax a
budget. Forecast-range trips use forecasts; later trips use labelled historical
climate data when covered. Unsupported or ambiguous inputs may need clarification.
The current implementation permits outfits with or without outerwear; it does not
guarantee mandatory weather-based outerwear enforcement. Reliable size evidence is
not available for every listing. Check the seller for final stock, fit and delivery.

## G. Where files are saved

Each complete search uses a new timestamp, such as `20261022_123414_217382+0800`:

| Location | Contents |
|---|---|
| `runs/web_jobs/<job-id>/` | Dialogue state, per-turn audit, reviewed request, job status and logs. |
| `runs/live_pipeline/<timestamp>/` | Student 1 understanding, fresh Student 2 PT/TXT/JSON/model information and pipeline record. |
| `runs/product_retrieval_diagnostics/<timestamp>/` | Raw acquisition details, query/scoring diagnostics and candidate/reference images. |
| `runs/product_retrieval/<timestamp>/` | Student 3 portable product/wardrobe handoff, relative image paths, embeddings and scores. |
| `runs/outfit_generation/<timestamp>/result.json` | Student 4 combinations, compatibility scores, original item catalogue and issues. |
| `runs/outfit_generation/<timestamp>/student5_input.json` | Original request and typed dynamic combinations for Student 5. |
| `runs/recommendation/<timestamp>/result.json` | Initial web recommendation ModuleResult, final outfits, filtering summary and state. |
| `artifacts/previews/*.png` | Real generated preview images. Web copies are in `runs/web_jobs/student5-preview-media/`. |
| `runs/wardrobe/` | Persisted owned items, predictions, embeddings and managed images. |
| `runs/web_profile/` | Local profile and wardrobe display labels. |

The browser displays selected fields, not intermediate JSON/PT files. Runtime files
are local and not included in a clone. Recommendation/refinement sessions are held
in memory; not every later interaction is automatically exported as a separate JSON
checkpoint. Student 5 JSON can include machine-local image references: transferring
it alone does not transfer those images. Use the portable product bundle and adapt
paths on the recipient machine. Coordinate registered-data cleanup for any copies.

## H. Backend-only execution and checks

For an already human-reviewed input JSON, run Student 1–4 without opening the UI:

```bash
uv run --no-sync python -m scripts.run_live_product_pipeline \
  --input /path/to/reviewed_input.json --with-outfits
```

For a **freshly cloned environment**, the UI generates new upstream handoffs; you
do not need the original author's `data/team_input/` PT files to start a new trip.
The CLI command above stops after writing Student 4 and Student 5 input artifacts.
To run Student 5 directly, using that timestamp:

```bash
uv run --no-sync python -m travel_agent.outfits \
  --products runs/product_retrieval/<timestamp>/result.json \
  --ranking-report runs/product_retrieval_diagnostics/<timestamp>/run_report.json \
  --understanding runs/live_pipeline/<timestamp>/understanding.json \
  --output runs/outfit_generation/<timestamp>/result.json \
  --verify-recommendation
```

Replace each `<timestamp>` before running. This reuses the same retrieval handoff,
reruns compatibility scoring and saves `result_student5.json` beside the outfit
result. It uses the actual original request, not a temporary test budget. For
confirmation and preview, prefer the website's current-session buttons rather
than the historical fixed-input test scripts.

```bash
uv lock --check
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q
```

Node.js is optional for exercising the browser-handler regression tests; it is not
required to serve the website. Real-model/API tests need their documented opt-in
settings. Passing offline tests does not prove external services are configured.

## I. Common problems and verified scope

- **Missing Unsplash configuration:** verify `.secrets/unsplash.env`, not only `.env`.
- **eBay access error:** verify Production activation, App/Cert ID mapping and environment.
- **Model unavailable offline:** complete the Qwen/FashionCLIP downloads before starting.
- **Outfit model missing/incompatible:** run aligned training; do not relabel old vectors.
- **Saved journey has no outfit output:** old product-only runs cannot supply Student 5;
  confirm a new trip with the current runner.
- **Stale chat/session after a restart:** reload and use New journey. Saved results remain.
- **Preview unavailable:** confirm a current Top-3 first; check private provider settings,
  image access, model permission, balance and installed preview dependencies.

On 22 October 2026, a complete browser dialogue/review led to real weather, Unsplash,
eBay Production and three Student 5 recommendations. A separate same-day test
produced one real OpenAI flat-lay image. Local regression at `68aa3fd` reported
857 passed and 7 skipped. These validate application integration, not retrieval
accuracy, fit, universal city coverage or autonomous LLM-based orchestration of all
tools. The basic offline demo and five-module harness have separate scopes.
