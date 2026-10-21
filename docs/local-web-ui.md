# Local travel product interface

The AWAY interface is a local, additive UI over the existing Understanding, Inspiration and Product Retrieval modules. It uses a quiet editorial layout inspired by the supplied runway reference; it does not claim affiliation with the reference brand.

## Run

From the repository root, using the configured environment:

```bash
.venv/bin/python -m travel_agent.web --port 8765
```

Open `http://127.0.0.1:8765/`. Keep the terminal running. The service binds to loopback only; this is a private single-user prototype, not a hosted/multi-user service. Do not expose it through a public tunnel.

Prerequisites are the same as the verified CLI pipeline: Python 3.11; products/inspiration/Qwen/notification extras; cached pinned FashionCLIP and Qwen3-0.6B weights; local `.secrets/unsplash.env`; Production `.secrets/ebay-production.json`; and `scripts/run_live_product_pipeline.py`. No new framework or Node dependency is required. The cream/beige illustrated companion is a static transparent asset animated with CSS; reduced-motion preferences disable its animation. Secrets never enter browser JavaScript.

## Interaction

1. Tell the travel companion about a trip in the English chat interface. Cities and countries are open inputs, such as Tokyo, Los Angeles, Paris or Seoul; no city is preselected.
2. Local Qwen extracts a provisional request and remembers it across turns. The companion asks for missing dates, activities, budget/currency, optional style and required occasion dress codes. Dates must fit the next 15 days for live weather. No missing dress code is guessed.
3. Review the parsed trip and choose the intended geocoding result. Go back to chat to correct any details. Only confirmation starts downstream API retrieval.
4. Confirm to start live Open-Meteo weather, Unsplash retrieval/FashionCLIP ranking, then eBay Production hybrid acquisition and final product ranking. The UI shows stage progress without raw JSON, PT files or logs. It does not invoke a cloud Qwen API.
5. Browse category tabs, inspect real product details, save favorites in this browser, or follow the eBay listing link. Prices retain their original currency; total budget is not silently converted to a per-item ceiling. Stock labels mean API snapshot evidence, not checkout guarantees.

The page initially shows the latest saved results, explicitly labeled as historical. These are not a new search and are not necessarily for the country/city being discussed in chat. New results replace the displayed edit only after a real run completes. Changing inspiration slides does not claim to regenerate or pair outfits; outfit generation remains downstream work.

## Artifacts for downstream modules

- `runs/web_jobs/<job-id>/conversation.json`: conversation messages and provisional request.
- `turns/<number>.json`: per-turn Qwen extraction and source message audit.
- `form.json`, `message_input.json`, `extraction.json`: retained for the legacy form API.
- `review.json`: parsed trip and candidate locations.
- `confirmed_input.json`: request/coordinates explicitly approved in the review dialog.
- `prepare.log`, `run.log`, `status.json`: private diagnostics and UI job state.
- `runs/live_pipeline/<timestamp>/`: Understanding output, newly generated Inspiration PT/TXT/context/model metadata, evaluation data and pipeline record.
- `runs/product_retrieval_diagnostics/<timestamp>/`: internal acquisition, photos and complete report.
- `runs/product_retrieval/<timestamp>/`: portable downstream folder with result JSON, relative product/wardrobe images and README. This is the Student 4/5 handoff; existing interfaces remain unchanged.

The page serves curated display fields and allowlisted local images, not these raw artifacts. It serializes model/API work to avoid concurrent model loads. Keep the service running during a job; this prototype does not provide a durable task scheduler or recovery of interrupted subprocesses. The existing draft KG remains explicitly enabled and is disclosed at confirmation.

## Lifecycle

The UI does not start the deletion worker or purge a notification queue. Existing registered Production artifacts can still be invalidated by a separately operated cleanup worker. Generating new results does not establish that notification obligations are current. Copied recipient outputs require coordinated cleanup. Input/diagnostic retention remains a separate local operational responsibility.

## Verification

The browser-triggered Seoul trip ran through Qwen review, live weather, live Unsplash and eBay Production successfully, returning four categories with five products each and one owned item. Tokyo and Los Angeles receive separate geocoding/requirements review; that is distinct from asserting that every possible city/activity combination has been evaluated. HTTP boundary tests reject foreign origins/hosts, private file paths and invalid dates; display projection excludes raw vectors and provenance paths. Existing complete pipeline tests remain applicable.


## Clarification is not an API failure

Bare replies to explicit budget/style/dress-code questions are bound to that question before extraction; raw and contextualized messages are retained separately. Unknown or ambiguous answers trigger clarification. The confirmation dialog derives required occasion dress codes from the existing Activity KG. Dinner and business meetings require separate explicit choices; supported choices differ (for example, business casual/business formal for a business meeting). Missing or unsupported choices are rejected before any social/product search. No code is guessed on behalf of the user. The command-line runner preserves `needs_input` and questions with exit code 4, and the web layer returns to review instead of reporting generic API configuration errors. Operational failures are classified by understanding, weather, inspiration or product stage.

The Tokyo 2026-10-23 through 2026-11-04 failure was reproduced: Open-Meteo successfully returned forecast data, but Understanding requested dinner and business-meeting dress codes; neither Unsplash nor eBay was reached.

## Conversational UI verification

The Tokyo 2026-10-23 to 2026-10-25 browser test collected the missing 500 SGD budget and dinner smart-casual dress code in separate chat turns, then completed live weather, Unsplash and eBay Production retrieval. The portable result is `runs/product_retrieval/20261021_171104_502784+0800/`: five products in each of four categories and one owned item. Conversation/extraction audit is retained under the associated web job. This verifies connectivity and handoff generation; it is not an evaluation of recommendation relevance or category accuracy. Desktop (1440px) and mobile (390px) layouts were checked, including avatar loading and horizontal overflow.

## Local Profile and wardrobe

Open `/profile` from the top navigation. The English page supports optional name, age, height in cm and weight in kg, saved under `runs/web_profile/profile.json`. This is a single-device profile, not a multi-user login/account system. Personal measurements are not forwarded to Qwen, social APIs or eBay and do not imply a size recommendation.

Choose or drop up to five JPG/PNG/WebP photos per batch, each up to 10 MB and 20 megapixels. The server decodes images, normalizes orientation and removes metadata before invoking the existing offline Phase 2 FashionCLIP ingestion service. Processing is serialized with chat and retrieval tasks. Items persist in `runs/wardrobe/items.json` with managed photos in `runs/wardrobe/images/`; these are the same paths already read by the live product pipeline. Future retrieval handoffs include the current owned pool; existing saved handoffs remain historical. Duplicate image pixels reuse the same item ID. Uncertain category predictions appear as Unsorted; unknown attributes remain unknown.

The user-provided `wardrobe.zip` baseline imported 20 photos without issues, preserving the original owned item (21 total). The import manifest is `runs/web_profile/baseline_import.json`; normalized input photos are under `data/raw/wardrobe/baseline_20261021/`. Display names are saved separately in `runs/web_profile/wardrobe_labels.json`. All profile, image, vector and runtime data are Git-ignored.

After restart, baseline records and profile values were preserved. Browser category filtering/details and profile persistence were verified. A real duplicate-photo POST completed through FashionCLIP with `added=0`, preserving 21 items. Full regression: 1457 passed, 21 skipped. The browser file chooser and selected-photo preview were verified; upload decoding, dispatch and the real FashionCLIP processing endpoint were also verified.

## Baseline branch

`feat/travel-ui-wardrobe-baseline` starts directly from `0f054a4a99a4ea46080aceb5ef00a4cc6f5278e1`. It adds the local Student 1–3 runner, conversational website, Profile/wardrobe page, avatar, tests and documentation. Later commits from `main` are deliberately not merged. The `outfit-compatibility/` tree is identical to that baseline; this branch does not attempt to repair or validate later teammates' changes.

For a fresh clone, follow the product runbook to install the Python 3.11 extras and download the pinned models, then supply local Unsplash/eBay credentials. Launch `.venv/bin/python -m travel_agent.web --port 8765`, review a trip in chat and confirm it to run the real pipeline. Runtime photos, profile data, API credentials, model weights and generated results are not in Git. A fresh clone starts with an empty profile/wardrobe until the user imports photos. The existing CLI wardrobe import remains available.
