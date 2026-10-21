# Local travel product interface

The WearTo interface is a local, additive UI over the existing Understanding, Inspiration and Product Retrieval modules. It uses a quiet editorial layout inspired by the supplied runway reference; it does not claim affiliation with the reference brand.

## Run

From the repository root, using the configured environment:

```bash
.venv/bin/python -m travel_agent.web --port 8765
```

Open `http://127.0.0.1:8765/`. Keep the terminal running. The service binds to loopback only; this is a private single-user prototype, not a hosted/multi-user service. Do not expose it through a public tunnel.

Chat and optional KG discovery need Python 3.11 and locally cached Qwen3-0.6B weights. The worker uses `.cache/huggingface` when available, falling back to `artifacts/huggingface/hub`. The complete confirmed retrieval pipeline additionally needs the same prerequisites as the verified CLI pipeline: Python 3.11; products/inspiration/Qwen/notification extras; cached pinned FashionCLIP and Qwen3-0.6B weights; local `.secrets/unsplash.env`; Production `.secrets/ebay-production.json`; and `scripts/run_live_product_pipeline.py`. No new framework or Node dependency is required. The cream/beige illustrated companion is a static transparent asset animated with CSS; reduced-motion preferences disable its animation. Secrets never enter browser JavaScript.

## Interaction

1. Tell the travel companion about a trip in the English chat interface. Cities and countries are open inputs, such as Tokyo, Los Angeles, Paris or Seoul; no city is preselected.
2. Local Qwen extracts provisional facts and generates natural replies through the shared understanding conversation controller. Destination, start/end dates and budget/currency are required and can be supplied separately. Missing activities and styles trigger KG-backed multi-select cards; choose any, skip the current step, or skip both. No candidate becomes a user fact before selection. Occasion dress codes remain optional; missing values are never guessed. Future trips outside forecast coverage use historical climate references instead of requiring different dates.
3. Review the parsed trip and choose the intended geocoding result. Go back to chat to correct any details. Only confirmation starts downstream API retrieval.
4. Confirm to start weather planning (live forecasts within coverage, labelled historical references outside), Unsplash retrieval/FashionCLIP ranking, then eBay Production hybrid acquisition and final product ranking. The UI shows stage progress without raw JSON, PT files or logs. It does not invoke a cloud Qwen API.
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

Bare replies to explicit budget/style/dress-code questions are bound to that question before extraction; raw and contextualized messages are retained separately. Unknown or ambiguous answers trigger clarification. The confirmation dialog lists optional occasion dress codes from the existing Activity KG. Dinner and business meetings have separate choices; supported choices differ (for example, business casual/business formal for a business meeting). Blank choices stay unspecified and do not block search; an explicitly supplied unsupported choice is rejected. Selecting the blank option clears a prior code. No code is guessed on behalf of the user. The command-line runner preserves `needs_input` and questions with exit code 4, and the web layer returns to review instead of reporting generic API configuration errors. Operational failures are classified by understanding, weather, inspiration or product stage.

The Tokyo 2026-10-23 through 2026-11-04 failure was reproduced: Open-Meteo successfully returned forecast data, but Understanding requested dinner and business-meeting dress codes; neither Unsplash nor eBay was reached.

## Conversational UI verification

The Tokyo 2026-10-23 to 2026-10-25 browser test collected the missing 500 SGD budget and dinner smart-casual dress code in separate chat turns, then completed live weather, Unsplash and eBay Production retrieval. The portable result is `runs/product_retrieval/20261021_171104_502784+0800/`: five products in each of four categories and one owned item. Conversation/extraction audit is retained under the associated web job. This verifies connectivity and handoff generation; it is not an evaluation of recommendation relevance or category accuracy. Desktop (1440px) and mobile (390px) layouts were checked, including avatar loading and horizontal overflow.

## Local Profile and wardrobe

Open `/profile` from the top navigation. The English page supports optional name, age, height in cm and weight in kg, saved under `runs/web_profile/profile.json`. This is a single-device profile, not a multi-user login/account system. Personal measurements are not forwarded to Qwen, social APIs or eBay and do not imply a size recommendation.

Choose or drop up to five JPG/PNG/WebP photos per batch, each up to 10 MB and 20 megapixels. The server decodes images, normalizes orientation and removes metadata before invoking the existing offline Phase 2 FashionCLIP ingestion service. Processing is serialized with chat and retrieval tasks. Items persist in `runs/wardrobe/items.json` with managed photos in `runs/wardrobe/images/`; these are the same paths already read by the live product pipeline. Future retrieval handoffs include the current owned pool; existing saved handoffs remain historical. Duplicate image pixels reuse the same item ID. Uncertain category predictions appear as Unsorted; unknown attributes remain unknown.

The user-provided `wardrobe.zip` baseline imported 20 photos without issues, preserving the original owned item (21 total). The import manifest is `runs/web_profile/baseline_import.json`; normalized input photos are under `data/raw/wardrobe/baseline_20261021/`. Display names are saved separately in `runs/web_profile/wardrobe_labels.json`. All profile, image, vector and runtime data are Git-ignored.

After restart, baseline records and profile values were preserved. Browser category filtering/details and profile persistence were verified. A real duplicate-photo POST completed through FashionCLIP with `added=0`, preserving 21 items. Full regression: 1457 passed, 21 skipped. The complete browser upload flow (file chooser, preview, submit, real FashionCLIP processing and duplicate confirmation) was verified; the owned pool remained at 21 items.

## Baseline branch

`feat/travel-ui-wardrobe-baseline` starts directly from `0f054a4a99a4ea46080aceb5ef00a4cc6f5278e1`. It adds the local Student 1–3 runner, conversational website, Profile/wardrobe page, avatar, tests and documentation. Later commits from `main` are deliberately not merged. The `outfit-compatibility/` tree is identical to that baseline; this branch does not attempt to repair or validate later teammates' changes.

For a fresh clone, follow the product runbook to install the Python 3.11 extras and download the pinned models, then supply local Unsplash/eBay credentials. Launch `.venv/bin/python -m travel_agent.web --port 8765`, review a trip in chat and confirm it to run the real pipeline. Runtime photos, profile data, API credentials, model weights and generated results are not in Git. A fresh clone starts with an empty profile/wardrobe until the user imports photos. The existing CLI wardrobe import remains available.

### Verification on the baseline branch (2026-10-21)

A fresh run used the previously human-reviewed Seoul 2026-10-22 to 2026-10-25 message: local pinned Qwen extraction, real Open-Meteo forecast, two live Unsplash searches and live eBay Production hybrid acquisition. It produced 10 fresh social references, processed 80 of 349 API candidates and returned five products in each of four categories. The portable output `runs/product_retrieval/20261021_191904_054545+0800/` contains 20 product photos and 21 owned-wardrobe photos, all referenced with existing relative paths; every item has a 512-dimensional embedding. The upstream PT/TXT/context/model-info files were regenerated for this run. Diagnostics remain local. Normal regression: 1457 passed, 21 skipped; relevant source lint and diff whitespace checks passed. This is a connectivity/integration check, not a ranking-quality evaluation.

## Inspiration slideshow

Social reference photos advance every four seconds after the current image loads. Previous/next arrows remain available and restart the dwell timer. The pause/play control lets users stop or resume autoplay. Background tabs pause the timer; returning starts a fresh interval. Reduced-motion preferences start the slideshow paused. This only changes the displayed inspiration photo, not product retrieval or ranking.

The frontend brand is **WearTo**, with the signature **Where to? WearTo!** across the home/profile headers, page titles and footers.

## Shared understanding dialogue integration

The WearTo page now uses `understanding.conversation.Conversation` for multi-turn state, missing-field prompts, activity/style discovery, explicit choices and skip decisions. `conversation.json` embeds its `dialogue` checkpoint; a new worker process restores that checkpoint on each turn. Original input, contextualized input, raw extraction/reply metadata and full KG evidence stay in private turn logs. The browser receives only public labels and a menu key; stale button selections are rejected. Existing UI drafts migrate their provisional request without losing fields.

The activity/style cards support multiple selections, `Use selected`, `Skip this step` and `Skip both`. Typed numbers/names and `skip`/`skip all` work too. Required facts and optional selections remain provisional until the review dialog is confirmed. Confirmation commits the reviewed request, then starts the original downstream pipeline. Empty activity/style arrays and unspecified dress codes are valid. No social/product search runs while merely collecting facts or selecting options. Country-only inputs request a city; geocoding failures retain all trip facts.

Local verification covers real Qwen chat and the browser controls up to the final review. Tests use injected model/geocoding boundaries for offline regressions. A complete Unsplash/eBay run still requires the user's local API configuration; chat success does not establish those services are configured or product results are current.

## Quiet choice bubbles

Activity and style options appear as gently floating bubbles. Selecting a bubble preserves its DOM node while animating its color, outline, shape and checkmark. The motion pauses on hover, focus and selection; reduced-motion preferences disable it. Continue submits the selected labels; Skip and Skip all remain available. Model prose is not generated for option menus or review summaries because the bubbles and existing review dialog own those surfaces. Original extraction/state evidence remains in private logs; legacy duplicate narration is filtered only from the display projection.

Operational failures use a dismissible notice rather than a companion message. Search begins in setup, then advances to weather only when Understanding actually starts. Missing Unsplash configuration is labelled setup; it does not imply the weather API failed. The Tokyo weather-only check returned a real forecast successfully.

## Flexible chat dates

The conversation controller normalizes dates before passing the message to the
unchanged Qwen model. Dates can appear alongside destination, budget, or other
conditions, or as a short reply to a departure/return question. Supported forms
include `2026/10/28`, `2026.10.28`, `28 Oct 2026`, `Oct. 28, 2026`, and
`2026年10月28日`. Two dates in source order form a range unless explicit
start/return labels specify their roles. Numeric day/month/year forms are accepted
only when the order is unambiguous; `03/04/2026` asks which interpretation is meant.

`today`, `tomorrow`, `day after tomorrow`, `今天`, `明天`, and `后天` use the
Singapore calendar date when the message is processed. `next Friday` / `下周五`
suggest a concrete date in the following calendar week and request explicit date
confirmation. Durations such as `for five days` ask for a return date to avoid
confusing days with nights. Month/day expressions without years ask for a year;
for `Oct 28 to Nov 2` or `10月28日到11月2日`, a year-only follow-up completes the
saved dates. Cross-year ranges need explicit years for both endpoints. Invalid
calendar dates, reverse ranges, and more than two date mentions require clarification.
Other successfully extracted facts stay provisional while dates are clarified.
The review modal remains the final confirmation of all parsed trip details.

These are bounded parsing rules, not a guarantee for arbitrary date phrasing.
Original messages and normalized extraction input remain available in private turn
audit files. The model ID, weights, and generation settings are unchanged.
