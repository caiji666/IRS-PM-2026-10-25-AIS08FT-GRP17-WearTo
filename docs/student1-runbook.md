# Student 1 — today's supervised flow

Reference: proposal Section 7.1 Stages 1–3 and Appendix A. This delivery implements the understanding boundary; FashionCLIP, retrieval and outfit recommendation remain owned by other modules. [Contracts](contracts.md) explains the proposal/scaffold differences.

## 1. Reproduce the complete offline context flow

From the cloned `travel-style-agent` repository:

```bash
uv sync --locked
uv run python -m travel_agent.understanding.demo
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

This uses `examples/understanding/seoul_winter.json`: proposal-shaped request → validation → explicitly synthetic winter weather → deterministic clothing rules → draft Activity KG → ExpandedFashionContext text. Dinner's smart casual code is supplied in the request. It is not a December forecast or a demonstration of LLM accuracy. The serialized output is in `examples/understanding/seoul_winter_result.json`. `qwen_seoul_result.json` also records actual local-model output verified against the fixed English developer input and then passed through that same synthetic-weather/KG flow.

To save session state locally:

```bash
uv run python -m travel_agent.understanding.demo --save-state artifacts/session.json
```

To revise only the budget and exclude long coats, use `examples/understanding/budget_patch.json`:

```bash
uv run python -m travel_agent.understanding.demo \
  --input examples/understanding/budget_patch.json \
  --state artifacts/session.json --save-state artifacts/session.json
```

The unchanged destination, dates, activities and style remain in state. Without an explicitly supplied weather provider/fixture, the next turn's weather is unknown. Prior weather is not silently reused for a changed destination/date. The caller can provide the same scoped fixture or live provider again.

The proposal-style session is serialized to a caller-owned file; a shared production checkpointer is not configured. Do not share one state file between users. The recommendation owner must populate the latest displayed IDs and set AWAITING_CONFIRMATION before outfit confirmation can work.

## 2. Run actual local Qwen3-0.6B extraction

```bash
uv sync --locked --extra qwen
HF_HOME="$PWD/artifacts/huggingface" uv run --extra qwen \
  python -m travel_agent.understanding.demo \
  --message 'I am going to Seoul from 2026-12-20 to 2026-12-27 for sightseeing and dinner. My shopping budget is 500 SGD. I prefer minimalist outfits. The dinner dress code is smart casual.' \
  --output artifacts/extraction-preview.json
```

The fixed model is `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`. First use downloads weights into ignored `artifacts/`. No API key or paid model service is needed. This Mac uses MPS when available; CPU/CUDA are alternatives. No model or provider is loaded at module import time. Advanced optional POI/owned-item extraction is not yet evaluated.

**Human checkpoint:** inspect `proposed_request` against your exact original message. The model can produce valid JSON with wrong facts or omit facts. The default preview stores accumulated facts in `state.pending_request` while keeping accepted facts in `state.current_request` unchanged. Save and resend returned state on every turn, including previews; later messages need only the new or corrected details. Correct the structured request or patch and submit it through `--input`. For a previously reviewed exact input, the direct CLI's `--accept-extraction` flag permits application; this is explicit caller approval, not a model's approval. The LangChain tool forcibly disables that flag for messages so an agent cannot approve its own extraction.

Budget/currency evidence guards, explicit ISO/Chinese date checks, qualified dress-code checks, and limited activity/style vocabulary normalization reduce some hallucinations. They are conservative: unsupported date wording, synonyms and cross-sentence dress-code evidence may be omitted. They do not prove semantic correctness or replace review. A malformed patch returns a clarification/error with prior state preserved; do not accept it just to keep a demo moving.

`evaluation/understanding_smoke.py` contains seven hand-written developer cases. Results, raw outputs, postprocessing warnings and failures are saved in `examples/understanding/qwen_smoke_results.json`. These cases have been used while editing prompts, so their pass count is not an independent accuracy metric. On this delivery, 4/7 development cases passed (complete English/Chinese requests, missing-information handling and an unsupported activity); the three natural-language patch cases failed. Natural-language patch extraction remains experimental; structured patches are the reliable demonstration path today.

To rerun the live-model development check:

```bash
HF_HOME="$PWD/artifacts/huggingface" uv run --extra qwen \
  python -m evaluation.understanding_smoke \
  --output artifacts/qwen-smoke-results.json
```

## 3. Run forecasts or historical climate references

Prepare a reviewed request JSON (`{"request":{...},"allow_draft":true}`) with exact dates, then:

```bash
uv run python -m travel_agent.understanding.demo \
  --input artifacts/reviewed-trip.json --live-weather \
  --output artifacts/live-context.json
```

Open-Meteo city search can return several locations with the same name. The module returns `needs_location` and candidate coordinates instead of choosing silently. **Human checkpoint:** confirm the city/country and create a coordinate file using the returned provider values:

```json
{
  "destination": "Tokyo",
  "name": "Tokyo, Japan",
  "latitude": 35.6895,
  "longitude": 139.69171
}
```

Pass `--location artifacts/confirmed-location.json`. `destination` must match the current request; coordinates cannot silently carry over after a destination edit. Do not use city-centre weather as a mountain or indoor-venue forecast. The current live adapter supports city scope only; exact venue/mountain microclimates need separate supported input.

`--live-weather` / `weather_provider=open_meteo` selects automatic routing. Whole trips within today through day 15 use the API; trips outside or crossing that window use the local historical database for the whole trip. The public dataset covers 15 cities in 12 countries using 2015-2024 ERA5 monthly statistics. Country-only and ambiguous inputs require city confirmation; uncached confirmed cities have a bounded explicit acquisition path. See [city coverage](weather-location-coverage.md). Historical values are labelled reference data, never forecasts. API failures within the window stay unknown instead of silently falling back. Unsupported historical locations stay unavailable. No weather API key is required for this non-commercial prototype. See [weather setup, data and sharing](weather-data.md).

Provider documentation: [forecast](https://open-meteo.com/en/docs), [geocoding](https://open-meteo.com/en/docs/geocoding-api). The provider returns at most 16 days; the actual returned coverage is checked. Temperatures are Celsius, snowfall is cm and precipitation probability is converted from percent to 0–1. Source/retrieval time and dates are included in output.

## 4. Human supervision that remains necessary

| Checkpoint | What the person does | Consequence before completion |
|---|---|---|
| Qwen extraction | Compare dates, activities, amount/currency, style, exclusions and patch edits with the original text | Preview only by default; never accept an invalid patch |
| Activity KG evidence | Independently review sources, conditional edges, formality/style suggestions, then record a real reviewer | Pending knowledge requires explicit draft mode; strict mode has no invented substitutes |
| Place KG evidence | Use the existing source KG review tool, inspect venue-specific rules and all dependency records, explicitly apply decisions | Place output remains draft/conditional; it is not a universal religious/country rule |
| Weather rules | Review `weather_rules.json` thresholds, garment alternatives and personal sensitivity; approve or adjust versioned rules based on evidence | Trace stays pending; cold/hot/rain cutoffs are prototype heuristics |
| Destination ambiguity | Confirm returned city/country/coordinates; obtain appropriate scope for mountain/venue trips | No silent city choice or weather propagation |
| Team interface handoff | Agree on required fields, currency set, per-activity/POI constraints and text field with downstream owners | Schema changes are locally ready; peer review is still needed before merge |
| Model evaluation | Label a separate Chinese/English set, include negation, dates, ambiguous money and several patch rounds; have another reviewer check labels | Developer smoke results do not justify deployment or accuracy claims |
| Neo4j | Start a dedicated development DB, provide credentials locally, import the seed, compare one-hop/multi-hop evidence with the local backend | Database adapter is available but live DB connectivity is unverified |

Existing place review UI (outside the shared repository): from the original Practice Module directory, run `python3 -m student1.place_kg.review_app --open`. Decisions require human action. Do not bulk-mark records approved for a demonstration. The repository contains a snapshot; refreshing it must explicitly preserve review history and follow the copied `kg_data/place/KG_SPEC.md`. Do not rebuild seeds over reviewed source data.

Activity KG approval is recorded on individual edges in the source `student1/data/activity_kg.json`; review source URL, rationale, conditions, review status and reviewer before refreshing the repository snapshot. The copied source/CSV hashes are in `kg_data/manifest.json`. No existing knowledge was approved or rewritten during integration.

## 5. Optional Neo4j backend

The offline backend provides the same bounded typed paths as the existing engine. To meet the proposal's Neo4j runtime path, install `uv sync --locked --extra neo4j` and use a separate Neo4j 5+ development database. Credentials belong in local `.env`, never Git:

```text
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your local password>
NEO4J_DATABASE=neo4j
ACTIVITY_KG_VERSION=0.2.0
```

The CLI loads `.env`. In your development database, import `travel_agent/understanding/kg_data/activity/seed.cypher` through Neo4j Browser or `cypher-shell`. The version-scoped seed uses MERGE and performs no deletes. Reimporting this snapshot updates that version's properties, including reviews, so do not use it on an already reviewed shared graph; use a separate development DB/version or a reviewed migration.

Add `"kg_backend":"neo4j"` to the input payload. The adapter reads graph data and verifies retained one-/multi-hop paths using bounded Cypher traversal. The shared engine enforces direction, node/relationship types, conditions, evidence, depth and review filtering. It never auto-imports or approves knowledge. Real Neo4j connection/query behaviour still requires the live check described above; mocked or offline tests are not a live DB verification.

## Today's scope

The reproducible offline pipeline, structured patches, local-Qwen review path and live city weather are implemented. Draft knowledge, prototype thresholds, model semantics, persistent session integration and the live Neo4j connection still need the checks above. No product retrieval, learned compatibility, complete recommendations, or deployment is claimed by this delivery.

Driver/query references: [Neo4j Python sessions](https://neo4j.com/docs/python-manual/current/query-advanced/), [bounded variable-length paths](https://neo4j.com/docs/cypher-manual/current/patterns/reference/variable-length-paths/). The mock adapter test checks seed-property decoding and path retention; it does not execute Cypher against a server.

## Broader real-model diagnostic set (2026-10-11)

A separate frozen synthetic set now contains 32 scenarios / 40 turns and three continuous dialogue rollouts. See [coverage and scoring](../evaluation/datasets/README.md), [all messages and gold facts](../evaluation/datasets/qwen_understanding_v1_cases.md), and the [audited results](../evaluation/results/qwen_understanding_v1_audited.md). Gold labels are assistant-authored and pending human review. The prompt and production guards were not tuned during this evaluation.

The audited strict full-correctness results are 3/40 for raw Qwen and 14/40 after existing guards; no rollout reached its complete gold final state. The earlier 4/7 smoke result remains a development-set result with a narrower subset-field comparison. These scores are not directly comparable. Continue supervised extraction review; the broader set exposes additional omissions, hallucinations, negation/patch errors and activity/dress-code key consistency issues.


## Season output extension (2026-10-12)

Outputs now include `data.season` and `data.season_context`, and include supported season tokens in `expanded_fashion_context`. The Seoul winter fixture now starts `Seoul winter cold ...`. This is deterministic calendar derivation with location provenance, not Qwen inference, and is distinct from the weather class. Cross-season trips use null plus a seasons list; unsupported/tropical locations stay null. See [the season handoff contract](contracts.md#season-handoff-2026-10-12). Historical Qwen evaluation files remain unchanged; they evaluated extraction before this output-only extension.

## Current English release

English scenarios are the primary workflow. Follow the README for the supervised Qwen preview command and `evaluation/README.md` for current English diagnostics. The historical multilingual scores above remain archived evidence and are not the current English release scores. Raw Qwen output remains separate from Qwen plus deterministic source-evidence reconciliation.

Review each proposed request/edit/confirmation before accepting it. Team review is still required for KG assertions, weather thresholds, season conventions and downstream schema adoption. Live Neo4j, product/recommendation modules and autonomous end-to-end agent integration remain unverified or unimplemented.

## Incremental collection and empty optional results

Required fields are destination, start/end dates and budget including currency. Activities, styles, places and dress codes are optional. Empty activity output maps, empty clothing arrays and `place_context=null` are valid; missing occasion codes appear in `optional_questions` and do not block downstream work. No activity is invented when none was provided. Explicit null optional collections normalize to empty lists/maps.

For a multi-turn local dialogue, use the same state file for every message:

```bash
uv run python -m travel_agent.understanding.demo --message 'I am going to Seoul.' --save-state artifacts/session.json
uv run python -m travel_agent.understanding.demo --message '2026-12-20 to 2026-12-27' --state artifacts/session.json --save-state artifacts/session.json
uv run python -m travel_agent.understanding.demo --message '500 SGD' --state artifacts/session.json --save-state artifacts/session.json
```

Review the final accumulated `proposed_request`, then confirm it without repeating any input or rerunning the model:

```bash
uv run python -m travel_agent.understanding.demo --confirm-pending --state artifacts/session.json --save-state artifacts/session.json
```

A corrected structured `patch` also merges onto pending facts and applies them; an empty patch accepts them unchanged after user review. Single ISO dates need a start/end label unless exactly one date slot is still missing. Relative/yearless dates and ambiguous currencies still need clarification. The LangChain caller must carry returned state between tool calls; automatic production session persistence is still not configured.
