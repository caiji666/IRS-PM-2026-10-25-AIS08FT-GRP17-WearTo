# Five-module full-pipeline integration (Module 11)

Student 5 allocation item 7: coordinate end-to-end integration and testing across all five
modules. This page describes the test, what it really exercises today and what is missing.

**The five-module pipeline has not passed end to end, and no such pass is claimed.**

- **Default (offline) runs** stop at Student 2 with `prerequisite_missing`: its live run needs a
  model, an API key and network access.
- **Student 4 update:** The dynamic backend is implemented; a trained normalized model and compatible current handoff are required. See [outfit integration](outfit-integration.md).
- **Student 5** is implemented in this repository (`travel_agent.recommendation`) and is called
  through `travel_agent.recommendation.service.run`. Its adapter now accepts the dynamic Student 4 output; the original budget currency must still match purchase prices.

**Historical harness notes:** The detailed harness observations below describe the earlier scaffold baseline. For the current real Student 1–4 runner, dynamic contracts and validation, use [outfit integration](outfit-integration.md). The five-module agent harness is not the same as that verified backend runner.

## Module status

| Module | Entry point | Status in this repository |
|---|---|---|
| Student 1 | `travel_agent.understanding.service.run` | Implemented; runs offline with the reviewed structured example (synthetic weather, draft KG). |
| Student 2 | `travel_agent.inspiration.service.run` | **Implemented**. See below. |
| Student 3 | `travel_agent.products.service.run` | Implemented. A live run needs the products extra, FashionCLIP weights and a product data source. |
| Student 4 | `travel_agent.outfits.service.run` | Dynamic normalized-MLP scorer; explicit inputs and local trained model required. |
| Student 5 | `travel_agent.recommendation.service.run` | Implemented (`travel_agent/recommendation/student5/`): Student 4 adapter, Stage 10 constraints, Stage 12 ranking, Stage 13 decision loop, Stage 14 preview. Typed handoff verified separately; complete agent orchestration still needs validation. |

Student 2 details:

- **What it does:** FashionCLIP context and image embeddings, Unsplash retrieval, cosine
  reranking, and a file-based handoff to Student 3.
- **What a live run needs:** the `inspiration` extra, the FashionCLIP weights,
  `UNSPLASH_ACCESS_KEY` and network access.
- See [inspiration-runbook.md](inspiration-runbook.md).

## Files

| File | Purpose |
|---|---|
| `tests/test_full_pipeline_integration.py` | The integration, contract and boundary tests. |
| `tests/full_pipeline_harness.py` | Runner, prerequisite checks, handoff checks and run-record builder shared by the tests and the script. Not a test module. |
| `scripts/run_full_pipeline_integration.py` | Optional: runs the pipeline once and writes a run record. |
| `pyproject.toml` | Registers the `full_pipeline_live` pytest marker. |

## Flow

```
fixed request ─► Student 1  understanding.service.run          → ModuleResult
              ─► Student 2  inspiration.service.run           ({"understanding": Student 1 .data})
                              └─ data["product_handoff"]      (upstream files + reference_mode)
              ─► Student 3  products.service.run              (payload = product_handoff, unchanged)
              ─► Student 4  outfits.service.run               (scaffold: not_implemented)
              ─► Student 5  recommendation.service.run        ({"action": "recommend", "request", "student4"})
                              └─ student4_adapter → Stage 10 → Stage 12
              ─► Top-3 RecommendationSet or structured ConstraintConflict
              ─► action "confirm" by outfit_id ─► action "preview" (MockPreviewProvider), same entry point
```

The harness only calls these entry points. It never writes a module's output, never fills or
drops fields between modules, and never re-implements constraints, scoring, ranking, state
updates, retrieval or preview.

A run stops at the first stage that cannot continue, and keeps that stage's status:

| Status | Meaning |
|---|---|
| `needs_input` / `no_candidates` / `error` | returned by a real module |
| `not_implemented` | returned by a scaffold entry point; never assigned by the harness |
| `prerequisite_missing` | a model, extra, credential, data source or the live opt-in is missing; checked before the call |
| `handoff_undefined` | no agreed contract yet |

Data that violates a contract raises `HandoffContractError` and fails the test.

## Handoff contracts

**Student 1 → Student 2** (`student2_payload`).

- **Payload:** `{"understanding": <Student 1 ModuleResult.data>}`, never the `ModuleResult`
  envelope. `top_n` is added only when given explicitly; otherwise Student 2's default applies.
- **Checked before the call:**
  - Student 1 `status == "ok"`;
  - `data` is an object, and `ready_for_downstream` is true;
  - `trip_request.destination` is non-empty;
  - `season` is present;
  - `expanded_fashion_context` is a non-empty string;
  - `weather_clothing_requirements` and `activity_style_requirements` are objects.

  These are the fields `inspiration.service.prepare_inspiration_context` reads.

**Student 2 → Student 3** (`student3_payload`).

- **Payload:** `ModuleResult.data["product_handoff"]`, written by Student 2's
  `save_inspiration_handoff`, passed **unchanged** as the Student 3 payload:
  `{"upstream": {"pt", "text", "context", "model_info"}, "reference_mode": "imported"}`.
  The paths are relative to `STUDENT3_DATA_ROOT`.
- **Checked:**
  - Student 2 `status == "ok"`;
  - `product_handoff` is present and is an object;
  - it validates against Student 3's public `ProductToolInput`
    (`travel_agent/products/retrieval/integration.py`);
  - `upstream` has `pt`, `text` and `context` (plus optional `model_info`) and no other keys;
  - every path is relative.
- **Run records** keep only the reference mode and upstream keys. Embeddings and image lists
  are never recorded.
- **Note:** `product_handoff` sets no `source`, so Student 3 uses its default `photos20`. That
  needs the private `data/photos20` dataset under `STUDENT3_DATA_ROOT`. This is checked as a
  prerequisite and never changed by the harness.

**Student 3 → Student 4.** No agreed contract. The harness calls the real entry point with a
PROVISIONAL payload `{"understanding": Student 1 data, "products": Student 3 data}` and records
its status (currently `not_implemented`).

**Student 4 → Student 5.** The harness calls `travel_agent.recommendation.service.run` with
`{"action": "recommend", "request": <projected request>, "student4": <Student 4 output,
unchanged>, "beta", "session_id"}`, plus `state` for a follow-up round.

- **Contract.** Student 5's entry point uses an explicit `action` contract: `recommend`,
  `modify_requirements`, `confirm`, `explain` and `preview`. A missing or unknown `action`
  returns `error`. A `patch` keeps its own Stage 1 `intent: "MODIFY_REQUIREMENTS"`.
- **Adapter.** Inside the service, Student 5's `student4_adapter` converts the Student 4 output.
  The Student 4 output format and the adapter remain PROVISIONAL until the real Student 4
  hand-off.
- **Contract violations** return `error` with the adapter's `code` and `location`, and the
  harness raises `HandoffContractError`.
- **No feasible outfit.** The service returns `no_candidates` with the `ConstraintConflict`. This
  completes the round and is not a failure.
- **Other operations.** Confirmation (`confirm`), requirement changes (`modify_requirements`) and
  the mock preview (`preview`) also go through `recommendation.service.run`.

**Request into Student 5.** A minimal projection is implemented
(`travel_agent.recommendation.projection.project_trip_request`, used by `student5_request`).

- **Fields taken.** From Student 1's validated `trip_request`, only the eight shared fields:
  `destination`, `start_date`, `end_date`, `activities`, `budget` (major units; checked against
  Student 1's `budget_minor`), `currency`, `style` and `excluded_items`. Omitted list fields take
  Student 1's schema defaults.
- **Fields excluded.** Student 1 extension fields (`owned_item_ids`, `places`, `dress_codes`,
  `enter_interior`, `gender`, `cold_sensitive`, `price_preference`) are not forwarded. Student 5's
  `TripShoppingRequest` still forbids extra fields.
- **Missing required fields** (`destination`, `start_date`, `end_date`, `budget`, `currency`) are
  never invented. Student 5 stops with `needs_input`, and the record lists the missing fields.
- **`user_size`** is not set. Its source is still to be agreed by the team; with
  `user_size=None`, Student 5 does not enable the size constraint.

## Tests

**Default, offline.** None of these is an end-to-end test.

- Entry points of Students 1–5 import; the Student 2 import needs the `inspiration` extra.
- Student 2's real service answers an empty payload with `needs_input`, not `not_implemented`.
- Student 5 is reached through `travel_agent.recommendation.service.run`. It answers an empty
  payload with `needs_input`, and the harness imports no external `student5` package.
- Student 1 handoff for the fixed reviewed request.
- The flow stops and keeps Student 1 `needs_input`.
- The default run stops at Student 2 with `prerequisite_missing` and a specific reason (opt-in,
  weights, `UNSPLASH_ACCESS_KEY`), never `not_implemented`.
- The pending report names the stop status and real gaps, and contains no unsafe values.
- **Student 1 → Student 2 contract:**
  - `student2_payload` passes `s1.data`, not the envelope;
  - incomplete or not-ready data fails with a named field.
- **Student 1 → Student 2 boundary:** the real Student 2 service runs with its retrieval
  pipeline isolated by monkeypatch, and the test checks that Student 1's context fields arrive.
- **Student 2 → Student 3 contract:** the product handoff is passed unchanged. It must fail
  explicitly when:
  - `product_handoff` is missing;
  - it has the wrong type;
  - `upstream` is missing a required file;
  - a key is unknown or a path is absolute or uses `..`;
  - it breaks the schema (`reference_mode`, extra fields).
- **Student 2 → Student 3 boundary:**
  - The real Student 2 service runs with a fixed unit-vector pipeline output (no model, no
    Unsplash).
  - Its real `save_inspiration_handoff` writes the files into a temporary
    `STUDENT3_DATA_ROOT`.
  - Student 3's real `ProductToolInput` and `prepare_input` accept them as
    `declared_compatible`.
  - The run-record summary contains no embeddings.
- **Student 4 boundary:**
  - A Student 3 result from Student 3's own offline test mode (prepared catalogue plus an
    explicit encoder test double) reaches the real Student 4 entry point.
  - The chain stops at Student 4 with `not_implemented` and a skip reason that names
    Student 4.
- **Run-record safety check.**
- **Student 1 / Student 5 shared request fields** are checked for consistency.
- **Request projection into Student 5.**
  - The real Student 1 request is projected onto the eight shared fields; extension fields
    are dropped and `user_size` is None.
  - A copy of Student 1's output with required fields removed stops Student 5 with
    `needs_input` and the missing field names.

The Student 5 service contract itself is tested in `tests/recommendation/`: the migrated Student 5
unit tests and `test_recommendation_service.py`, which uses mock Student 4-shaped fixtures.

**Full pipeline** (`full_pipeline_live`). These are skipped with the blocking stage until every
real module runs:

1. normal request → 1–3 outfits;
2. infeasible budget → structured `ConstraintConflict`;
3. missing compatibility score → explicit adapter error;
4. `needs_input` / `no_candidates` / `error` stop the flow;
5. contract mismatches fail;
6. a requirement change re-runs the needed stages;
7. confirm the latest `outfit_id`;
8. old or invalid `outfit_id` rejected;
9. `MockPreviewProvider` preview;
10. safe run record;
11. natural-language request through Student 1 review (local Qwen weights).

## Running

```bash
uv sync --locked --all-extras          # as CI; packages only, no model weights
uv run pytest tests/test_full_pipeline_integration.py -v
uv run pytest -q
```

On Windows, set `PYTHONUTF8=1` for the test process, as on the Linux CI. Some existing
`read_text()` calls rely on the platform encoding.

**Default behaviour.** Default test runs:

- download no model and call no external API;
- do not read `.env`; prerequisites are checked from installed packages, the local Hugging Face
  cache and the process environment.

`full_pipeline_live` tests stay skipped until every real module runs.

**Live prerequisites.** A live run needs all of the following:

- `RUN_FULL_PIPELINE_TESTS=1`;
- the `inspiration` and `products` extras;
- the pinned FashionCLIP revision in the local Hugging Face cache (Student 2) and in
  `STUDENT3_MODEL_CACHE` (Student 3);
- `UNSPLASH_ACCESS_KEY` in the process environment;
- network access;
- a Student 3 product source: `data/photos20` for the default source, the prepared catalogue,
  or eBay Sandbox credentials.

Even when all of these are met, the chain currently stops at Student 4 (`not_implemented`).

Student 5 needs no separate installation: it is part of the `travel_agent` package.

**Run record (optional).**

```bash
uv run python -m scripts.run_full_pipeline_integration --output-dir artifacts/integration
```

`full_pipeline_run.json` is written only after all five real modules completed and the checks
passed (exit 0). Otherwise `full_pipeline_run.pending.json` is written, and it is never
reported as passed:

| Pending status | Exit code | Meaning |
|---|---|---|
| `blocked_by_missing_prerequisites` | 2 | a model, extra, credential, data source or opt-in is missing |
| `blocked_by_undefined_handoff` | 2 | no agreed contract yet |
| `blocked_by_missing_modules` | 2 | a scaffold returned `not_implemented` |
| `stopped_by_module_status` | 1 | a real module returned `needs_input`, `no_candidates` or `error` |

A contract failure also exits 1.

Records contain the input, run time, module statuses, versions, data sources, safe summaries,
contract probes and gaps. They never contain keys, tokens, portraits, base64 images, absolute
paths, embeddings or other large vectors.

## Current gaps

| Owner | Needed for a full end-to-end run |
|---|---|
| Student 2 | Nothing structural. Live runs need the FashionCLIP weights, `UNSPLASH_ACCESS_KEY` and network. |
| Student 2 + Student 3 | Decide which Student 3 product `source` a pipeline run uses. `product_handoff` sets none, so the default `photos20` needs the private local dataset. |
| Student 3 | A product source with verified prices and availability; carry `product_url` and the product score through to Student 4. |
| Student 4 | Implement `outfits.service.run` with an input contract for Student 3 results. Return outfits with a real `compatibility_score` and Student 5 `Product` fields. **Current structural blocker.** |
| Student 4 + Student 5 | Final verification of real Student 4 output through `recommendation.service.run` (its `student4_adapter`), and agreement on the PROVISIONAL Student 4 payload and Student 5 service payload. |
| Students 1, 4 + 5 | Agree the source of Student 5 `user_size` (the minimal eight-field request projection is implemented), and agree who owns the conversation-state transitions. |
