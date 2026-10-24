# Selective Full-Pipeline Requirement Refinement

## Purpose

Allow a user who already has a Student 5 Top-3 recommendation to modify any supported trip or shopping requirement and receive a new, internally consistent Top-3. Changes that affect upstream evidence must rerun the required Student 1–4 stages before Student 5 ranks the new outfits. The user can then confirm one of the new outfits and request an AI preview.

## Current Gap

The web demo currently creates a Student 5 session from a fixed Student 4 output. Its refinement endpoint directly supports budget, size, and product exclusion, but rejects date, destination, activity, and style changes. Student 5 already produces a correct rerun plan, and the repository already contains a dynamic Student 1–4 pipeline, but the web refinement flow does not connect those two capabilities or continue the dynamic pipeline into Student 5.

## User Experience

### Fast refinements

Budget, size, and product-exclusion changes remain direct Student 5 refinements. They update the recommendation state and Top-3 without opening the full-pipeline confirmation dialog.

### Upstream refinements

Date, destination, activity, style, currency, and other upstream-dependent changes return a confirmation-required response. The dialog shows:

- which fields changed;
- the earliest affected stage;
- the stages that will rerun;
- that weather, product, or model APIs may be called;
- that the operation can take several minutes.

After confirmation, a background job runs while the old Top-3 remains visible but read-only. The progress dialog advances through trip update, weather, inspiration, products, outfit scoring, and final ranking. On success, the new Top-3 atomically replaces the old one. On failure, the old Top-3 remains available and the UI identifies the failed stage with retry and return actions.

Any successful requirement change invalidates the previous confirmation and preview. The user must confirm an outfit from the newest recommendation round before generating a new preview.

## API Design

### `POST /api/student5/refine`

Accepts the current Student 5 session id and either a structured patch or a natural-language message.

- A Student 5-only change returns the updated session and new Top-3 immediately.
- An upstream-dependent change validates the patch and returns:

```json
{
  "status": "confirmation_required",
  "pending_change_id": "32-character identifier",
  "changed_fields": ["start_date", "end_date"],
  "rerun_from_stage": 2,
  "affected_stages": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
  "message": "This change requires weather, products and outfits to be updated."
}
```

The pending change is stored server-side and bound to the current session and request version. The client cannot alter the validated patch or rerun plan after receiving this response.

### `POST /api/student5/refine/confirm`

Accepts the session id and pending change id. It rejects expired, already consumed, mismatched, or stale changes. A successful request creates one background job and returns its id. Repeated submission of the same pending change returns the existing job rather than starting duplicate API work.

### `GET /api/job/{id}`

The existing job endpoint remains the progress interface. For refinement jobs it may return:

- `status`: `running`, `completed`, or `failed`;
- `stage`: the current pipeline stage;
- a public progress message;
- the updated public Student 5 session on completion;
- a safe public failure category on error.

No local file paths, provider responses, credentials, embeddings, or provenance data are exposed.

## Session State

Each Student 5 web session stores:

- the current validated trip-shopping request;
- Student 5 conversational state and current recommendation set;
- the current Student 4 handoff;
- references to the latest reusable pipeline artifacts;
- the confirmed outfit and preview, when valid;
- at most one pending upstream change;
- at most one active refinement job.

The previous successful session snapshot remains unchanged while a background refinement is running. The replacement is committed under the session lock only after the new Student 4 handoff and Student 5 Top-3 have both passed validation.

## Pipeline Coordinator

A deterministic coordinator, rather than the language model, decides which modules run. It consumes the validated requirement patch, Student 5 rerun plan, previous artifact references, and a progress callback.

The coordinator reuses existing public module services:

1. update and validate the shared trip request;
2. rerun weather and trip context when required;
3. rerun inspiration/context stages when required;
4. rerun product retrieval and product ranking when required;
5. call `travel_agent.outfits.service.run` for Student 4;
6. build the Student 5 handoff with `travel_agent.outfits.output.recommendation_input`;
7. call `travel_agent.recommendation.service.run` for the new Top-3.

The Student 5 rerun plan remains the dependency source of truth. The coordinator maps the earliest affected stage to the earliest reusable module boundary. It never allows an LLM to skip required deterministic stages.

## Checkpoint Reuse

Dynamic pipeline runs save stage outputs and their request identity. A checkpoint is reused only when its relevant input fields and source identities still match the updated request. Downstream checkpoints are never reused after an upstream dependency changes.

The current fixed Student 4 demonstration input has no matching Student 1–3 checkpoint chain. Therefore its first upstream-dependent refinement performs a complete dynamic run. Once that run succeeds, subsequent refinements can reuse its checkpoints according to the rerun plan.

## Concurrency and Failure Handling

- Only one upstream refinement job may run per Student 5 session.
- Confirmation, preview generation, and additional refinement submission are disabled while it runs.
- The old Top-3 stays visible and read-only.
- A failed job cannot partially replace the current session.
- A successful job clears the prior confirmed outfit and preview.
- A stale recommendation cannot be confirmed or previewed.
- API and model errors are converted into stage-specific public messages; raw external responses remain in private logs.
- Retry creates a new job from the last successful session snapshot.

## Frontend Behaviour

The Chatbox remains the single requirement-editing surface. The Adjust requirements button focuses it and explains supported changes.

For upstream changes, the frontend:

1. displays the confirmation dialog;
2. submits the pending change only after user approval;
3. shows stage progress while retaining the old Top-3;
4. prevents duplicate actions;
5. replaces the Top-3 only on success;
6. scrolls to the updated recommendations;
7. requires a new confirmation before enabling AI Preview.

The progress sequence shown to users is:

```text
Updating your trip
Checking weather
Finding inspiration
Searching products
Building outfits
Ranking your new Top-3
```

## Security and Data Boundaries

- All request patches are validated by the shared request schema and Student 5 patch processor.
- Pending changes are stored server-side and referenced by opaque identifiers.
- Artifact paths remain server-side.
- Product and preview media continue to use constrained local routes.
- The browser never receives embeddings, provenance, API keys, provider configuration, or raw model output.

## Verification

Automated tests must prove:

- date changes plan and execute the Stage 2–12 dependency chain;
- activity and style changes start at their mapped stages;
- budget, size, and exclusion changes do not create full-pipeline jobs;
- duplicate confirmation cannot create duplicate jobs;
- the coordinator passes the new Student 4 result into Student 5;
- success atomically replaces the Top-3 and invalidates old confirmation and preview state;
- failure preserves the previous Top-3;
- stale recommendations cannot be confirmed;
- the new Top-3 can be confirmed and previewed;
- the frontend shows confirmation, progress, success, and failure states;
- private artifacts and secrets never enter public responses.

One real-data acceptance run must change travel dates, execute through Student 4 and Student 5, produce a different validated Top-3 where the inputs warrant it, confirm a new outfit, and generate a real preview. This acceptance run is separate from offline automated tests because it incurs external API calls and cost.

## Scope

This design covers requirement refinement after an initial Top-3 exists. It does not redesign the module scoring algorithms, Student 4 model, currency conversion, portrait storage, or AI image provider. It reuses those existing interfaces and adds deterministic orchestration around them.
