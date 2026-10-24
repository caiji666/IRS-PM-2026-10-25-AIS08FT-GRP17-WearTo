# Selective Full-Pipeline Requirement Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let web users modify upstream trip requirements, approve the resulting rerun plan, execute the real Student 1–5 pipeline in a background job, and then confirm and preview an outfit from the new Top-3.

**Architecture:** Student 1 parses a natural-language refinement against the current request; Student 5 remains the source of truth for affected stages. A deterministic coordinator executes or reuses module checkpoints, and the web layer atomically replaces a session only after the new Student 4 handoff and Student 5 recommendation both validate.

**Tech Stack:** Python 3.12, Pydantic, existing module services, `ThreadingHTTPServer`, subprocess/background jobs, vanilla JavaScript, pytest.

**Spec:** `docs/superpowers/specs/2026-10-22-selective-pipeline-refinement-design.md`

## Global Constraints

- Keep all work local: no commit, push, or pull request.
- Preserve the last successful Top-3 until a replacement pipeline completes successfully.
- Never expose local paths, embeddings, provenance, credentials, or raw provider output.
- Never create duplicate external API work from repeated confirmation.
- A successful upstream refinement invalidates the previous confirmation and preview.
- Real API acceptance runs require explicit cost-aware confirmation.

## Review Focus

- A stale pending change must fail after the request version changes; Task 2 tests this binding.
- Duplicate confirmation must return the existing job rather than start another; Task 4 tests idempotency.
- A failed upstream module must preserve the complete previous session; Task 3 tests atomic replacement.
- A fixed-input session without checkpoints must fall back to a complete run; Task 3 tests this path.
- Natural-language dates with an invalid or reversed range must be rejected before any job starts; Task 1 tests validation.

---

### Task 1: Parse and Validate Upstream Requirement Changes

**Files:**
- Modify: `travel_agent/web/student5_session.py`
- Test: `tests/test_web.py`

**Interfaces:**
- Consumes: current Student 5 request and natural-language user message.
- Produces: `propose_refinement(session: dict, message: str, understand: Callable) -> dict` with a validated patch, updated request, Student 1 state, and Student 5 rerun plan.

- [ ] **Step 1: Write failing tests** for date, destination, activity, and style messages; reversed dates; extraction review; and unchanged input.
- [ ] **Step 2: Run the focused tests** and verify failures are caused by the missing proposal interface.
- [ ] **Step 3: Implement `propose_refinement`** by constructing a Student 1 conversation state from the current request, extracting against that state, validating the proposed request, deriving an exact patch, and calling Student 5 `modify_requirements` only to obtain its validated updated state and rerun plan.
- [ ] **Step 4: Keep `patch_from_message`** for fast budget, size, and product-exclusion changes; return a typed upstream proposal for all other supported fields rather than the current generic rejection.
- [ ] **Step 5: Run focused tests** and verify all parsing and validation cases pass.

### Task 2: Pending Change Store and Stale-Change Protection

**Files:**
- Create: `travel_agent/web/refinement_jobs.py`
- Test: `tests/test_web_refinement.py`

**Interfaces:**
- Consumes: a session id, current request version, validated patch, rerun plan, and updated Student 1 state.
- Produces: `PendingRefinement`, `create_pending(...)`, `consume_pending(...)`, and `public_plan(...)`.

- [ ] **Step 1: Write failing tests** for opaque ids, one pending change per session, session mismatch, request-version mismatch, expiration, one-time consumption, and sanitized public output.
- [ ] **Step 2: Run the focused tests** and confirm the module is missing.
- [ ] **Step 3: Implement the in-memory store** under a lock, using 32-character opaque ids and an explicit expiration timestamp.
- [ ] **Step 4: Implement idempotent consumption state** so repeated confirmation can refer to one active job without rerunning external work.
- [ ] **Step 5: Run focused tests** and verify all pending-change lifecycle cases pass.

### Task 3: Deterministic Pipeline Coordinator

**Files:**
- Create: `travel_agent/refinement_pipeline.py`
- Modify: `scripts/run_live_product_pipeline.py`
- Test: `tests/test_refinement_pipeline.py`

**Interfaces:**
- Consumes: `run_refinement(request, rerun_plan, checkpoints, progress, services) -> RefinementOutcome`.
- Produces: a validated request, updated checkpoint references, Student 4 handoff, Student 5 state, recommendation set, and filter summary.

- [ ] **Step 1: Write failing offline tests** with injected fake services for Stage 2–12 date changes, mapped activity/style starts, checkpoint reuse, no-checkpoint full-run fallback, Student 4-to-5 handoff, progress ordering, and failure atomicity.
- [ ] **Step 2: Run the focused tests** and confirm the coordinator is missing.
- [ ] **Step 3: Extract reusable stage functions** from `run_live_product_pipeline.py` without changing its existing CLI outputs.
- [ ] **Step 4: Implement `run_refinement`** so the rerun plan selects the earliest reusable module boundary and invalidates all downstream checkpoints.
- [ ] **Step 5: Add final Student 5 execution** after `travel_agent.outfits.output.recommendation_input`, validating both Student 4 and Student 5 results before constructing `RefinementOutcome`.
- [ ] **Step 6: Run coordinator and existing pipeline tests** and verify both new and existing entry points pass.

### Task 4: Web Confirmation Endpoint and Background Job

**Files:**
- Modify: `travel_agent/web/__main__.py`
- Modify: `travel_agent/web/student5_session.py`
- Test: `tests/test_web_refinement.py`

**Interfaces:**
- Consumes: Task 1 proposals, Task 2 pending ids, and Task 3 coordinator.
- Produces: enhanced `POST /api/student5/refine`, new `POST /api/student5/refine/confirm`, and refinement progress through `GET /api/job/{id}`.

- [ ] **Step 1: Write failing HTTP tests** for confirmation-required responses, confirmation, duplicate confirmation, active-job rejection, completion, sanitized failure, and retained old Top-3.
- [ ] **Step 2: Run the focused tests** and verify endpoint failures.
- [ ] **Step 3: Update `/api/student5/refine`** to keep fast refinements synchronous and store upstream proposals server-side.
- [ ] **Step 4: Add `/api/student5/refine/confirm`** to create one background job or return its existing id.
- [ ] **Step 5: Add the refinement worker** that emits public stage progress and atomically swaps the Student 5 session under `STUDENT5_LOCK` only on success.
- [ ] **Step 6: Ensure success clears confirmation and preview** while failure leaves the previous session byte-for-byte equivalent.
- [ ] **Step 7: Run HTTP and session tests** and verify all cases pass.

### Task 5: Frontend Confirmation and Progress Flow

**Files:**
- Modify: `travel_agent/web/static/index.html`
- Modify: `travel_agent/web/static/app.js`
- Modify: `travel_agent/web/static/style.css`
- Test: `tests/test_web.py`

**Interfaces:**
- Consumes: confirmation-required responses and refinement job status from Task 4.
- Produces: confirmation dialog, rerun progress, retry/return failure controls, and atomic Top-3 replacement in browser state.

- [ ] **Step 1: Write failing static and browser-facing tests** for the confirmation dialog, changed fields, stage list, disabled controls, retained old cards, success replacement, and failure controls.
- [ ] **Step 2: Run focused tests** and verify the new UI states are absent.
- [ ] **Step 3: Add the confirmation dialog** with explicit API/time warning and continue/cancel actions.
- [ ] **Step 4: Extend polling** to map refinement stages to the approved progress sequence while leaving the old Top-3 rendered but read-only.
- [ ] **Step 5: On success** replace `state.outfits`, reset selected outfit, confirmation, and preview, then scroll to The edit.
- [ ] **Step 6: On failure** retain the old state and show Retry and Back to recommendations.
- [ ] **Step 7: Run web tests and JavaScript syntax checks** and verify all UI states pass.

### Task 6: End-to-End Offline Integration

**Files:**
- Modify: `tests/test_full_pipeline_integration.py`
- Modify: `tests/full_pipeline_harness.py`

**Interfaces:**
- Consumes: the complete Tasks 1–5 flow with injected offline providers.
- Produces: repeatable proof that a date change reaches Student 4 and Student 5 and yields a confirmable new recommendation round.

- [ ] **Step 1: Write a failing integration test** starting from an existing Top-3, changing dates, approving rerun, observing all expected stages, receiving a new Top-3, confirming a new outfit, and invoking a fake preview provider.
- [ ] **Step 2: Add failure-path integration tests** for upstream failure, stale recommendation confirmation, and duplicate approval.
- [ ] **Step 3: Run integration tests** and fix only issues within the approved design.
- [ ] **Step 4: Run all Student 5, web, contract, and full-pipeline tests** and record exact results.
- [ ] **Step 5: Run Ruff, format, JavaScript syntax, and `git diff --check`** and record exact results.

### Task 7: Real-Data Acceptance Run

**Files:**
- Output only: `artifacts/` and `runs/` ignored directories.
- No production-code change unless a verified defect is first reproduced by a failing test.

**Interfaces:**
- Consumes: configured real weather, inspiration, eBay, Student 4, Student 5, and preview services.
- Produces: one saved acceptance record for a date change through new Top-3, confirmation, and real AI Preview.

- [ ] **Step 1: Verify required environment-variable names and provider files exist** without printing secret values.
- [ ] **Step 2: Start from the current fixed Student 4 session** and submit an explicit date change through the frontend.
- [ ] **Step 3: Approve the cost warning once** and observe the pipeline through Student 4 and Student 5.
- [ ] **Step 4: Verify the new request dates, Student 4 artifact identity, recommendation round, Top-3, and invalidated old confirmation.**
- [ ] **Step 5: Confirm one new outfit and generate one real AI Preview**; verify the browser receives a constrained media URL.
- [ ] **Step 6: Save a sanitized acceptance summary** without credentials, embeddings, provenance, or private paths.

## Execution Notes

- Implement in the current local branch because it contains the uncommitted Student 5 and UI integration work this feature depends on.
- Use test-driven development for every task: observe the focused failure before production edits, then rerun the focused and broader suites.
- Do not commit or push, despite generic plan templates normally including commit steps; the user explicitly prohibited both.
