# Student 4 → Student 5 Local Integration Implementation Plan

> **For agentic workers:** Execute this plan natively in the current session. Do not commit or
> push. Use test-driven development for every production behavior change.

**Goal:** Connect the current local Student 5 work to the latest dynamic Student 4 public service,
remove the unnecessary same-currency FX dependency, and reduce the restored Student 5 suite to a
course-appropriate regression set.

**Architecture:** Port the latest Student 4 public boundary from `origin/main` without merging the
dirty branch. The full-pipeline harness calls that boundary and hands its `ModuleResult.data` to
Student 5. Student 5 evaluates same-currency requests directly and invokes ECB conversion only for
genuinely mixed currencies.

**Tech Stack:** Python 3.11, Pydantic, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-10-22-student4-student5-local-integration-design.md`

## Global Constraints

- Preserve local uncommitted work and all divergent web/Student 1 files.
- Do not merge, rebase, commit, or push.
- Do not call paid APIs.
- Cached real output must remain labelled separately from live end-to-end output.
- Final changes must be applicable to current `origin/main` without semantic conflicts.

## Review Focus

- All-request and all-product currencies are identical: recommendation must run without FX data.
- Product currencies differ from the request: conversion must still be required and validated.
- Student 4 returns `ModuleResult(status="ok", data=...)`: only `data` reaches the Student 5 adapter.
- Student 4 returns non-ok status: the pipeline stops at Student 4 and preserves its reason.
- Test reduction must retain business rules, representative SSRF checks, privacy, state, and module boundaries.

---

### Task 1: Establish a recoverable local baseline

**Files:**
- Read only: the full current working tree and `origin/main`
- Record: temporary inventory outside tracked source files

- [ ] Capture modified and untracked paths, exact revisions, Python environment, targeted collection
      count, and targeted test results.
- [ ] Diff local in-scope files against both `HEAD` and `origin/main`; explicitly exclude divergent
      `travel_agent/web/`, Student 1, and weather files from edits.
- [ ] Save copies of every in-scope file before changing it in a temporary recoverable directory.

### Task 2: Make same-currency recommendations independent of ECB FX

**Files:**
- Modify: `travel_agent/recommendation/service.py`
- Modify: `travel_agent/recommendation/currency_converter.py` only if the service boundary alone
  cannot express the rule cleanly
- Test: `tests/recommendation/test_recommendation_service.py`
- Test: `tests/recommendation/test_recommendation_currency_converter.py`

- [ ] Add a failing test where request and every product use USD, the FX loader raises if called,
      and recommendation still succeeds in USD.
- [ ] Run that test and confirm failure occurs because the service calls the FX loader.
- [ ] Implement the minimal rule: convert only when an item currency differs from the request
      currency; otherwise use the request and candidates unchanged.
- [ ] Verify same-currency success, mixed-currency conversion, unavailable-rate failure, budget,
      totals, and display currency.

### Task 3: Port and verify the latest Student 4 public service

**Files:**
- Port from `origin/main`: `travel_agent/outfits/__main__.py`
- Port from `origin/main`: `travel_agent/outfits/engine.py`
- Port from `origin/main`: `travel_agent/outfits/output.py`
- Port from `origin/main`: `travel_agent/outfits/service.py`
- Port from `origin/main`: `travel_agent/outfits/tools.py`
- Port from `origin/main`: `travel_agent/outfits/train.py`
- Port from `origin/main`: `outfit-compatibility/compatibility_service.py`
- Port/adapt: `tests/test_outfits_dynamic.py`

- [ ] Run the latest main Student 4 dynamic test against the current local implementation and
      confirm it fails because the local service is still a scaffold.
- [ ] Port the latest main implementation without unrelated main changes.
- [ ] Run dynamic Student 4 tests with injected model/manifest fixtures, including the all-USD
      Student 5 contract path with network disabled and no FX cache.
- [ ] Confirm the public service never silently reads fixed `result.json`.

### Task 4: Connect the full-pipeline harness to Student 4

**Files:**
- Modify: `tests/full_pipeline_harness.py`
- Modify: `tests/test_full_pipeline_integration.py`
- Modify if required: `travel_agent/recommendation/student5/student4_adapter.py`
- Test: `tests/recommendation/test_recommendation_student4_adapter.py`

- [ ] Replace the obsolete scaffold expectation with a failing dynamic handoff test using the
      public Student 4 service and an injected lightweight model/manifest.
- [ ] Update `student4_payload` to match the latest service input and ensure only successful
      `ModuleResult.data` is passed into Student 5.
- [ ] Adapt only genuine schema mismatches between latest Student 4 output and Student 5; preserve
      strict missing-score, stock-evidence, category, provenance, and currency validation.
- [ ] Verify non-ok Student 4 results stop the pipeline and successful results reach Student 5
      Top-3 with compatibility and product scores intact.

### Task 5: Simplify tests by requirements, not branch coverage

**Files:**
- Modify: `tests/recommendation/test_recommendation_*.py`
- Modify: `tests/test_full_pipeline_integration.py`
- Modify: `tests/test_student5_real_scripts.py`
- Do not modify: `tests/test_contracts.py`

- [ ] Inventory the 468 collected Student 5 cases by business/security/integration category.
- [ ] Retain one or a compact parameterized set for every category in the design spec.
- [ ] Remove restored tests whose sole purpose is exhaustive configuration parsing, `.env`
      syntax, equivalent exception subclasses, internal helper branches, repeated serialization,
      repeated immutability, or validation already exercised at the owning boundary.
- [ ] Preserve representative IP literal, DNS-private, redirect-allowlist, redirect-private,
      size, MIME, decoding, timeout, cleanup, and URL-redaction image safety tests.
- [ ] Preserve exact Student 4 schema, hard constraints, ranking, stale-state, confirmation,
      preview gating/consent, and offline module handoff tests.
- [ ] Collect and run the reduced suite after each file group. Expected final collection is
      approximately 280–340; do not delete a required category merely to hit the range.

### Task 6: Verify local and latest-main compatibility

**Files:**
- No new production files

- [ ] Run Student 4 dynamic tests and Student 4→5 adapter/service tests.
- [ ] Run all Student 5 tests, contracts, full-pipeline tests, main-agent tests, and real-data script tests.
- [ ] Run cached `run_student4_real_top3` offline and report the actual current Top-3 without
      assuming older IDs.
- [ ] Run Ruff check, Ruff format check, and `git diff --check` including an explicit untracked-file review.
- [ ] Apply the final in-scope patch to a temporary checkout of current `origin/main` and repeat
      the targeted offline verification.
- [ ] Run the full project suite and report every failure or skip by scope; do not claim full
      success if unrelated baseline failures remain.
- [ ] Report final collection counts, changed files, preserved local files, remaining web
      divergence, live E2E blockers, and Git status. Do not commit or push.
