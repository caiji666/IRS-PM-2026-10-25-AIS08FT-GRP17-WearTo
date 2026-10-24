# UI Dialogue and Student 5 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the latest intent-dialogue experience into the local UI and connect the complete Student 5 refinement, confirmation, and preview flow.

**Architecture:** Manually reconcile the feature branch with local web changes, then add session-scoped web endpoints that delegate to the existing Student 5 service. The browser renders only current server state and explicitly triggers confirmation and preview.

**Tech Stack:** Python 3.11, standard-library HTTP server, vanilla HTML/CSS/JavaScript, pytest.

**Spec:** `docs/superpowers/specs/2026-10-22-ui-dialogue-student5-integration.md`

## Global Constraints

- Preserve all current uncommitted Student 5, portrait, currency, and diversity changes.
- Do not commit or push.
- Use fixed Student 4 output only as the current test fixture/data source.
- Delegate recommend, modify, confirm, and preview to the real Student 5 service.
- Never expose secrets, local portrait paths, embeddings, provenance, or evidence.

## Review Focus

- Reject an outfit from an earlier recommendation round at confirmation.
- Clear earlier confirmation and preview after requirement changes.
- Show a safe fallback for unavailable preview providers, never a fake generated image.
- Do not let empty optional choices block typed chat.
- Do not show count zero or stale action buttons when Top-3 exists.

---

### Task 1: Integrate intent dialogue

**Files:** `travel_agent/web/`, related understanding/chat modules, and their tests.

- [ ] Add regression tests for floating choices and current local endpoints; run them RED.
- [ ] Reconcile `origin/feat/ui-intent-dialogue` without overwriting local additions.
- [ ] Run dialogue and web tests GREEN.

### Task 2: Add session-scoped Student 5 web actions

**Files:** `travel_agent/web/__main__.py`, a focused Student 5 web helper, and web tests.

- [ ] Add endpoint tests for recommend/current, refinement, confirmation, preview, stale selection, and failure behavior; run RED.
- [ ] Implement minimal session persistence and endpoints; run tests GREEN.

### Task 3: Complete the Student 5 frontend

**Files:** `travel_agent/web/static/index.html`, `app.js`, `style.css`, and static UI tests.

- [ ] Add tests for removal of history UI, dynamic count, action sequencing, and stale-state reset; run RED.
- [ ] Implement UI while preserving product details, seller links, floating choices, and responsive behavior.
- [ ] Run tests GREEN and validate JavaScript syntax.

### Task 4: Fixed-input end-to-end verification

**Files:** Existing tests/scripts and ignored artifacts only as needed.

- [ ] Exercise Top-3 → refinement → new Top-3 → confirm → preview through the web contract.
- [ ] Run focused Student 5, dialogue, profile, and web suites.
- [ ] Run complete pytest and record pre-existing failures separately.
- [ ] Inspect desktop/mobile UI and real preview behavior without exposing credentials.
- [ ] Run Ruff, formatting, JavaScript syntax, and whitespace checks.
