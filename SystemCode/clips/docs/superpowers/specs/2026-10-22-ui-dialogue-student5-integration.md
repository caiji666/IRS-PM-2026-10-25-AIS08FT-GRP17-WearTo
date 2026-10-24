# UI Dialogue and Student 5 Integration Design

## Goal

Integrate `origin/feat/ui-intent-dialogue` into the current local WearTo UI without losing the existing Student 5, portrait, currency, and fixed-input work. Extend the UI so a user can review three final outfits, refine requirements through the existing chat, confirm one current recommendation, and explicitly request a real AI preview.

## User flow

1. The multi-turn dialogue collects the trip request, including optional floating activity/style choices.
2. Student 5 returns the current Top-3. The edit count equals the outfit count; obsolete saved-search controls are absent.
3. Outfit tabs show one outfit at a time, with one product per row and the existing product-detail dialog.
4. Chat refinements call `modify_requirements`, produce a new Top-3, and invalidate earlier selection state.
5. Confirmation calls Student 5 `confirm`; the browser never manufactures `CONFIRMED`.
6. An explicit Generate AI preview click calls `preview_auto`, using the managed portrait when present and styled garments otherwise.

## Architecture

- Preserve caiji666's optional-choice and flexible trip dialogue contract.
- Manually reconcile web files rather than overwriting local portrait and Student 5 additions.
- Persist Student 5 state, recommendation set, confirmed outfit, and preview metadata in the current web job directory.
- Keep fixed Student 4 input as current test data, but route it through session-scoped Student 5 actions.
- Keep preview user-triggered to avoid unexpected API charges.

## Safety

- No commit and no push.
- Never expose API keys, absolute portrait paths, embeddings, provenance, or evidence.
- Confirmation and preview must use the latest recommendation round.
- A failed confirmation must not call preview.
- A requirement change invalidates confirmation and preview from the previous round.
