# qwen-understanding-mini-v1 — observed results

Run: 2026-10-22T08:11:11.837039+00:00

Assistant-authored synthetic gold labels require human review. See the evaluation role and audit provenance below for whether results informed development. This is a small diagnostic set, not an independent deployment accuracy claim.

Evaluation role: `reused_dataset_live_regression`. New model generations on a previously inspected synthetic dataset. Not held out. Original labels and raw outputs are preserved; label review is separate from KG approval.

Model: `Qwen/Qwen3-0.6B`; revision `c1899de289a04d12100db370d81485cdf75e47ca`.

Dataset SHA-256: `d5d735bd0b9a2297eb1019b695c05ef61ea76f11313a49032f015d619b448b80`.

Raw strict JSON objects: **8/40**. Objects accepted by the code-fence parser: **40/40**.

| Metric | Raw Qwen | After existing guards |
|---|---|---|
| Full proposal correctness | 4/40 | 23/40 |
| Contract-valid output on valid-input cases | 21/35 | 29/35 |
| Follow-up intent classification | 1/13 | 8/13 |
| Populated expected fields | 228/253 | 211/253 |
| Intended changed fields | 0/15 | 8/15 |

Safe rejection of invalid input is reported separately; it is not successful extraction. Invalid-input cases require faithful raw facts and validator rejection. Semantic equivalences are lexical translations/plurals only. Dress-code keys must also exactly reference activity values in the submitted request, matching the actual downstream dictionary lookup. Partial/empty fields must remain empty; extra weather/schema fields and unintended edits fail.

## Per-turn results

| ID | Coverage | Language | Raw | Guarded |
|---|---|---|---|---|
| S01 | complete_request | en | FAIL | PASS |
| S02 | complete_request | zh | FAIL | FAIL |
| S03 | complete_request | mixed | FAIL | PASS |
| S04 | multiple_activities | en | FAIL | PASS |
| S05 | date_year_boundary | en | FAIL | PASS |
| S06 | currency_precision | en | PASS | PASS |
| S07 | missing_information | en | FAIL | PASS |
| S08 | missing_information | zh | FAIL | FAIL |
| S09 | ambiguous_currency | en | FAIL | PASS |
| S10 | ambiguous_currency | en | FAIL | PASS |
| S11 | ambiguous_date | zh | FAIL | FAIL |
| S12 | ambiguous_date | zh | FAIL | FAIL |
| S13 | negation | en | FAIL | PASS |
| S14 | negation | zh | FAIL | FAIL |
| S15 | dress_code_scope | en | FAIL | PASS |
| S16 | unsupported_activity | en | FAIL | PASS |
| S17 | weather_separation | en | FAIL | PASS |
| S18 | quoted_instruction | en | FAIL | PASS |
| S19 | exclusions | mixed | FAIL | FAIL |
| S20 | invalid_input | en | FAIL | FAIL |
| S21 | invalid_input | zh | FAIL | FAIL |
| S22 | invalid_input | en | PASS | FAIL |
| S23 | currency_precision | en | PASS | FAIL |
| S24 | quoted_instruction | zh | FAIL | FAIL |
| S25 | missing_information | en | FAIL | PASS |
| P01 | scalar_patch | en | FAIL | PASS |
| P02 | preference_patch | zh | FAIL | FAIL |
| P03 | list_patch | en | FAIL | PASS |
| P04 | reset_patch | zh | FAIL | FAIL |
| D01_T1 | dialogue_missing | zh | FAIL | FAIL |
| D01_T2 | dialogue_missing | zh | FAIL | FAIL |
| D01_T3 | dialogue_missing | zh | FAIL | FAIL |
| D01_T4 | dialogue_missing | zh | FAIL | FAIL |
| D02_T1 | dialogue_edits | en | PASS | PASS |
| D02_T2 | dialogue_edits | en | FAIL | PASS |
| D02_T3 | dialogue_edits | en | FAIL | PASS |
| D02_T4 | dialogue_edits | en | FAIL | PASS |
| D03_T1 | dialogue_confirmation | en | FAIL | PASS |
| D03_T2 | dialogue_confirmation | en | FAIL | PASS |
| D03_T3 | dialogue_confirmation | en | FAIL | PASS |

## Dialogue rollout

Teacher-forced turns receive the gold previous state, isolating each extraction. Rollout carries the actual guarded state, including mistakes and failed edits. This simulates proposals without writing any real user session.

- D01: 0/4 turns; final gold-state match: False.
- D02: 4/4 turns; final gold-state match: True.
- D03: 3/3 turns; final gold-state match: True.

## Human review and next steps

Review all gold labels before reporting formal metrics, especially ambiguous dates, negation, quoted instructions, dress-code scope, resets and confirmation transitions. Inspect raw output and field diffs in the result JSON. Keep this baseline result; prompt/guard changes should produce a new run, not overwrite or adjust labels. Use a fresh held-out set for subsequent prompt tuning evaluation.
