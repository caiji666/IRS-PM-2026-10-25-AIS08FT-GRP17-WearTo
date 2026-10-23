# qwen-english-fresh-diagnostic-v2 — observed results

Run: 2026-10-22T08:04:58.855851+00:00

Assistant-authored synthetic gold labels require human review. See the evaluation role and audit provenance below for whether results informed development. This is a small diagnostic set, not an independent deployment accuracy claim.

Evaluation role: `reused_dataset_live_regression`. New model generations on a previously inspected synthetic dataset. Not held out. Original labels and raw outputs are preserved; label review is separate from KG approval.

Model: `Qwen/Qwen3-0.6B`; revision `c1899de289a04d12100db370d81485cdf75e47ca`.

Dataset SHA-256: `c07f38a63674a270c375619766a20cd358d013a043d0ccc1047acda4e5db12cb`.

Raw strict JSON objects: **3/16**. Objects accepted by the code-fence parser: **16/16**.

| Metric | Raw Qwen | After existing guards |
|---|---|---|
| Full proposal correctness | 1/16 | 16/16 |
| Contract-valid output on valid-input cases | 7/15 | 15/15 |
| Follow-up intent classification | 1/7 | 7/7 |
| Populated expected fields | 100/105 | 105/105 |
| Intended changed fields | 0/5 | 5/5 |

Safe rejection of invalid input is reported separately; it is not successful extraction. Invalid-input cases require faithful raw facts and validator rejection. Semantic equivalences are lexical translations/plurals only. Dress-code keys must also exactly reference activity values in the submitted request, matching the actual downstream dictionary lookup. Partial/empty fields must remain empty; extra weather/schema fields and unintended edits fail.

## Per-turn results

| ID | Coverage | Language | Raw | Guarded |
|---|---|---|---|---|
| E01 | new_city_activity | en | FAIL | PASS |
| E02 | new_style | en | FAIL | PASS |
| E03 | decimal_money | en | FAIL | PASS |
| E04 | english_date_words | en | FAIL | PASS |
| E05 | missing_destination | en | FAIL | PASS |
| E06 | dollar_ambiguity | en | FAIL | PASS |
| E07 | relative_date | en | FAIL | PASS |
| E08 | negated_activity | en | PASS | PASS |
| E09 | missing_currency | en | FAIL | PASS |
| E10 | end_date_patch | en | FAIL | PASS |
| E11 | remove_one_exclusion | en | FAIL | PASS |
| E12 | activity_and_code_patch | en | FAIL | PASS |
| E13 | explanation_intent | en | FAIL | PASS |
| E14 | selection_by_id | en | FAIL | PASS |
| E15 | ambiguous_selection | en | FAIL | PASS |
| E16 | cheaper_no_numeric_change | en | FAIL | PASS |

## Human review and next steps

Review all gold labels before reporting formal metrics, especially ambiguous dates, negation, quoted instructions, dress-code scope, resets and confirmation transitions. Inspect raw output and field diffs in the result JSON. Keep this baseline result; prompt/guard changes should produce a new run, not overwrite or adjust labels. Use a fresh held-out set for subsequent prompt tuning evaluation.
