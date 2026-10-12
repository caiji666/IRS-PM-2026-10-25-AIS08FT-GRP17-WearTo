# Qwen diagnostic datasets

The primary English release uses `qwen_english_development_v2.json` (25 previously inspected turns) and `qwen_english_fresh_v2.json` (16 new turns). Their manifests freeze source/dataset fingerprints and identify development versus fresh diagnostic roles. Both have assistant-authored labels pending human review; neither establishes broad production accuracy. See [current evaluation workflow](../README.md).

# Historical real Qwen mini-set v1

This is a small **synthetic diagnostic set** for proposal Stages 1–2 and Appendix A. It contains **32 scenarios / 40 turns**: 29 independent cases and 3 dialogue scenarios (4 + 4 + 3 turns). Language coverage is 25 English, 13 Chinese and 2 mixed-language turns. Gold labels were authored separately from model output and frozen before the first run. They are **assistant-authored and pending human review**, not expert-approved ground truth.

The existing seven smoke examples remain a separate prompt-development set. No new dataset examples were added to the model prompt. The current prompt and production guards were frozen for this baseline; only a raw-generation entry point was extracted so the same model output can be graded before and after guards.

## Coverage

| IDs | Coverage | Expected behaviour |
|---|---|---|
| S01–S03 | Complete English, Chinese and mixed requests | Extract explicit facts, numeric major-unit money, date range and preferences |
| S04 | Three activities and dinner dress code | Preserve all activities; attach dress code only to dinner |
| S05 | Cross-year dates | Preserve 2026 departure and 2027 return |
| S06 | Whole-unit JPY budget | Preserve amount and currency without conversion |
| S07–S08, S25 | Style-only, destination-only and undecided trip | Leave unstated facts null/empty; no destination/activity defaults |
| S09–S10 | Currency omitted and ambiguous dollar symbol | Preserve amount; leave currency unknown |
| S11–S12 | Relative date and month/day without year | Ask for exact dates rather than guessing a year or duration |
| S13–S14 | Negated style/activity | Do not extract a negated preference or excluded activity as positive |
| S15 | Venue/occasion-specific dress code | Keep sightseeing unqualified; retain dinner's explicit code |
| S16 | Activity outside the KG | Keep snorkeling; do not replace it with a supported activity |
| S17 | User-supplied weather speculation | Keep weather outside TripShoppingRequest; do not invent garment preferences |
| S18, S24 | English/Chinese quoted webpage instructions | Extract actual trip facts, ignoring conflicting instructions inside quotations |
| S19 | Two garment exclusions | Preserve both exclusions and the positive style preference |
| S20–S22 | Impossible date, reversed date range, negative budget | Faithful raw facts plus business validation rejection; no silent repair |
| S23 | Fractional JPY amount | Reject unsupported currency precision without silently rounding |
| P01 | Budget and currency edit | Change only budget/currency; preserve dates, style and owned items |
| P02 | Cheaper preference plus exclusion | Keep explicit budget, add lower-price preference and excluded boots |
| P03 | Remove one exclusion | Remove skirt while keeping boots excluded |
| P04 | Explicit style reset | Clear style and preserve all other requirements |
| D01 | Four-turn Chinese clarification/refinement | Retain destination/activity/amount, fill dates, add style, then revise budget |
| D02 | Four-turn English activity/destination edits | Add hiking, cancel dinner/code, change city, remove exclusion; preserve other facts |
| D03 | Explanation → selection → invalid further selection | Recognize explanation; resolve second displayed outfit; preserve confirmed state on rejected further confirmation |

Every turn stores a **complete expected request**, including missing optional fields. Follow-ups include an explicit prior session fixture and complete expected final facts. Dialogue gold trajectories are handwritten declarations; they are not generated using model predictions or the production patch merge.

## Scoring

1. **Strict JSON object rate**: raw object output without code fences or extra prose. A separate parser-accepted rate permits one JSON code fence and a complete thinking block, matching the application parser.
2. **Raw Qwen correctness**: inspect the model response before production guards. Grade full facts, explicit intent, patch validity, expected rejection and selection resolution.
3. **After-guard correctness**: apply the existing evidence guards to a separate copy of the same generated output; grade again. A guard success does not hide a raw-model mistake.
4. **Field results**: report each field, populated expected facts, and intended changed fields separately. Many unchanged/null fields must not hide missed updates.
5. **Missing-information behaviour**: compare required missing fields; amounts/dates/currencies must not be invented.
6. **Unrequested edits**: identify changes to fields the user did not change, including owned-item IDs.
7. **Invalid-input handling**: report safe guard/validator rejection separately. Safe rejection alone is not proof the model extracted the user's facts correctly. Invalid-input cases that produce no surviving guarded candidate do not receive full extraction credit.
8. **Dialogue rollout**: carry actual guarded state between turns, including wrong facts, omissions and rejected edits. Compare final state to the gold trajectory. Teacher-forced scoring separately supplies the intended prior state to isolate individual-turn extraction errors.

List order does not matter. Registered lexical translations/plurals can be equivalent; no date repair, numeric-string conversion, inferred activities or dropped extra fields are allowed for grading. Equivalent valid patches can use set/add/remove differently if they produce the same final facts. Old outfit IDs and disagreement between position and ID must not silently select another outfit.

This benchmark evaluates **proposed extraction/control behaviour**. It simulates acceptance of valid proposals to measure state evolution; the real application still requires human extraction review. It does not call weather providers, Neo4j, product APIs or recommendation models, and does not write real user sessions. Displayed outfit IDs are explicit synthetic fixtures, not generated recommendations.

## Run

From the repository root, install the optional runtime:

```bash
uv sync --locked --extra qwen
HF_HOME="$PWD/artifacts/huggingface" uv run --locked --extra qwen \
  python -m evaluation.qwen_benchmark --rollout \
  --output artifacts/qwen-understanding-run.json
```

`--model-path` can point to a previously downloaded local Qwen snapshot. The default fixed model revision is recorded by the extractor. `--limit` runs a debug subset and must not be presented as the full baseline. Progress is saved after each teacher-forced turn and each completed rollout dialogue.

Files:

- `qwen_understanding_v1.json`: synthetic messages, complete gold facts, session fixtures and expected outcomes.
- `qwen_understanding_v1.manifest.json`: frozen dataset/extractor SHA-256 fingerprints.
- `../qwen_benchmark.py`: actual local-model runner and separate scoring layers.
- `../results/qwen_understanding_v1_baseline.json`: raw outputs, field diffs, timings, model revision, per-input prompt hashes and state trajectories.
- `../results/qwen_understanding_v1_baseline.md`: original readable generation/scoring report.
- `../results/qwen_understanding_v1_audited.json` / `.md`: final score audit of those same raw generations. The original baseline stays intact. The audit additionally checks exact dress-code/activity dictionary joins and service-compatible no-op confirmation state; it makes no model calls and changes no prompts/gold labels.

Weights stay in ignored local storage. Tests of the scoring harness use fixed synthetic responses and are separate from actual model runs.

## Human review before formal reporting

Review labels for currency ambiguity, relative dates, negation, quoted instructions, dress-code scope, list reset semantics, currency minor units and confirmation state transitions. Confirm that lexical equivalences are adequate and do not merge different concepts (e.g. skirt versus dress). Pending gold review should be disclosed with all results.

Keep this frozen baseline. If model prompts or guards are improved after inspecting failures, save a new run and retain these results. Because this set has now been inspected, create a new held-out set before claiming improvements generalize. The sample is too small for broad accuracy or deployment claims.

## Observed first run

The first full run made 40 teacher-forced generations plus 11 dialogue-rollout generations. Strict JSON objects: 30/40; all 40 outputs were accepted by the code-fence parser. Final audited full-correctness scores are **3/40 raw** and **14/40 after existing guards**. None of the three rollouts reached the complete gold final state.

The strict score includes all expected facts and null/empty fields, unwanted additions, intent/patch validity, expected validator outcome, selection, and actual activity/dress-code key association. It is not the same as average field accuracy. Four invalid-input guard rejections are recorded separately and do not receive full extraction credit simply for being rejected.

Failures include invented minimalist preferences, unrequested dress codes, omitted dates, negation mistakes, quoted-instruction contamination, and full-request-shaped responses where a patch is required. Chinese activity normalization can leave a Chinese dress-code key alongside an English activity value; the current service then cannot look it up. This was an integration finding in the historical run. The English release subsequently corrected canonical dress-code joins while preserving these original results.

Raw initial-request instruction hashes agree with the earlier seven-case smoke runner, confirming that this broader run did not tune the model prompt. Keep human extraction review enabled while these issues remain.
