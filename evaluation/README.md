# Evaluation

System-wide fixed-workflow versus agent evaluation remains unmeasured. Student 1 extraction diagnostics below do not test product retrieval or completed recommendations.

## English-first diagnostics

- `datasets/qwen_english_development_v2.json`: 25 previously inspected English turns from v1; development regression, not held-out evidence. Actual run includes seven additional rollout calls.
- `datasets/qwen_english_fresh_v2.json`: 16 newly authored English diagnostic turns, frozen before the first model run. Small synthetic set, with labels pending human review.
- Corresponding result JSON/Markdown in `results/` preserve raw Qwen generations, augmented correctness, field errors, timing and model/prompt fingerprints.

```bash
uv sync --locked --extra qwen
HF_HOME="$PWD/artifacts/huggingface" uv run --locked --extra qwen \
  python -m evaluation.qwen_benchmark \
  --dataset evaluation/datasets/qwen_english_fresh_v2.json \
  --output artifacts/english-evaluation.json
```

Use `--model-path` for a downloaded local snapshot; weights are ignored. Use `--rollout` for the development dataset's two conversations. An augmented result measures Qwen plus source-evidence rules, never Qwen alone. Safe validation rejection is recorded separately from extraction correctness. Benchmarks simulate acceptance to test state evolution; production still requests human review.

## Historical multilingual baseline

The original 40-turn set and raw/audited results are preserved. Its audited scores were 3/40 raw and 14/40 guarded, with no correct final rollout state. These historical results precede the English release changes. See [dataset documentation](datasets/README.md). Inspecting these failures informed development, so improvements on that set cannot be described as held-out performance.

## Observed English release results

The development run made 25 teacher-forced plus seven actual rollout calls: 4/25 raw, 21/25 augmented, and both English dialogues reached their expected final state. The first fresh run made 16 actual calls: 1/16 raw and 15/16 augmented. Its style-only case exposed a destination leak. The original outputs are preserved.

After correcting source-evidence scope for style-only/undecided requests, saved-output regression audits score 22/25 development and 16/16 fresh augmented proposals. These are **post-inspection audits with zero new model calls**, not new independent test scores. Three invalid-business-input cases in development are safely rejected but receive no full extraction credit. Raw model accuracy remains poor; human extraction review remains required.
