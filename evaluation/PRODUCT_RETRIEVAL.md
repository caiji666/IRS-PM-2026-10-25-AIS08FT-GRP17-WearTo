# Product Retrieval Evaluation (Local)

This evaluates saved **Production** eBay Browse retrieval runs without making a new API request. It reuses each run's indexed product images and the same context and social FashionCLIP vectors to compare three ranking arms: context only (1/0), reference only (0/1), and the deployed hybrid (0.5/0.5). The hybrid Top-K IDs and scores must reproduce the original output exactly or the evaluator stops.

## Pilot run

From the repository root, with the project's `.venv` installed:

```bash
.venv/bin/python -m evaluation.product_retrieval_eval \
  runs/product_retrieval_diagnostics/20261022_123414_217382+0800/run_report.json \
  runs/product_retrieval_diagnostics/20261022_115329_724977+0800/run_report.json \
  runs/product_retrieval_diagnostics/20261022_124252_044173+0800/run_report.json \
  --output-dir runs/product_retrieval_evaluation/my_pilot
```

Those three reports are saved local runs for Seoul, Tokyo, and Beijing. They are not in Git. Substitute other `run_report.json` files if needed; use distinct trip scenarios rather than treating repeated runs of the same trip as independent cases. Each output directory must be new.

`metrics.json` records source hashes, the candidate funnel, category coverage, field completeness, and Top-K overlaps between arms. It also records `human_evaluation.status = pending_labels` until judged labels are supplied. Overlap and similarity scores are **not** measures of relevance. Price/image/link completeness means the fields are present; it does not prove a listing remains live. The `verified_stock_fraction` is evidence from the saved API snapshot, not a checkout guarantee. `*_top_k.json` contains only product IDs and scores for each arm. `case_briefings.json` gives trip and social-reference context for judges.

## Human relevance study

Use `labels_template.csv` to obtain independent, blinded judgments: its candidate rows are shuffled and omit arm/score/rank. Have two reviewers each copy the template and fill `true_category` (`TOP`, `BOTTOM`, `FOOTWEAR`, `OUTERWEAR`, or `UNKNOWN`) and `relevance_0_to_3` for **every** row, using product image/link and `case_briefings.json`. Grade 0 = irrelevant, 1 = weak, 2 = relevant, 3 = highly relevant. Reviewers should not inspect the arm output files while labeling. Resolve disagreements into one completed, adjudicated CSV, preserving the template's IDs and columns.

Run again with the adjudicated file:

```bash
.venv/bin/python -m evaluation.product_retrieval_eval \
  runs/product_retrieval_diagnostics/20261022_123414_217382+0800/run_report.json \
  runs/product_retrieval_diagnostics/20261022_115329_724977+0800/run_report.json \
  runs/product_retrieval_diagnostics/20261022_124252_044173+0800/run_report.json \
  --labels /path/to/adjudicated_labels.csv \
  --output-dir runs/product_retrieval_evaluation/my_judged_pilot
```

This computes Precision@5 (grades 2–3 in the correct category divided by five), Recall@5 within the **saved indexed pool** (relevant returned divided by all relevant indexed candidates), NDCG@5, mean relevance, and category accuracy/macro-F1. Recall is not recall over the whole eBay marketplace; only downloaded and encoded products can be judged. A category with no relevant indexed item has undefined recall/NDCG and is omitted from those macro means. Category accuracy compares the final assigned category with the human label; it does not compare metadata with a separate FashionCLIP zero-shot classifier.

For a stronger final study, collect varied new trip scenarios and judge all indexed candidates. The three-case pilot checks implementation and produces operational baselines, but cannot establish broad recommendation quality by itself. The `runs/` folder is git-ignored, so reports and labels stay local unless explicitly exported.
