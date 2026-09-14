# Evaluation commands

Generated predictions and sensitive error-review artifacts belong under the
ignored `evaluation/results/` directory.

Create deterministic grouped error analysis after a complete frozen evaluation:

```bash
.venv/bin/python scripts/evaluation/analyze_errors.py \
  --dataset data/processed/length_views/main-768/test.jsonl \
  --predictions evaluation/results/<run>/predictions.jsonl \
  --expected-dataset-sha256 <frozen-test-sha256>
```

The default destination is a content-addressed subdirectory of
`evaluation/results/`. An explicit `--output-dir` must also be inside that
directory. The command writes aggregate `statistics.json` and `summary.md`,
sensitive `review_samples.jsonl`, and `inspection_manifest.json` with artifact
hashes. Identical reruns reuse their files; differing content is never
overwritten. Incomplete prediction coverage or a dataset hash mismatch is an
error.

Every false negative and false positive is grouped by label and suspected cause;
counts reconcile with the exact evaluator. Causes are heuristics for human
review, not verified explanations. Schema-invalid documents are also reported,
including negative documents whose strict entity FN/FP counts are both zero.
Sampling uses seed 42, up to 20 document nominations per group, and 500 unique
documents overall; override with `--seed`, `--samples-per-group`, and
`--max-review-records`. Selected documents include all their errors and all
group selection reasons. Group counts cover the full dataset regardless of
sampling limits. Logs contain aggregate metadata only.
