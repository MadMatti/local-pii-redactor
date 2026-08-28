# Evaluation workflow

Dataset V1 uses exact occurrence-level evaluation. A prediction is correct only
when its canonical type and exact source character occurrence match the frozen
ground truth. Duplicate values are treated as distinct occurrences.

The evaluator also reports strict JSON/schema validity, hallucinated substrings,
source-order validity, per-class precision/recall/F1, complete-document recall,
exact-document match, and PII-free false-positive rate. It never repairs model
output before strict scoring.

## Frozen test set

The frozen definition is `evaluation/baselines/frozen-test-v1.json`. Commands
should pass its SHA-256 value with `--expected-dataset-sha256` so an accidental
dataset rebuild cannot silently change comparisons.

## Evaluator self-check

```bash
python scripts/evaluation/run_oracle_check.py \
  --expected-dataset-sha256 eca377c2185c87430bce78ba997789092e186af7c619230dd72b25508ee6b52f
```

Every strict oracle metric must equal one.

## Regex baseline

```bash
python scripts/evaluation/run_regex_baseline.py \
  --expected-dataset-sha256 eca377c2185c87430bce78ba997789092e186af7c619230dd72b25508ee6b52f
```

The baseline is intentionally conservative and cannot recognize unconstrained
person names. It exists to establish a deterministic non-model reference.

## MLX model or adapter

```bash
python scripts/evaluation/run_model_baseline.py \
  --output-dir evaluation/results/base-qwen3-v1 \
  --expected-dataset-sha256 eca377c2185c87430bce78ba997789092e186af7c619230dd72b25508ee6b52f
```

Add `--adapter models/adapters/<run>` to evaluate an adapter. Generation uses
temperature zero, thinking disabled, the frozen system prompt, and resumable
content-free progress logs. Pass `--resume` after an interrupted run.

Generated predictions and reports remain local under `evaluation/results/`.
