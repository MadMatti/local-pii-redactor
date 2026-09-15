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

## Main checkpoint selection

The completed selection and full-test evidence are recorded in
[decision 0005](decisions/0005-main-adapter-review.md). Checkpoint 19,000 is
proposed for human approval; the commands below describe the frozen protocol.

Compare the final, 19,000, and 18,000 main checkpoints on the frozen smoke
**validation** split (`data/processed/stages/smoke/valid.jsonl`, 100 records).
Its SHA-256 is
`49f1df7ba0826f913a364aebf155441a7e5e58b574adb212ee4dec59a2d14042`.
Rank the three reports with `compare_evaluations.py` using the approved exact
recall-first order. This is a small validation screen; rare-label results have
limited support. Do not use subsequent test results to switch checkpoints.

Evaluate the selected checkpoint on the original frozen **2,000-record test**.
Then reuse those predictions for the 1,924-record `main-768` view. Training at
768 tokens does not impose that input limit on inference. The filtered view
omits all 76 synthetic long-context test examples, so it is insufficient as the
sole quality report.

Earlier V0 architecture comparisons used nested test subsets. Report that
exposure explicitly and additionally score the 1,500 records outside V0 test:

```bash
python scripts/evaluation/partition_test.py \
  --dataset data/processed/test.jsonl \
  --previously-used-dataset data/processed/stages/v0/test.jsonl \
  --expected-dataset-sha256 eca377c2185c87430bce78ba997789092e186af7c619230dd72b25508ee6b52f \
  --output-dir evaluation/results/v1-test-exposure
```

The two views are disjoint and exhaustive, preserve complete records, and have
checksummed manifests. `not_previously_used.jsonl` means not used in prior
adapter comparisons; it is not a pristine holdout because earlier baselines
were evaluated across the full test. Use `filter_predictions.py` and
`evaluate_predictions.py` to score the same saved generations on each view.

To report the long examples excluded from the length view separately, use:

```bash
python scripts/evaluation/partition_test.py \
  --dataset data/processed/test.jsonl \
  --reference-dataset data/processed/length_views/main-768/test.jsonl \
  --partition-role length_filter \
  --expected-dataset-sha256 eca377c2185c87430bce78ba997789092e186af7c619230dd72b25508ee6b52f \
  --output-dir evaluation/results/v1-test-length-views
```

This produces `retained.jsonl` and `excluded.jsonl`; the latter contains 20
positive and 56 negative long-context examples. The command partitions by
existing IDs and preserves record content; it does not retokenize the data.

## Error review

```bash
python scripts/evaluation/analyze_errors.py \
  --dataset data/processed/test.jsonl \
  --predictions evaluation/results/<selected-run>/predictions.jsonl \
  --expected-dataset-sha256 eca377c2185c87430bce78ba997789092e186af7c619230dd72b25508ee6b52f \
  --output-dir evaluation/results/<selected-run>/error-analysis
```

The report groups every exact false negative and false positive by label and
suspected cause, with counts reconciled against the evaluator. Cause labels
are review heuristics; they do not establish whether a model or annotation is
wrong. It also records invalid-schema documents, including negatives: strict
scoring discards malformed predictions, which can otherwise make a low
negative false-positive rate misleading.

Seed 42, per-group nominations, a global document cap, and input/output hashes
make the local review bundle reproducible. Source text, gold spans, and model
responses appear only in the ignored review JSONL. Inspect that local file
before final adapter approval.

## macOS local-file availability

If iCloud has offloaded project or virtual-environment files, restore them
before running inference. Offloaded bytecode can also delay Python startup.
A temporary cache outside the synced Documents folder can be selected with
`python -X pycache_prefix=/private/tmp/local-pii-redactor-pycache ...`.
This only changes Python's bytecode-cache location; the model, dataset, and
evaluation settings remain the ones recorded in the run configuration.
