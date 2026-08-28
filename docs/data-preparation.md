# Dataset acquisition, inspection, and preparation

This project pins, inspects, and deterministically converts the Gretel PII
masking dataset. Full source snapshots, review artifacts, prepared JSONL, and
calibration text remain outside Git.

## Environment

The data pipeline uses Python 3.11 and does not import MLX or require Metal or
PyTorch.

```bash
source .venv/bin/activate
python -m pip install -e ".[data,dev]"
```

## Download and pin the source

```bash
python scripts/data/download_dataset.py
```

The downloader resolves `main` to an immutable Hugging Face commit, validates
the documented columns and 50k/5k/5k split counts, saves an explicit local
snapshot, and writes:

```text
data/intermediate/gretel-pii-masking-en-v1/source-lock.json
```

Rerun without network access after a successful download:

```bash
python scripts/data/download_dataset.py --offline
```

The command refuses to replace a locked revision silently. Use
`--refresh-revision` only when deliberately evaluating a new upstream revision;
the new revision is stored in a separate SHA-named directory.

## Inspect the source

```bash
python scripts/data/inspect_dataset.py
```

The inspector uses the local Qwen tokenizer and writes a review bundle under:

```text
data/intermediate/gretel-pii-masking-en-v1/<revision>/inspection/
```

The bundle contains:

- `summary.md`: concise human-readable results;
- `statistics.json`: split and overall distributions;
- `label_inventory.csv`: every source label and its candidate policy;
- `quality_issues.jsonl`: structural blockers and warnings;
- `split_overlap.json`: UID, text, normalized-text, and template overlap;
- `review_samples.jsonl`: deterministic records selected for manual review;
- `inspection_manifest.json`: versions, hashes, and artifact sizes.

Exit codes are:

- `0`: inspection completed without structural blockers;
- `2`: reports were written, but human review is required for blockers;
- `1`: the command could not complete.

Human approval is required even when the exit code is zero.

## Current pinned inspection

- Revision: `e06eb1499ca8d54470f085021cd8e54f9efac7fd`
- Records: 60,000
- Parsed source entities: 254,706
- Structural blockers: 0
- Source-annotation order warnings: 33,385
- Review blockers: 0
- Candidate-policy negatives: 1,618 records (2.697%)
- Records mixing included and excluded source labels: 21,742
- Exact or normalized text duplicates across splits: 0
- Cross-split description/template groups: 68
- Deterministic review samples: 969

Review the ignored `review_samples.jsonl` and `split_overlap.json` before
approving conversion to MLX chat JSONL. In particular, inspect `name` spans that
contain titles or suffixes, address boundaries, excluded generic labels, records
that become negative, source annotations that need sorting into text order, and
repeated template descriptions.

## Build the prepared datasets

After approving the candidate policy, run:

```bash
python scripts/data/build_dataset.py
```

The builder preserves the official Gretel train/validation/test boundaries,
filters excluded labels, anchors every retained entity to exact source
character offsets, orders outputs by source position, and writes compact
MLX-LM chat records. It also adds deterministic split-isolated synthetic
examples for negatives, hard negatives, long contexts, repeated entities, and
rare-label coverage.

Prepared outputs are:

```text
data/processed/train.jsonl                 20,000 records
data/processed/valid.jsonl                  2,000 records
data/processed/test.jsonl                   2,000 records
data/processed/stages/v0/                   5,000/500/500
data/processed/stages/smoke/                  500/100/100
data/processed/full_source/                all exactly convertible source rows
data/processed/rejected_source.jsonl       source rows rejected without repair
data/processed/synthetic_review_samples.jsonl  reason-tagged human-review set
data/processed/manifest.json               config, counts, and hashes
data/calibration/pii_calibration.txt       separate 500-document corpus
```

Smoke is a strict subset of V0, and V0 is a strict subset of the main candidate.
Every prepared stage has a 25% negative ratio and includes all 14 labels in all
three splits.

Exact conversion identified 13 train rows with duplicate single-occurrence
annotations or overlapping person-name annotations. They are recorded in the
reject ledger and are not silently repaired or included in prepared data.

## Validate prepared data

```bash
python scripts/data/validate_dataset.py
```

The validator checks JSONL and chat-message shape, assistant JSON, canonical
labels, exact span membership, source order, negative flags, sample-ID
uniqueness, exact and normalized split leakage, synthetic template isolation,
stage nesting, counts, distributions, artifact hashes, and local-Qwen token
lengths. It writes:

```text
data/processed/validation/validation_report.md
data/processed/validation/statistics.json
```

Exit codes are `0` for pass, `2` for completed validation with dataset errors,
and `1` when validation cannot run. The current build passes with zero errors
and one warning for the 13 intentionally rejected source defects.

## Tests

Run the offline suite:

```bash
pytest -m "not network and not mlx"
```

Run the opt-in live schema test:

```bash
RUN_NETWORK_TESTS=1 pytest -m network tests/integration/test_gretel_dataset.py
```
