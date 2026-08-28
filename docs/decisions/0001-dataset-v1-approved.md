# Decision 0001 — Dataset V1 approved

- **Status:** Approved
- **Date:** 2026-08-28
- **Owner:** Mattia Evangelisti
- **Dataset build:** `prepared-dataset-v1`
- **Source revision:** `e06eb1499ca8d54470f085021cd8e54f9efac7fd`
- **Policy version:** `pii-taxonomy-v1`
- **Prompt version:** `pii-extraction-v1`

## Decision

The prepared dataset is approved for evaluator development, baseline execution,
and staged QLoRA training. The approved policy retains the 14 canonical labels,
preserves exact source substrings, orders occurrences by source character
position, keeps official Gretel split boundaries, and targets 25% negative
records in prepared stages.

The 13 source records listed in `data/processed/rejected_source.jsonl` are
approved for exclusion. They contain duplicate single-occurrence annotations or
overlapping person-name annotations and must not be silently repaired.

The reviewed synthetic additions—including simple negatives, hard negatives,
long contexts, repeated occurrences, and rare-label templates—are approved.
Credential/security source labels remain excluded from V1.

## Evidence

- Source inspection bundle for the pinned revision.
- Prepared dataset manifest and artifact hashes.
- Synthetic review bundle containing 268 reason-tagged samples.
- Strict validation report: zero errors and one documented source-reject warning.
- Offline unit suite: 39 passed and one opt-in network test deselected.
- Byte-identical repeated builds.

## Consequences

- Gate G3 is open and model training may begin.
- The prepared test split and canonical prompt are frozen for baseline and
  adapter comparisons.
- Any later mapping, prompt, split, source-repair, or synthetic-template change
  creates a new dataset version and requires a new approval record.
- Model selection still requires evaluation evidence; dataset approval alone is
  not approval of a final adapter or quantization.
