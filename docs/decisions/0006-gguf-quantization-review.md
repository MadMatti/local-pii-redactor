# Decision 0006 — GGUF quantization trials require review

Status: approved trials completed; Q8 passes validation, smaller formats rejected

Date: 2026-09-18

Reverified: 2026-09-24. All four comparisons reproduce exactly from unchanged
local artifacts, and all 154 offline tests pass. Power was rechecked at 100%
charge on AC; no additional generation or deployment was started.

## Outcome

Q8_0 is the only quantized artifact that passes the approved frozen validation
gates. All 100 Q8 output strings are byte-identical to the BF16 GGUF reference.
Q5_K_M, ordinary Q4_K_M, and importance-matrix Q4_K_M fail all three gates:
exact recall, complete-document recall, and schema validity.

Do not promote a failed format, relax a threshold, or transfer a candidate to
the Pi on the strength of file creation. Q8 remains a passing quality reference,
not an approved deployment selection. No quantized full-test evaluation or Pi
benchmark has been performed. Gate G7 remains open for candidate review.

## Controlled comparison

The immutable H-045 plan remains unchanged. Every candidate was quantized from
the same verified BF16 GGUF, never from another quantized model. The pinned
llama.cpp revision, exact prompt-token sequences, tokenizer, context, decoding,
output budget, and evaluator are shared by all runs.

The set contains 100 frozen validation records: 75 positive, 25 negative, and
246 gold entities across 14 labels. These are not test-set scores. The permitted
losses relative to BF16 are at most **one percentage point** in exact recall and
complete-document recall, with no schema-validity decline.

| Artifact | Size, decimal GB | Exact recall | Complete-doc recall | F1 | Schema valid | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| BF16 reference | 3.447 | 87.80% | 73.33% | 87.63% | 100% | Reference |
| Q8_0 | 1.834 | 87.80% | 73.33% | 87.63% | 100% | Pass |
| Q5_K_M | 1.258 | 84.55% | 69.33% | 86.13% | 99% | Fail |
| Q4_K_M | 1.107 | 78.05% | 61.33% | 77.26% | 99% | Fail |
| Q4_K_M + imatrix | 1.107 | 78.86% | 60.00% | 80.50% | 98% | Fail |

The commit-safe, machine-readable evidence is
[`quantization-v1.json`](../../evaluation/baselines/quantization-v1.json).
It includes checksums, exact counts, per-label scores/deltas, gate outcomes,
changed-output counts, calibration provenance, and limited timing diagnostics.
The report is recomputed from frozen predictions, rather than copying previously
reported aggregate scores.

## What the regressions show

- **Q8:** 216 TP, 31 FP, 30 FN; zero changed outputs versus BF16 on this set.
  This is not a claim of numerical equivalence on every possible input.
- **Q5:** 208 TP, 29 FP, 38 FN; 17 changed outputs. Fewer false positives do
  not compensate for the eight additional misses under the recall-first gate.
- **Ordinary Q4:** 192 TP, 59 FP, 54 FN; 45 changed outputs. The largest net
  losses are seven correct PERSON_NAME and seven correct MEDICAL_ID occurrences.
  Local error analysis groups name-boundary errors, missed entities, and wrong
  labels; those groups are diagnostic hypotheses, not proven numerical causes.
- **Calibrated Q4:** 194 TP, 42 FP, 52 FN; 39 changed outputs. Calibration
  improves F1 over ordinary Q4, but complete-document recall and schema validity
  are worse. It is not an acceptable replacement for the BF16/Q8 reference.

Q5 and ordinary Q4 each produce one response containing an unsupported entity
type; calibrated Q4 produces two. All those responses end normally at EOS,
not at the output-token cap. They remain invalid under the unchanged strict
parser. No JSON repair, label remapping, output-cap change, or target correction
was introduced to improve the scores.

All candidates pass native prompt/template and EOS checks. The evidence does
not indicate a detected prompt-format mismatch. It establishes observed
quantization regressions, not a complete explanation of their numerical cause.

## Calibration and provenance

The approved 500-document corpus reproduces byte-for-byte from the train-only
generator. Exact and normalized overlaps are 287 training records and zero
validation/test records. No held-out document was used for calibration.

The native run processes all 55 complete 1,024-token chunks (56,320 tokens),
leaving any final partial chunk unused. All 196 expected dense layer weights
have finite, nonnegative activation statistics and complete counts in 392 F32
tensors. The output embedding is not calibrated. This is a narrow synthetic
corpus; it does not establish optimal calibration representativeness.

An inspector serialization correction and its startup/actual verification
hashes are documented in [the packaging guide](../model-packaging.md). Original
model files, manifests, and raw predictions remain preserved locally.

## Local evidence and reproduction

Model artifacts and per-run manifests are under:

```text
models/gguf/v1-ckpt19000-bf16/
models/gguf/v1-ckpt19000-q8_0/
models/gguf/v1-ckpt19000-q5_k_m/
models/gguf/v1-ckpt19000-q4_k_m/
models/gguf/v1-ckpt19000-imatrix/
models/gguf/v1-ckpt19000-q4_k_m-imatrix/
```

Generation reports use `evaluation/results/v1-gguf-<format>-smoke-valid/`.
Their sibling `-parity/` directories contain independently recomputed gate
reports. Failed-format sibling `-errors/` directories contain local review
bundles: 29 Q5, 44 Q4, and 40 calibrated-Q4 documents. Do not commit or publish
those raw samples or predictions.

The wrappers, required checks, and native generation command are documented in
[the packaging guide](../model-packaging.md). To recompute the aggregate without
overwriting existing evidence:

```bash
.venv/bin/python scripts/model/report_quantization.py \
  --output evaluation/baselines/quantization-v1-recheck.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -m "not network and not mlx"
.venv/bin/python scripts/check_repository.py
```

All 154 offline tests pass; the network integration test is excluded. Reports
with failed quality gates use exit 2 intentionally; every native generation and
artifact-creation subprocess completed successfully. No failure was concealed
by changing the evaluator or tolerances.

## Timing limitations

These trials were not controlled performance benchmarks. Four Q8 records have
native phase durations exceeding client request durations by more than one
second, including two large discrepancies. The cause is unconfirmed; different
clock behavior around suspension is one possibility. Q8 prompt/decode rates
are therefore suppressed, and the report records the anomaly count. Original
native timing remains in local predictions. The initial aggregate is preserved
at `evaluation/results/v1-gguf-quantization-initial-summary.json`.

Other throughput numbers are local Mac diagnostics only. Peak native RAM was
not measured. No Pi speed, temperature, power, or memory claims are justified.

## Next review gate

1. Review these results and the failed-format local error bundles. Do not
   nominate Q4 or Q5 for deployment under the existing tolerance.
2. Decide whether to advance the larger passing Q8 artifact to the full frozen
   2,000-record test, or propose a separately approved compact-model experiment.
   A Q8 test would assess generalization, not automatically approve Pi deployment.
3. Confirm stable external power before another long run. At final-trial startup
   the Mac reported AC attached but a discharging battery at 22%.
4. Any changed calibration corpus, mixed-precision recipe, training, dataset,
   decoding, or tolerance needs its own approved/versioned experiment. Do not
   reuse frozen test results for quantization selection.
5. Pi transfer, hardware benchmarking, and deployment remain separate gates,
   including H-046. No files have been transferred and no service deployed.
