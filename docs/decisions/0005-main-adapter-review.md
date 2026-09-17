# Decision 0005 — Main adapter selected for human review

Status: approved for fusion/parity and quantization trials (`H-045`)

Date: 2026-09-15

## Outcome

Approve checkpoint **19,000** from the completed main experiment for the next
fusion/parity stage. It was selected using frozen **validation** task metrics,
before the selected weights were evaluated on the full 2,000-record test.

The full test has 0.8882 exact recall, 0.8898 exact F1, and 0.7520
complete-document recall. This materially improves on the same-test base and
regex baselines, satisfying the baseline-relative improvement condition.
It is not a deployment-safety claim: 575 annotated PII occurrences were missed,
372 of 1,500 PII-containing documents had at least one exact miss, and four
responses were invalid.

On 2026-09-15 the user explicitly approved this checkpoint in response to the
request to proceed with fusion and quantization trials. This authorizes local
fusion, parity evaluation, pinned GGUF tooling, conversion, and quantization
experiments under the existing quality gates. It does not authorize deployment,
new training, dataset corrections, or changes to the frozen evaluator.

## Training audit

The user-run main experiment completed successfully on 2026-09-02 at
04:46:54 UTC, with return code 0 and finite losses.

| Item | Recorded result |
| --- | --- |
| Architecture | Qwen3-1.7B 4-bit base; 16 LoRA layers, rank 8, scale 20, dropout 0 |
| Data | 19,255 train / 1,997 validation / 1,924 test complete-record 768-token view |
| Training | Batch 1, accumulation 8, learning rate 0.00001, seed 42 |
| Memory controls | Prompt masking and gradient checkpointing enabled |
| Microbatch iterations | 19,255 |
| Duration | 40,359.679 seconds: 11h 12m 40s |
| Peak MLX allocation | 2.485 GB; not total system memory |
| Final logged losses | Train 0.021; validation 0.015 |
| Lowest logged validation loss | 0.014 at iteration 19,000; validation sampled 200 batches |
| Training Git commit | `8b5b48699867b1c11da3e9143cb8b27d8abd85bd` |

The manifest is `training/runs/v1-selected-l16-r8-768/manifest.json`.
Configuration, dataset, log, and adapter hashes were checked against the saved
run. Original intermediate and final weights were preserved. Candidate
directories contain checksum-verified copies with a matching adapter config;
the original experiment directory was not repurposed.

Accounting caveat from the inspected MLX-LM trainer: iteration means a
microbatch, not an optimizer update. With accumulation 8, this run made 2,406
updates; its last seven microbatches were processed but their residual
gradients were not applied before the final save. The final weights reflect
the update at iteration 19,248. Checkpoint 19,000 falls on an update boundary.

Validation loss is logged before that iteration's training/update, while the
checkpoint is saved afterward. Consequently the logged loss at 19,000 is not
an exact measurement of the saved 19,000 checkpoint. Task-based evaluation of
the actual saved weights is the selection evidence; this accounting detail
does not invalidate their measured results.

## Validation-only checkpoint selection

The fixed candidates were final (19,255), 19,000, and 18,000. Selection used
100 frozen smoke-validation records: 75 positive, 25 negative, and 246 gold
entities across all 14 labels. Their IDs are in the main validation set and
do not intersect the main test.

The approved lexicographic order is exact recall, complete-document recall,
precision, schema validity, F1, then lower negative false-positive rate.

| Candidate | Recall | Complete-doc recall | Precision | F1 | Schema valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| ckpt19000 | 0.8740 | 0.7200 | 0.8776 | 0.8758 | 1.0000 |
| ckpt18000 | 0.8699 | 0.7333 | 0.8917 | 0.8807 | 0.9900 |
| final | 0.8699 | 0.7200 | 0.9030 | 0.8861 | 1.0000 |

Checkpoint 19,000 finds 215 of 246 exact entities; each alternative finds 214.
Its 30 false positives exceed the final checkpoint's 23, and the final has
higher precision and F1. The recall-first rule nevertheless selects 19,000.
This one-entity margin on 100 records is not statistically established
superiority. Rare labels have little support, including four BANK_ACCOUNT
and five USERNAME occurrences.

The selection protocol is
[`main-checkpoint-selection-v1.json`](../../evaluation/baselines/main-checkpoint-selection-v1.json).
The local comparison is
[`comparison.md`](../../evaluation/results/v1-validation-selection/comparison.md).
Only the selected weights were taken through this main full-test evaluation;
no checkpoint was switched in response to test results.

## Full frozen test

All 2,000 records were scored: 1,500 positives, 500 negatives, and 5,143 gold
entity occurrences. A true positive requires the correct canonical type and
exact source occurrence. A malformed response is discarded, not repaired.

| Model | Recall | Complete-doc recall | Precision | F1 | Schema valid | Negative-document FP rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| selected-19000 | 0.8882 | 0.7520 | 0.8913 | 0.8898 | 0.9980 | 0.0080 |
| base-qwen3 | 0.3506 | 0.1813 | 0.5226 | 0.4196 | 0.6405 | 0.6580 |
| regex | 0.3265 | 0.1360 | 0.5144 | 0.3994 | 1.0000 | 0.1340 |

Selected-checkpoint totals: 4,568 TP, 575 FN, and 557 FP. Complete-document
recall is 1,128/1,500; exact-document match, which additionally requires no
false positives and valid schema, is 1,517/2,000. Four of the 500 negative
documents have false positives; none of the negatives has invalid schema.

There are four invalid responses, all on positive documents, accounting for
12 false negatives under strict scoring. Three reached the recorded
384-output-token cap. The cap and parser were not changed to improve this
test score. A future decoding/cap experiment must be separately versioned and
selected on validation data.

Relaxed overlap F1 is 0.9242, diagnostic only. It does not establish that
partial or wrong-boundary redaction is safe. Source-order validity is 0.9865.
The evaluator reports 12 unalignable predicted substrings: error analysis
distinguishes eight absent-from-source values from four overpredicted
occurrences of values already present.

## Diagnostic test views

These reuse the same selected-model, base, and regex predictions; there were
no additional model generations or checkpoint choices for these views.

| View | Records | Selected recall | Selected complete-doc recall | Selected F1 | Base F1 | Regex F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained 768-token view | 1924 | 0.8880 | 0.7493 | 0.8894 | 0.4186 | 0.3973 |
| Outside prior V0 adapter comparisons | 1500 | 0.8861 | 0.7476 | 0.8885 | 0.4261 | 0.4013 |
| Excluded synthetic long examples | 76 | 0.9500 | 0.9500 | 0.9744 | 0.7097 | 0.8500 |

The full frozen test is the primary result. The retained view alone omits all
76 synthetic long-context examples. Those 76 include only 20 positive
documents, each with one gold entity, and 56 negatives: the selected model
finds 19/20 entities and has no false positives. Their high score is not
evidence of general long-document or chunk-boundary performance.

Earlier V0 architecture/adapter decisions used nested test subsets. The
1,500-record view excludes all 500 V0 test IDs conservatively, but is not an
untouched holdout: base and regex had already been evaluated on the full test.
Its 0.8885 F1 is useful as a separate diagnostic, not proof that prior test
exposure had no effect. Future tuning based on this error review needs an
independent, frozen holdout and a separately versioned experiment.

## Error-review priorities

All entity-level FN/FP counts reconcile with the frozen evaluator.

| Label | Gold support | Recall | Precision | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| PERSON_NAME | 903 | 0.7254 | 0.7384 | 248 | 232 |
| VEHICLE_ID | 178 | 0.8090 | 0.9351 | 34 | 10 |
| EMPLOYEE_ID | 272 | 0.8640 | 0.9553 | 37 | 11 |
| BANK_ACCOUNT | 118 | 0.8898 | 0.9545 | 13 | 5 |
| CREDIT_CARD | 185 | 0.8865 | 0.9213 | 21 | 14 |
| DATE_OF_BIRTH | 643 | 0.9145 | 0.9423 | 55 | 36 |
| ADDRESS | 411 | 0.9027 | 0.8812 | 40 | 50 |

The main priorities are names and addresses with boundary differences, omitted
dates of birth, and confusion between identifier categories. Names alone
account for 248 of 575 false negatives; 202 name FNs and 200 name FPs are
classified as likely boundary mismatches. This suggests where to review, not
whether the model or source annotation is necessarily wrong.

| Suspected cause | FN | FP |
| --- | ---: | ---: |
| Boundary mismatch | 250 | 241 |
| Wrong label | 135 | 135 |
| Missed entity | 178 | — |
| Invalid schema | 12 | — |
| Spurious entity | — | 169 |
| Absent-from-source text | — | 8 |
| Overpredicted occurrence | — | 4 |

A wrong label or boundary may produce both an FN and an FP; these are not
independent defective-document counts. The four schema failures are already
accounted for in the 12 schema-related FNs.

The deterministic bundle covers all 2,000 records in aggregate, with 483
documents having issues. It selects 361 unique documents using seed 42,
up to 20 nominations per label/cause group and a 500-document global cap.
Every observed group is represented, including all four invalid outputs.
A repeated execution produced identical artifacts.

Start with the local
[aggregate error summary](../../evaluation/results/v1-selected-ckpt19000-full-test/error-analysis/summary.md),
then inspect
[`review_samples.jsonl`](../../evaluation/results/v1-selected-ckpt19000-full-test/error-analysis/review_samples.jsonl)
for source text, gold spans, raw output, individual errors, and selection
reasons. That JSONL contains sensitive content and must remain local.

Review FNs first, especially titles/suffixes and address boundaries; then label
confusions, negative-document FPs, hallucinations, and invalid outputs.
Record proposed annotation corrections separately without editing the frozen
data or predictions. Human review tasks `H-070`–`H-073` are not marked done by
generating this bundle.

## Runtime and provenance

The full evaluation finished on 2026-09-14 at 18:41:26 UTC. Its recorded
generation calls total 4,217.511 seconds (70m 18s), with median 2.245 seconds
per document, P95 3.720 seconds, and 2.059 GB peak MLX allocation. Re-encoding
saved outputs gives 99,859 tokens, or approximately 23.68 output tokens/s;
this is the runner's accounting, not a production latency or Pi benchmark.

Generation used seed 42, temperature 0, thinking disabled, and maximum
384 output tokens. Base-model fingerprint, prompt/generation settings, and
dataset matched the corresponding base comparison. Selected validation and
test runs used identical adapter and base fingerprints. The largest measured
test prompt was 884 tokens; the 768-token training filter was not imposed at
inference.

Selected adapter: `models/adapters/v1-selected-l16-r8-768-ckpt19000`.

```text
Selected weights SHA-256
a2458e1c67949ac6d70b676ea9d727b92f7e643acce19c0157c32aa0d735bbac

Selection validation SHA-256
49f1df7ba0826f913a364aebf155441a7e5e58b574adb212ee4dec59a2d14042

Full test SHA-256
eca377c2185c87430bce78ba997789092e186af7c619230dd72b25508ee6b52f

Full predictions SHA-256
7e45a7697918af8381557470c6788016df8b333e1cb2d2a079bc66bbf9b5ecab

Local review samples SHA-256
829ab5cb3298373d863a31c8cb2304c0e41424a4bf4e84771ec322b451787644
```

The committed
[aggregate evidence snapshot](../../evaluation/baselines/main-adapter-review-v1.json)
contains all 14 per-class scores, validation ranking, four view comparisons,
hashes, sampling settings, and error totals without source text.
Raw predictions, manifests, full reports, and weights remain ignored locally.

## Reproduce and next gate

No retraining or GPU inference is needed to read the existing results.
For CPU-only reconstruction of the identical local error bundle:

```bash
source .venv/bin/activate
python scripts/evaluation/analyze_errors.py \
  --dataset data/processed/test.jsonl \
  --predictions evaluation/results/v1-selected-ckpt19000-full-test/predictions.jsonl \
  --expected-dataset-sha256 eca377c2185c87430bce78ba997789092e186af7c619230dd72b25508ee6b52f \
  --output-dir evaluation/results/v1-selected-ckpt19000-full-test/error-analysis
```

See [the evaluation workflow](../evaluation.md) for partitioning, filtering,
scoring, comparisons, and resuming an interrupted inference run. All view
dataset/prediction hashes and review artifact hashes were verified.

Implementation verification: 75 offline tests passed, with the network test
deselected; the repository artifact/secret guard passed. Generated source text
and weights remain excluded from Git.

Human decision `H-045` is approved as of 2026-09-15. Approval for packaging
trials does not authorize deployment. Next work is:

1. Check disk/memory headroom and fuse/dequantize into a new local directory.
2. Verify fused-versus-adapter quality parity with frozen inputs and manifests.
3. Pin/build `llama.cpp`, convert, and verify floating-point GGUF parity.
4. Generate and compare quantized candidates under the approved recall-loss
   tolerance; choose Pi packaging only after actual device measurements.

Adapter selection is complete. Packaging must still stop on unexplained
quality regressions or unsafe resource pressure; release remains unapproved.
