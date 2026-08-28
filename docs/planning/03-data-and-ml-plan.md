# Data and machine-learning plan

## Objective

Produce a reproducible fine-tuned Qwen3-1.7B adapter, prove that it improves the
PII extraction task, convert it to a validated GGUF parent model, and measure
the quality/performance trade-offs of deployment quantization.

## 1. Source data discovery

- [x] **ML-001 — Pin the dataset source.** Record the exact revision of
  `gretelai/gretel-pii-masking-en-v1`, its license, split names, and download
  metadata.
- [x] **ML-002 — Implement the downloader.** Download through `datasets`, use a
  configurable cache location, and print the schema and a safe example summary.
- [x] **ML-003 — Inspect source structure.** Report all fields, entity formats,
  source offsets, label vocabulary, domains, document types, and split sizes.
- [x] **ML-004 — Measure source distribution.** Calculate document characters,
  Qwen tokens, entities per document, class counts, domain counts, and P50/P75/
  P90/P95/P99/max lengths.
- [x] **ML-005 — Detect source defects.** Report malformed offsets, text/span
  mismatches, empty entities, nested spans, duplicates, and suspiciously similar
  records across splits.
- [x] **ML-006 — Produce a human-review bundle.** Sample examples by label,
  domain, exclusion reason, ambiguity, and length for `H-030` through `H-036`.

### Exit gate

The source dataset is understood statistically and the human-approved mapping
policy covers every observed source label.

## 2. Canonical dataset construction

- [x] **ML-010 — Implement label mapping.** Map approved source labels to the
  canonical enum and discard excluded labels deterministically.
- [x] **ML-011 — Preserve exact entity substrings.** Prefer source offsets when
  provided, verify `text[start:end]`, and never normalize the target text.
- [x] **ML-012 — Order and retain occurrences.** Sort targets by source position
  and preserve duplicate occurrences rather than deduplicating their values.
- [x] **ML-013 — Render the canonical prompt.** Use one versioned system prompt
  in training, base evaluation, adapter evaluation, and deployment.
- [x] **ML-014 — Write MLX chat JSONL.** Each record contains system, user, and
  assistant messages with compact JSON assistant output.
- [x] **ML-015 — Group custom templates before splitting.** Prevent structurally
  equivalent templates with substituted values from appearing across train,
  validation, and test.
- [x] **ML-016 — Create stable sample IDs.** IDs must remain stable when reports
  are regenerated from the same dataset version.
- [x] **ML-017 — Write a dataset manifest.** Include source revision, mapping
  policy version, prompt version, seed, filters, counts, and file hashes.

## 3. Controlled synthetic additions

- [x] **ML-020 — Generate negative examples.** Target 20–30% of training records
  with `{"entities":[]}` and no hidden real PII.
- [x] **ML-021 — Generate hard negatives.** Pair semantically close positive and
  negative examples for DOB/generic date, address/city, username/field name,
  account/arbitrary number, and IP address/protocol discussion.
- [x] **ML-022 — Generate long-context composites.** Mix mostly non-PII prose
  with PII paragraphs at varied positions and produce examples near the intended
  500–1,000-token deployment chunk distribution.
- [x] **ML-023 — Generate repeated-entity examples.** Include identical entity
  values at several positions and require all occurrences in source order.
- [x] **ML-024 — Improve rare-class coverage.** Add controlled templates for
  underrepresented labels without allowing a template family to leak across
  splits.
- [x] **ML-025 — Create calibration data.** Build separate PII, non-PII, and
  hard-negative text for importance-matrix generation. Never derive it from the
  frozen test set.

## 4. Dataset validation

- [x] **ML-030 — Validate JSONL syntax and messages.** Reject malformed lines,
  missing roles, duplicate roles, or non-string content.
- [x] **ML-031 — Validate assistant JSON.** Require only the approved response
  shape and approved labels.
- [x] **ML-032 — Validate source membership.** Every entity value must be found
  in the user text with the expected occurrence count.
- [x] **ML-033 — Validate source order.** Returned entities must align to
  monotonically increasing source positions, including repeated values.
- [x] **ML-034 — Validate split isolation.** Check exact hashes, normalized
  hashes, and template-group identifiers across splits.
- [x] **ML-035 — Validate length constraints.** Report records that will be
  truncated at each proposed maximum sequence length.
- [x] **ML-036 — Validate distribution.** Fail on accidental disappearance of a
  required class or a negative ratio outside the configured tolerance.
- [x] **ML-037 — Generate the final statistics report.** Report counts, lengths,
  class distribution, negative ratio, domains, and validation status.

### Dataset release stages

| Dataset | Intended use | Approximate size |
| --- | --- | ---: |
| Smoke | Pipeline validation | 500 train records |
| V0 | First meaningful run | 5,000 train records |
| V1 candidate | Main experiments | 15,000–25,000 train records |
| Extended | Only after evidence of data limitation | Larger as justified |

## 5. Evaluation framework and frozen baselines

- [ ] **ML-040 — Implement exact occurrence matching.** Match type and exact
  source occurrence, treating duplicates as distinct instances.
- [ ] **ML-041 — Implement metrics.** Report micro/macro precision, recall, F1,
  per-class results, complete-document recall, and PII-free false-positive rate.
- [ ] **ML-042 — Report validity.** Measure raw JSON validity, schema validity,
  unknown labels, hallucinated substrings, and missing repeated occurrences.
- [ ] **ML-043 — Add relaxed diagnostics.** Optional overlap or partial-text
  metrics may help analysis but must not replace strict metrics.
- [ ] **ML-044 — Implement regex baseline.** Cover syntactically constrained
  labels and make unsupported categories explicit.
- [ ] **ML-045 — Evaluate untouched Qwen.** Run the frozen prompt/test set and
  save raw and parsed predictions with a manifest.
- [ ] **ML-046 — Freeze test artifacts.** Record hashes so later data or prompt
  changes cannot silently alter the comparison.

## 6. QLoRA smoke run

- [ ] **ML-050 — Create smoke configuration.** Start with batch 1, accumulation
  8, eight layers, rank 8, 768 sequence length, prompt masking, gradient
  checkpointing, and about 100 iterations.
- [ ] **ML-051 — Capture resource metrics.** Save peak memory, throughput,
  training/validation loss, and duration.
- [ ] **ML-052 — Verify adapter artifacts.** Confirm configuration and weights
  can be loaded after the process exits.
- [ ] **ML-053 — Verify generation parity.** Use the same chat template and
  prompt rendering as base evaluation.
- [ ] **ML-054 — Evaluate the smoke adapter.** Quality is diagnostic; success is
  a functioning end-to-end pipeline with finite loss.

## 7. Staged QLoRA experiments

- [ ] **ML-060 — Run 8-layer/rank-8/768 baseline experiment.** Establish the
  first meaningful adapter on 5k–10k records.
- [ ] **ML-061 — Test 16 LoRA layers.** Only proceed if memory remains safe.
- [ ] **ML-062 — Test rank 16.** Compare only after holding dataset and other
  variables fixed.
- [ ] **ML-063 — Test sequence length 1024.** Measure truncation improvement,
  memory, throughput, and quality.
- [ ] **ML-064 — Train the main candidate.** Use the winning small experiment on
  the 15k–25k dataset.
- [ ] **ML-065 — Evaluate checkpoints, not just the last step.** Rank checkpoints
  using validation task metrics.
- [ ] **ML-066 — Produce error-analysis samples.** Group false negatives and
  false positives by class and suspected cause.
- [ ] **ML-067 — Select the adapter.** Require human approval `H-045` and save a
  model-selection decision record.

Every experiment must record:

- model and tokenizer revision;
- dataset and prompt versions;
- Git commit and configuration hash;
- random seed;
- layers, rank, scale, dropout, learning rate, batch, accumulation, sequence;
- iterations and approximate examples consumed;
- training/validation loss and task metrics;
- throughput, peak memory, duration, and adapter checksum.

## 8. Adapter fusion and GGUF conversion

- [ ] **ML-070 — Fuse and dequantize.** Fuse the selected adapter into the MLX
  base and write a floating-point checkpoint.
- [ ] **ML-071 — Validate fused MLX.** Compare it with base-plus-adapter on a
  parity subset and the full frozen evaluation if practical.
- [ ] **ML-072 — Pin and build `llama.cpp` on the Mac.** Record its exact commit,
  converter dependencies, compiler, and build options.
- [ ] **ML-073 — Convert to BF16 GGUF.** Capture converter output and produce a
  checksum.
- [ ] **ML-074 — Validate chat metadata.** Confirm tokenizer, chat template,
  special tokens, EOS handling, context size, and thinking behavior.
- [ ] **ML-075 — Run conversion-parity evaluation.** Compare selected-adapter,
  fused-MLX, and BF16-GGUF metrics and example-level differences.

Stop if conversion causes an unexplained quality regression.

## 9. Deployment quantization

- [ ] **ML-080 — Generate Q8_0.** Use the BF16 GGUF as the common parent.
- [ ] **ML-081 — Generate Q5_K_M.** Record command, size, and checksum.
- [ ] **ML-082 — Generate Q4_K_M.** Record command, size, and checksum.
- [ ] **ML-083 — Generate an importance matrix.** Use only the separate
  calibration corpus.
- [ ] **ML-084 — Generate imatrix Q4_K_M.** Keep it distinct from ordinary Q4.
- [ ] **ML-085 — Evaluate all formats identically.** Use the same prompt,
  decoding, test samples, parser, and metrics.
- [ ] **ML-086 — Select Pi candidates.** Normally carry Q4 and Q5 to the Pi,
  while retaining BF16/Q8 as quality references.

### Quantization report fields

| Field | Reason |
| --- | --- |
| Artifact size | Storage and transfer cost |
| Peak RAM | Deployment feasibility |
| Prompt tokens/s | Long-chunk preprocessing speed |
| Generation tokens/s | Output latency |
| JSON/schema validity | Serving reliability |
| Exact recall/F1 | Privacy-task quality |
| Complete-document recall | End-to-end risk signal |
| Per-class delta from BF16 | Detect hidden class collapse |

## ML stop conditions

Stop the current stage and diagnose when:

- dataset validation fails;
- the base-evaluation prompt differs from the training prompt unintentionally;
- training loss is non-finite;
- memory pressure makes the Mac unstable;
- the adapter does not materially improve on the base model;
- fusion or conversion introduces unexplained output changes;
- quantization exceeds the human-approved recall tolerance;
- calibration/test leakage is discovered;
- any artifact cannot be tied to a reproducible manifest.
