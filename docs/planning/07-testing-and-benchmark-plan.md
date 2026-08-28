# Testing and benchmark plan

## Test principles

- Build the evaluation system before training.
- Test deterministic logic without loading a model.
- Keep the frozen test set isolated from training, hyperparameter selection,
  synthetic-template reuse, and importance-matrix calibration.
- Separate model-quality, application-quality, and hardware-performance results.
- Save machine-readable results and the configuration needed to reproduce them.
- Never treat JSON validity as proof of semantic correctness.

## 1. Test organization

- [ ] **QA-001 — Define pytest markers.** Recommended markers: `unit`,
  `integration`, `model`, `slow`, `mac`, and `pi`.
- [ ] **QA-002 — Create deterministic fixtures.** Include all labels, negatives,
  hard negatives, repeats, overlap conflicts, Unicode, and long text.
- [ ] **QA-003 — Create fake tokenizer and model clients.** Make document and API
  tests fast and independent of external processes.
- [ ] **QA-004 — Add golden outputs.** Version small expected JSON and redacted
  outputs with explicit review when they change.
- [ ] **QA-005 — Add result manifests.** Record code/config/data/model versions
  for every real-model or benchmark output.

## 2. Schema and data unit tests

- [ ] **QA-010 — Test canonical labels and serialization.** Reject unknown or
  misspelled types.
- [ ] **QA-011 — Test label mapping.** Cover every observed source label and
  every explicit exclusion.
- [ ] **QA-012 — Test malformed JSONL.** Cover syntax, message roles, invalid
  assistant JSON, empty values, and missing fields.
- [ ] **QA-013 — Test entity/source consistency.** Detect offsets outside source,
  mismatched text, missing occurrences, and incorrect source order.
- [ ] **QA-014 — Test split isolation.** Detect exact duplicate, normalized
  duplicate, and reused template-group IDs.
- [ ] **QA-015 — Test deterministic builds.** Identical inputs, seed, and config
  must produce identical sample IDs, manifests, and file hashes.

## 3. Chunking unit tests

- [ ] **QA-020 — Test empty and short input.** Define results for zero characters
  and a document that fits one chunk.
- [ ] **QA-021 — Test exact source slicing.** Assert each chunk equals its source
  slice for ordinary and Unicode text.
- [ ] **QA-022 — Test complete coverage.** Every source position appears in at
  least one chunk.
- [ ] **QA-023 — Test token budgets.** Verify target, overlap, reserved context,
  and hard maximum behavior.
- [ ] **QA-024 — Test segmentation.** Cover paragraphs, blank lines, CRLF, no
  punctuation, abbreviations, and extremely long sentences.
- [ ] **QA-025 — Test overlap.** Verify expected shared source spans and forward
  progress.
- [ ] **QA-026 — Property-test invariants.** Generate arbitrary text and ensure
  valid bounds, exact slices, coverage, and termination.

## 4. Alignment, reconciliation, and redaction tests

- [ ] **QA-030 — Test exact matching.** Accept exact substrings and reject
  normalization or hallucination.
- [ ] **QA-031 — Test repeated occurrences.** Align first, middle, and last
  repeated values in source order.
- [ ] **QA-032 — Test local/global conversion.** Include nonzero chunk starts and
  multibyte Unicode before entities.
- [ ] **QA-033 — Test duplicate collapse.** Same type/start/end from several
  chunks becomes one detection.
- [ ] **QA-034 — Test near-overlap handling.** Cover spans above, below, and
  exactly at the IoU threshold.
- [ ] **QA-035 — Test boundary preference.** Prefer the candidate with better
  surrounding chunk context.
- [ ] **QA-036 — Test label conflicts.** Exercise deterministic policy and
  diagnostics.
- [ ] **QA-037 — Test right-to-left redaction.** Cover length-changing,
  adjacent, first-character, and end-of-document replacements.
- [ ] **QA-038 — Test unchanged-text proof.** Reconstruct the expected document
  and verify that no character outside selected spans changes.
- [ ] **QA-039 — Test invalid final spans.** Reject overlap, reversed positions,
  out-of-range positions, and text mismatch.

## 5. Model and evaluator tests

- [ ] **QA-040 — Test prediction parsing.** Cover valid, malformed, wrapped,
  extra-field, unknown-label, and empty-output responses.
- [ ] **QA-041 — Test exact metric arithmetic.** Use hand-calculated fixtures
  containing duplicates and per-class errors.
- [ ] **QA-042 — Test complete-document recall.** Distinguish high entity F1 from
  documents with at least one missed entity.
- [ ] **QA-043 — Test negative-document scoring.** Measure and report documents
  where the model invents PII.
- [ ] **QA-044 — Test result immutability.** New runs create new result paths or
  fail rather than silently overwriting baselines.
- [ ] **QA-045 — Test prompt consistency.** Training, MLX evaluation, and
  `llama-server` rendering reference the same prompt version.

## 6. API and privacy tests

- [ ] **QA-050 — Test health and readiness.** Cover healthy app, unavailable
  model, loading model, and invalid configuration.
- [ ] **QA-051 — Test redaction API.** Validate response schema and fake-client
  end-to-end behavior.
- [ ] **QA-052 — Test file handling.** Cover supported extensions, invalid
  encoding, wrong type, empty file, and oversized file.
- [ ] **QA-053 — Test progress.** Verify ordered events, completion, failure,
  disconnect, and cancellation.
- [ ] **QA-054 — Test concurrency and queue limits.** Ensure bounded work and a
  stable busy response.
- [ ] **QA-055 — Test safe errors.** Responses contain no stack trace, prompt,
  content, or internal path.
- [ ] **QA-056 — Test logging privacy.** Submit recognizable canary PII and scan
  captured application/access logs for its absence.
- [ ] **QA-057 — Test browser persistence.** Confirm document content is not
  written to local/session storage, service-worker caches, or URL parameters.
- [ ] **QA-058 — Test security headers and CORS.** Verify the approved origin and
  embedding policy.

## 7. Boundary and long-document test corpus

Generate known-ground-truth documents for:

- 5k, 10k, 20k, and 50k token lengths;
- PII-free and PII-heavy distributions;
- every label near the document beginning, quartiles, and end;
- one token before, inside, and one token after intended chunk boundaries;
- repeated identical values;
- long names, addresses, and identifiers;
- Unicode, mixed line endings, and long unbroken text.

- [ ] **QA-060 — Version the generator.** Store seed, template versions, and
  injection spans.
- [ ] **QA-061 — Validate generated truth.** Confirm injected text and offsets
  before using documents in benchmarks.
- [ ] **QA-062 — Run chunk matrix.** Compare supported chunk/overlap pairs under
  identical model settings.
- [ ] **QA-063 — Report boundary-specific metrics.** Keep them separate from
  ordinary single-chunk model results.

## 8. Model-quality benchmark

Compare:

```text
Regex
Base Qwen3-1.7B
Selected QLoRA adapter
Fused MLX
BF16 GGUF
Q8_0 GGUF
Q5_K_M GGUF
Q4_K_M GGUF
Q4_K_M + imatrix GGUF
```

For every system, report:

- strict precision, recall, and F1;
- per-class precision, recall, F1, and support;
- complete-document recall;
- PII-free document false-positive rate;
- JSON and schema validity;
- hallucinated-substring count;
- missing-repeated-occurrence count;
- relaxed diagnostics, clearly labeled as non-primary;
- delta from the selected adapter and BF16 parent.

- [ ] **QA-070 — Fix decoding and prompt settings.** Differences must reflect
  the model representation, not accidental request changes.
- [ ] **QA-071 — Run repeated benchmarks where decoding is nondeterministic.**
  Report variation or eliminate it through deterministic settings.
- [ ] **QA-072 — Produce machine-readable output.** Preserve per-example results
  and summary tables.

## 9. Raspberry Pi performance benchmark

Benchmark short, 1k-token, 10k-token, 50k-token, PII-heavy, PII-free, and
boundary-test documents.

- [ ] **QA-080 — Define environment capture.** Record OS, firmware, Pi revision,
  RAM, storage, cooling, model hash, llama commit, build flags, and app commit.
- [ ] **QA-081 — Define warm-up and repetition.** Exclude first-load effects from
  steady-state results while reporting startup separately.
- [ ] **QA-082 — Measure process resources.** Model startup, resident/peak RAM,
  swap, CPU, temperature, and throttling.
- [ ] **QA-083 — Measure inference throughput.** Prompt tokens/s, generation
  tokens/s, seconds/chunk, and seconds/1k input tokens.
- [ ] **QA-084 — Measure end-to-end latency.** Include tokenization, HTTP,
  alignment, reconciliation, and redaction.
- [ ] **QA-085 — Run sustained workload.** Look for temperature plateau,
  throttling, memory growth, crashes, and service responsiveness.
- [ ] **QA-086 — Compare concurrency one versus two only if safe.** Keep one as
  the default unless two improves end-to-end throughput without quality or
  stability cost.

## 10. Release acceptance matrix

| Area | Required evidence | Pass condition |
| --- | --- | --- |
| Dataset | Validator report and manifest | No validation failures |
| Fine-tuning | Base versus adapter report | Human-approved material improvement |
| Conversion | Adapter/fused/BF16 parity | No unexplained material regression |
| Quantization | BF16/Q8/Q5/Q4 report | Selected model within approved tolerance |
| Chunking | Boundary and long-document report | Approved boundary recall and stable spans |
| Redaction | Unit/property tests | Only intended spans change |
| API privacy | Canary/log/persistence tests | No content leakage observed |
| Pi | Performance and sustained-run report | Fits, remains stable, and avoids unacceptable throttling |
| Deployment | Reboot/offline/isolation tests | All pass without manual intervention |

## Benchmark result layout

Store results under a run-specific directory containing:

```text
manifest.json
summary.json
summary.md
predictions.jsonl
errors.jsonl
environment.json
timings.csv
```

Evaluation data is synthetic, but result files should still be treated
carefully because they may contain entity-like values. Production documents must
never enter this benchmark storage path.
