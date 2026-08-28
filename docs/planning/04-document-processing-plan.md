# Document-processing plan

## Objective

Process text longer than the model context window while preserving exact
character positions, recovering boundary entities, rejecting hallucinated
values, and modifying only confirmed spans. This subsystem must be deterministic
and testable without a real model.

## Core contracts

### Chunk

```python
Chunk(
    chunk_id: int,
    char_start: int,
    char_end: int,
    text: str,
    token_count: int,
)
```

Required invariant:

```python
chunk.text == document[chunk.char_start:chunk.char_end]
```

### Detection

```python
Detection(
    type: EntityType,
    text: str,
    start: int,
    end: int,
    chunk_id: int,
    boundary_distance: int | None,
)
```

Required invariants:

```python
detection.text == document[detection.start:detection.end]
0 <= detection.start < detection.end <= len(document)
```

## 1. Tokenizer abstraction

- [ ] **DOC-001 — Define a tokenizer protocol.** The chunker receives an
  asynchronous `count_tokens(text)` capability and knows nothing about HTTP,
  Transformers, MLX, or GGUF.
- [ ] **DOC-002 — Add a local tokenizer adapter.** Use the exact Qwen tokenizer
  during Mac development and deterministic tests.
- [ ] **DOC-003 — Add a `llama-server` tokenizer adapter.** Call `/tokenize` and
  validate the returned token representation.
- [ ] **DOC-004 — Add batching/caching where safe.** Cache counts only for exact
  immutable strings and use a bounded cache.
- [ ] **DOC-005 — Test tokenizer parity.** Compare local and server token counts
  on representative Unicode, whitespace, punctuation, and long inputs.

## 2. Lossless text segmentation

- [ ] **DOC-010 — Identify paragraph boundaries.** Keep separators in the source
  representation so concatenation does not alter the document.
- [ ] **DOC-011 — Identify sentence boundaries.** Store character spans instead
  of normalized sentence strings.
- [ ] **DOC-012 — Preserve difficult input.** Cover blank lines, CRLF, tabs,
  Unicode punctuation, emoji, combining marks, and text with no terminal period.
- [ ] **DOC-013 — Avoid semantic cleanup.** Do not trim, collapse whitespace,
  normalize Unicode, or rewrite line endings before calculating spans.
- [ ] **DOC-014 — Implement long-unit fallback.** If one sentence exceeds the
  token target, locate the largest safe character slice through bounded token
  count checks while guaranteeing forward progress.

## 3. Token-aware chunk construction

- [ ] **DOC-020 — Pack units to a content budget.** Start at 1,024 tokens but
  make the value configurable.
- [ ] **DOC-021 — Reserve model context.** Account separately for system prompt,
  chat-template overhead, output allowance, and safety margin inside the 2,048
  context target.
- [ ] **DOC-022 — Add overlap by source spans.** Carry prior sentences or text
  until the overlap is near 128 tokens without reconstructing the text.
- [ ] **DOC-023 — Guarantee progress.** The next chunk must end after the current
  chunk even when one token or sentence is unexpectedly large.
- [ ] **DOC-024 — Guarantee coverage.** Every input character must appear in at
  least one chunk, excluding no whitespace or separator.
- [ ] **DOC-025 — Bound over-budget chunks.** Only the documented long-unit
  fallback may approach the model budget; it must never exceed the full safe
  request budget.
- [ ] **DOC-026 — Expose generator semantics.** V1 may return a list, but the
  internal interface should support yielding chunks to ease future streaming.

### Chunker acceptance tests

- Empty text produces a defined empty result.
- Short text produces exactly one exact source slice.
- Every character is covered.
- Neighboring chunks overlap when more than one chunk exists.
- The non-overlapping progression never moves backward.
- Token counts are exact according to the injected tokenizer.
- All chunk spans are within the document.
- A single huge sentence cannot cause an infinite loop.

## 4. Model-response alignment

- [ ] **DOC-030 — Validate model entities first.** Reject unknown labels, empty
  values, and response objects outside the approved schema.
- [ ] **DOC-031 — Implement exact substring search.** Do not use fuzzy matching
  for V1 redaction.
- [ ] **DOC-032 — Handle repeated values sequentially.** For source-ordered model
  output, search after the previous consumed occurrence where appropriate.
- [ ] **DOC-033 — Track occurrence use.** One source occurrence should not be
  assigned to several identical returned entities from the same chunk unless
  policy explicitly allows it.
- [ ] **DOC-034 — Reject hallucinated values.** Preserve a non-sensitive warning
  count, not the hallucinated value, for operational reporting.
- [ ] **DOC-035 — Convert local spans to global spans.** Add `chunk.char_start`
  and immediately verify the global source substring.
- [ ] **DOC-036 — Calculate boundary quality.** Store the minimum distance from
  the detection to either chunk boundary for reconciliation.
- [ ] **DOC-037 — Make ambiguity observable.** If identical text occurs several
  times and returned occurrence count cannot be aligned confidently, emit a
  typed diagnostic rather than silently inventing an offset.

### Known alignment limitation

If a chunk contains identical text in both PII and non-PII contexts but the
model returns only one occurrence, source order alone may not identify which
occurrence the model intended. Tests and error analysis must measure this case.
V1 should use deterministic behavior and diagnostics; a contextual adjudication
call belongs in V2.

## 5. Cross-chunk reconciliation

- [ ] **DOC-040 — Remove exact duplicates.** Collapse detections with identical
  type, start, and end.
- [ ] **DOC-041 — Group overlapping detections.** Build deterministic overlap
  groups without depending on input iteration order.
- [ ] **DOC-042 — Compare same-type near duplicates.** Use interval IoU with a
  configurable initial threshold near 0.7.
- [ ] **DOC-043 — Prefer better chunk context.** Prefer the candidate farther
  from its source chunk boundary.
- [ ] **DOC-044 — Apply deterministic tie breakers.** Define ordering for equal
  boundary distance, equal span length, and equal chunk position.
- [ ] **DOC-045 — Resolve different-label conflicts.** Follow the human-approved
  policy; initially prefer better boundary context and record a warning.
- [ ] **DOC-046 — Validate final non-overlap.** No unresolved overlapping spans
  may reach the redaction function.
- [ ] **DOC-047 — Preserve provenance for debugging.** Test/evaluation results
  may record contributing chunk IDs, but production logs must not record text.

## 6. Deterministic redaction

- [ ] **DOC-050 — Validate spans against the immutable source.** Recheck range,
  ordering, text equality, and non-overlap.
- [ ] **DOC-051 — Sort right to left.** Replace in descending start order so
  earlier offsets remain valid.
- [ ] **DOC-052 — Apply typed placeholders.** Use an explicit label-to-placeholder
  mapping and reject labels without a configured replacement.
- [ ] **DOC-053 — Preserve all unaffected characters.** Add a reconstruction
  assertion in tests that proves only selected spans changed.
- [ ] **DOC-054 — Return a structured result.** Include redacted text, final
  detections, counts by type, warnings, chunk count, and timings.
- [ ] **DOC-055 — Keep redaction pure.** The function must perform no I/O,
  logging, model calls, or global configuration mutation.

## 7. Complete processor

- [ ] **DOC-060 — Define a model-client protocol.** The processor consumes
  `detect(text) -> list[ModelEntity]` without runtime-specific knowledge.
- [ ] **DOC-061 — Process chunks sequentially.** Initial concurrency is one to
  avoid Pi contention and nondeterministic ordering.
- [ ] **DOC-062 — Add progress callbacks.** Report chunk index/total, elapsed
  time, and entity count without reporting document content.
- [ ] **DOC-063 — Add cancellation checks.** Stop between model calls when a job
  is cancelled or the client disconnect policy requires termination.
- [ ] **DOC-064 — Define partial-failure behavior.** Default recommendation:
  return an error and no falsely complete redacted document when any chunk fails
  after retry policy is exhausted.
- [ ] **DOC-065 — Reconcile only after detection.** Do not mutate source text
  while model calls are outstanding.
- [ ] **DOC-066 — Add a fake deterministic model.** Unit and integration tests
  must exercise the whole pipeline without loading Qwen.

## 8. Long-document validation

- [ ] **DOC-070 — Generate documents with known global spans.** Cover 5k, 10k,
  20k, and 50k token documents.
- [ ] **DOC-071 — Generate boundary torture cases.** Place each representative
  entity immediately before, inside, and immediately after intended boundaries.
- [ ] **DOC-072 — Generate repeated-context cases.** Repeat the same value in PII
  and non-PII contexts and across overlapping chunks.
- [ ] **DOC-073 — Benchmark chunk configurations.** Compare 512/64, 1024/64,
  1024/128, 1024/256, and 1536/128 where the context budget permits.
- [ ] **DOC-074 — Report application metrics.** Include boundary recall,
  duplicate rate, conflict rate, calls per 10k tokens, seconds per chunk, and
  complete-document recall.

## Document-engine exit gate

- All invariants pass for generated and adversarial fixtures.
- No hallucinated text can become a redaction span.
- No unresolved overlap reaches redaction.
- Non-redacted content is demonstrably unchanged.
- Boundary configuration is selected from measurements.
- The complete engine works with both a fake client and the selected real model.
