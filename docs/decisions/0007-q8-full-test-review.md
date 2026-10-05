# Decision 0007 — Q8 full-test results require deployment review

Status: approved evaluation complete; deployment decision pending

Date: 2026-09-25

## Outcome

The validation-selected Q8_0 model completed the entire frozen 2,000-record test
with exit code 0. Its exact entity recall is **89.19%**, F1 **89.16%**,
complete-document recall **75.87%**, and schema validity **99.80%**.

This completes the approved Q8 evaluation, not approval for autonomous redaction,
Pi transfer, or deployment. There are still **556 missed gold entity occurrences**
and **362 of 1,500 positive documents without complete recall**. Names, identifier
confusion, span boundaries, and invalid output require human review.

The candidate was chosen using validation and explicitly approved before test
execution. Neither this test nor its error examples may be used to reselect the
quantization, tune decoding, or populate training data. Q4 and Q5 remain
ineligible under the unchanged validation gates.

## Frozen evidence

- Pre-test protocol: [`models/q8-full-test-v1.json`](../../models/q8-full-test-v1.json).
- Aggregate results: [`evaluation/baselines/q8-full-test-v1.json`](../../evaluation/baselines/q8-full-test-v1.json).
- Model: checkpoint 19,000, Q8_0, 1,834,426,048 bytes.
- Model SHA-256: `d421e331a5b9920ebf99eebf9902b4a2db535a3505b623042b6ab1f230e02858`.
- Test SHA-256: `eca377c2185c87430bce78ba997789092e186af7c619230dd72b25508ee6b52f`.
- Predictions SHA-256: `769b4bdab35027e85ed9ea1774ad84c2d40742b79c648d001fe7cb34dbd405f1`.
- Completed run manifest SHA-256: `74aed9ab7197a2c76b041f4990a3a59045de509a0a85d8515108ca75315762d7`.
- Aggregate report SHA-256: `c47b6fcee9d6a2f94575ec60c4a99433a361f10ba77231f8a6f47f2d4c88c133`.

All 2,000 predictions are complete, unique, and in frozen source order. The
1,238 predictions present before the final resume are byte-for-byte unchanged.
The test contains 1,500 positive documents, 500 negative documents, and 5,143
gold entities. Every prompt passed native template/token-ID parity, and the
maximum prompt plus output budget was 1,268 tokens, within the 2,048-token
context. No prompt was truncated.

Seed 42, temperature 0, non-thinking mode, the 384-token output cap, model,
tokenizer, native toolchain, runner, transport, and strict evaluator remained
unchanged. Raw predictions and all source-containing review files remain local
and ignored by Git.

## Descriptive comparison with the selected MLX adapter

| Metric | Selected MLX adapter | Q8_0 GGUF |
| --- | ---: | ---: |
| Exact true positives | 4,568 | 4,587 |
| Exact false positives | 557 | 559 |
| Exact false negatives | 575 | 556 |
| Exact recall | 88.82% | 89.19% |
| Exact precision | 89.13% | 89.14% |
| Exact F1 | 88.98% | 89.16% |
| Complete-document recall, positive documents | 75.20% | 75.87% |
| Exact-document match, all documents | 75.85% | 76.20% |
| Schema-valid outputs | 99.80% | 99.80% |
| Negative-document false-positive rate | 0.80% | 0.80% |

Relative to the adapter, Q8 has 19 more true positives, 19 fewer false negatives,
and two more false positives. Recall rises by 0.37 percentage points and
complete-document recall by 0.67 points. There are 52 changed raw outputs and
52 changed parsed entity sequences; 1,948 raw outputs are identical.

The reference is the selected **MLX adapter**, not the BF16 GGUF parent. There
is no full-test BF16 output, so these differences cannot isolate quantization
effects or establish full-test BF16/Q8 parity. The earlier 100-record validation
parity remains the evidence for the quantization gates. These small descriptive
differences are not a statistical significance or universal-equivalence claim.

## Remaining errors

The local error bundle is
`evaluation/results/v1-gguf-q8_0-full-test-errors/`. Start with `summary.md`, then
inspect `review_samples.jsonl` locally. The latter contains source text and must
not be committed or copied into logs.

The deterministic sampler uses seed 42, up to 20 nominations per error group,
and a 500-document cap. It selected **353 unique documents** representing all
76 groups; the full test contains 476 documents with exact-scoring issues.
Its manifest SHA-256 is
`7831fb6dba345159c13fb8f794e554add677b7456a848aff98abb96725aa50c7`.

Priority review areas:

1. **Names and boundaries:** PERSON_NAME has 240 FN and 235 FP, with recall
   73.42%. The heuristic marks 198 name FN and 201 name FP as boundary mismatches.
   ADDRESS contributes another 34 boundary FN and 29 boundary FP. Inspect gold
   span conventions before declaring the cause established.
2. **Identifier confusion and misses:** 136 FN and 136 FP are heuristically
   assigned to wrong labels. EMPLOYEE_ID recall is 85.29%, down 1.10 percentage
   points from the adapter; CUSTOMER_ID recall is down 0.35 points. Aggregate
   improvement does not remove these per-class regressions. VEHICLE_ID recall
   remains 82.02%.
3. **Invalid outputs:** all four invalid documents are also invalid in the
   adapter reference. Three Q8 outputs end at the 384-token cap with unterminated
   JSON; one stops at EOS with a JSON delimiter error. The other 1,996 outputs
   are schema-valid. No repair, retry, or output-budget change was applied.
   Strict scoring discards the whole invalid prediction, contributing 12 FN.
4. **False positives:** four of 500 negative documents have false positives.
   The evaluator reports ten unassignable predicted substrings; the error
   heuristic separates these into seven absent-text predictions and three
   overpredicted occurrences. They are not all invented text.

Across labels, heuristic FN causes are 241 boundary mismatches, 167 missed
entities, 136 wrong labels, and 12 from invalid schema. FP causes are 239 boundary
mismatches, 174 spurious entities, 136 wrong labels, seven absent-text predictions,
and three overpredicted occurrences. These are deterministic diagnostic groups,
not established causal explanations or one-to-one FN/FP pairings.

## Execution and timing limits

The first and third attempts stopped for disk pressure; their lifecycle evidence
and the second attempt's unknown exit reason are preserved. The final resume
began with about 20.2 GiB free after additional space was made available and AC
was confirmed in the actual execution environment. Its 762 remaining records
completed in about 27 minutes of generation. The separate watchdog observed a
minimum 11.26 GiB free and 50% battery; no safety stop occurred. The native server,
runner, and scoped sleep-prevention process were confirmed stopped afterward.

No cache setting was changed, and no model or experiment artifact was deleted.
The pinned server's separate 8,192 MiB RAM prompt-state cache remains a plausible,
unmeasured contributor to memory/swap pressure. Resource recovery after stopping
must not be mistaken for additional headroom for a future run.

The full-test timing checker found no native/client clock anomalies. Its saved
rates are nevertheless **local diagnostics only**: the test spanned multiple
resumed sessions, resource pressure was not controlled, and peak native RAM
was not measured. These are not Pi speed, power, temperature, or memory results.

## Verification

All 174 offline tests pass; the network integration test is excluded. Rebuilding
the aggregate report reproduces it exactly, and rerunning error analysis reuses
byte-identical artifacts. Bundle checksums, unique sample counts, complete test
coverage, score reconciliation, and the frozen prompt digest were verified.
The repository artifact/secret guard and the aggregate privacy-field check pass.
Only aggregate results and documentation are committed; weights, source records,
raw predictions, and review samples remain ignored.

Reverified on 2026-10-05 before completing the previously blocked commit/push:
all 174 offline tests pass again, the aggregate reproduces exactly, and frozen
artifact, prompt, error-bundle, and privacy checks pass. No generation was rerun.

## Next human decision

Review the local error bundle and decide whether to approve a **Q8-only hardware
benchmark**, given the 1.834 GB model and remaining extraction errors, or request
a separately versioned development experiment. H-046/deployment approval and G7
closure remain pending. No Pi transfer, service installation, new training,
changed calibration, or relaxed quality threshold is authorized by this report.

Future development should use separately approved training/validation evidence,
not tune against these frozen test examples. Any deployment must also pass the
document engine, privacy, hardware, and end-to-end redaction gates.
