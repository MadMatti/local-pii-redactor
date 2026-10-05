# Q8-only Pi benchmark preparation

Date: 2026-10-05

Status: **draft; target hardware and benchmark authorization pending**

This is preparation for a hardware experiment, not permission to connect to a
device, copy weights, build native code, install packages, or deploy services.
The read-only [inventory command](README.md) is implemented and tested locally;
no real Pi inventory or benchmark has been obtained.

Local verification: all 204 offline tests pass, including 30 diagnostic tests.
The CLI works with Python site packages disabled. A read-only Mac smoke check
correctly returned `review_required` (exit 2); it is not Pi hardware evidence.
The completed Q8 report and frozen generation-code hashes remain unchanged.

## Candidate and evidence boundary

Only the already validation-selected Q8_0 artifact is eligible:

- Checkpoint: 19,000.
- Local model: `models/gguf/v1-ckpt19000-q8_0/model-q8_0.gguf`.
- Size: 1,834,426,048 bytes.
- SHA-256: `d421e331a5b9920ebf99eebf9902b4a2db535a3505b623042b6ab1f230e02858`.
- Native revision: `38a5b42d9a3e82e0a586bcd1caed121f36c87a73`.
- Frozen validation corpus: `data/processed/stages/smoke/valid.jsonl`, 100 records.
- Corpus SHA-256: `49f1df7ba0826f913a364aebf155441a7e5e58b574adb212ee4dec59a2d14042`.
- Mac quality reference: `evaluation/results/v1-gguf-q8_0-smoke-valid/predictions.jsonl`.

Q4, Q5, and imatrix-Q4 remain excluded because they failed the approved quality
gates. The [full Q8 test](../../docs/decisions/0007-q8-full-test-review.md) is
completed evidence, not a tuning corpus. Do not rerun or optimize against its
2,000 records during benchmark preparation.

## Gate A — Obtain hardware and authority

1. Obtain the human's explicit Q8-only benchmark/transfer approval and exact
   target host/user. Do not infer a target from SSH configuration or scan the LAN.
2. Review a fresh diagnostic snapshot and the human-supplied PSU/cooling/storage
   details. The original hardware target is a Pi 4; another model needs a recorded
   plan change. Architecture must be 64-bit ARM with a 64-bit Python process.
3. Define and approve minimum available RAM, minimum free disk, maximum temperature,
   swap-growth limits, and sustained-test duration for this device. Unknown
   telemetry blocks a claim of readiness. Do not infer that model file size alone
   is the peak RAM requirement.
4. Budget disk for the Q8 artifact, native build/environment, bounded reports,
   temporary transfer data if used, and the agreed reserve. Do not delete unrelated
   device data, resize swap, overclock, reboot, or change firmware to make it fit.
5. Record approval and limits in a new versioned benchmark protocol before any
   model execution. This draft does not satisfy that gate.

## Gate B — Build and verify, only after approval

1. Inspect existing dependencies read-only; obtain explicit authority for missing
   OS packages. Keep MLX and PyTorch out of the Pi environment.
2. Build the pinned source on the Pi as a CPU-only Release build. Mac Metal
   binaries are not portable to Linux. Record compiler/CMake versions, flags,
   CPU features, source cleanliness, build duration, and binary/library hashes.
   Do not change the native revision without revalidation.
3. Transfer only approved code, Q8 weights, tokenizer files, and the frozen
   synthetic validation corpus needed for the benchmark. Use the explicitly
   approved SSH target or a human-managed offline copy; verify the complete model
   and corpus hashes on the Pi. Do not transfer the full test, calibration corpus,
   private review samples, or real documents.
4. Use a new runtime implementation/configuration and new result directories.
   Never edit the frozen Mac runner or force its resume checks to accept a Pi run.

## Gate C — Proposed CPU runtime and quality check

The initial proposal is one CPU inference slot, four threads, context 2,048,
batch/ubatch 512, f16 KV cache, seed 42, temperature zero, non-thinking prompts,
and the original 384-token output cap. CPU-only execution and CPU attention
settings must be explicitly recorded and verified against the pinned build.

Explicitly disabling the separate RAM prompt-state cache (`--cache-ram 0`) is
proposed for this **new** runtime. The pinned source defaults it to 8,192 MiB,
separate from per-request `cache_prompt=false`. That is unsuitable as an
unexamined default for a constrained Pi experiment. This change has **not** been
applied or validated; approval and a new validation run are required. Preserve
the completed Mac experiment unchanged.

Before measurements:

1. Verify model identity, private loopback binding, disabled UI/proxy, context,
   EOS, and every validation prompt's template and token IDs.
2. Run the 100 validation records with the fixed configuration and strict scorer.
   Compare against the saved Mac Q8 validation reference, documenting output and
   per-class changes. The proposed gate retains at most 0.01 absolute recall and
   complete-document-recall drop, with no schema-validity decline; it needs to be
   frozen in the approved Pi protocol before execution.
3. Stop for prompt/context mismatch, generation failure, exceeded resource limits,
   or a failed quality gate. Do not repair outputs, adjust labels, enlarge output
   budgets, or choose new settings using the frozen test.

These cross-platform task-score checks are not bitwise numerical equivalence.
The benchmark runner and watchdog still need implementation once the hardware,
resource policy, and experiment are approved.

## Gate D — Measure and report

Proposed sequence, to be frozen before execution:

1. Record idle baseline counters and startup/load time separately.
2. Perform one unscored warm-up pass, then three measured passes over the same
   validation records in the same order. Record output hashes, not just speed.
3. Record prompt and decode throughput separately, request latency distribution,
   peak process RSS, system memory/swap, temperature, firmware flags, interruptions,
   and runtime/build/input hashes. Check clock consistency before reporting rates.
4. Run a bounded, approved sustained workload under an owned-process watchdog.
   Historical firmware flags require baseline comparison; active power/thermal
   flags, unknown monitoring state, or breached resource limits stop the run.
   Do not disable throttling or automatic protections.
5. Produce an aggregate report without source text. Retain raw predictions and
   counters locally. A stopped run must be labeled incomplete, never published as
   a completed benchmark.

These are single-chunk model measurements. Long-document, chunking, reconciliation,
API, and end-to-end redaction benchmarks cannot be claimed before those components
exist and pass their own gates. Concurrency remains one; no systemd service, LAN
endpoint, startup persistence, or production document handling is part of this
benchmark.

## Human exit decision

Review model errors, actual Pi quality/performance/resource behavior, and decide
H-046 production-model eligibility. Benchmark authorization is not deployment
approval. Pi transfer/build/execution have not happened during this preparation
slice, and G7/G10 remain open.
