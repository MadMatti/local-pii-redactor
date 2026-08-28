# Milestones and task register

This is the primary execution tracker. Detailed acceptance criteria and task
descriptions are in the linked workstream plans. Complete milestones in order
unless the dependency notes explicitly allow parallel work.

## Milestone 0 — Charter and policy freeze

**Goal:** eliminate ambiguity about labels, privacy, deployment exposure, and
success criteria.

### Human tasks

- [ ] H-001 through H-008: taxonomy, boundaries, prompt, and placeholders.
- [ ] H-010 through H-016: retention, network, authentication, limits, display,
  and warnings.
- [ ] H-020 through H-025: hardware and account inventory.
- [ ] H-040 through H-043: metric hierarchy and quality tolerances.

### Coding/documentation tasks

- [ ] Create `docs/annotation-policy.md`.
- [ ] Create `docs/privacy-model.md`.
- [ ] Create `docs/acceptance-criteria.md`.
- [ ] Create the decision-log format.

### Exit evidence

- Approved annotation and privacy policies.
- Approved V1 scope and explicit V2 exclusions.
- Recorded Mac and Pi hardware profile.
- Written stop/go criteria for training, quantization, and chunking.

**Gate G0:** The human owner approves dataset construction.

---

## Milestone 1 — Repository and environment

**Depends on:** G0 for policy-sensitive constants; basic scaffolding may start
before G0.

### Coding tasks

- [ ] C-001 through C-008: structure, dependencies, automation, CI, and guards.
- [ ] C-010 through C-015: configuration, manifests, diagnostics, and pinning.
- [ ] C-020 through C-025: shared domain schemas.

### Human tasks

- [ ] H-020 and H-021: verify/install Mac prerequisites.
- [ ] H-022: verify Hugging Face access and terms.

### Exit evidence

- Environment diagnostic report.
- Reproducible dependency files.
- Passing model-independent CI.
- Successful untouched-Qwen smoke inference.
- Initial repository commit without model binaries.

**Gate G1:** Base model, tokenizer, MLX, and repository automation work.

---

## Milestone 2 — Dataset discovery and mapping

**Depends on:** G0 and G1.

### Coding tasks

- [x] ML-001 through ML-006: source pinning, download, statistics, defects, and
  review bundle.
- [x] Draft the complete source-to-canonical label mapping.

### Human tasks

- [x] H-030 and H-031: inspect source and excluded-label examples.
- [ ] Reconfirm H-001 through H-006 against actual source labels.

### Exit evidence

- Source inspection report.
- Human-approved label mapping covering every observed source label.
- Dataset source and license recorded.

**Gate G2:** Approved 2026-08-28. Mapping is frozen for dataset version 1.

---

## Milestone 3 — Dataset construction and validation

**Depends on:** G2.

### Coding tasks

- [x] ML-010 through ML-017: conversion, stable IDs, split grouping, manifest.
- [x] ML-020 through ML-025: negatives, hard negatives, long context, repeats,
  rare classes, and calibration data.
- [x] ML-030 through ML-037: full validation and statistics.

### Human tasks

- [x] H-032 through H-036: review synthetic data, distribution, and split
  isolation.

### Exit evidence

- [x] Smoke, V0, and V1-candidate dataset files.
- [x] Separate calibration corpus.
- [x] Passing validation report.
- [x] Dataset statistics and manifest with hashes.

**Gate G3:** Approved 2026-08-28 in decision record 0001. Training remains
prohibited if any validator failure exists.

---

## Milestone 4 — Evaluator and frozen baselines

**Depends on:** G3.

### Coding tasks

- [ ] ML-040 through ML-046: exact metrics, regex, base Qwen, and frozen
  artifacts.
- [ ] QA-010 through QA-015: data/schema/evaluator-support tests.
- [ ] QA-040 through QA-045: parser, metric, prompt, and result tests.

### Human tasks

- [ ] H-040 through H-043: finalize selection thresholds using baseline evidence.
- [ ] Review initial base and regex false positives/negatives.

### Exit evidence

- Regex report.
- Untouched-Qwen report.
- Frozen test-set and baseline hashes.
- Verified evaluator arithmetic.

**Gate G4:** Metrics and baseline artifacts are frozen before meaningful QLoRA.

---

## Milestone 5 — QLoRA smoke run

**Depends on:** G4.

### Coding tasks

- [ ] ML-050 through ML-054: configuration, run, adapter verification, and
  diagnostic evaluation.

### Human tasks

- [ ] H-050 and H-051: prepare and observe the Mac.
- [ ] H-044: approve moving to longer training.

### Exit evidence

- Finite-loss training log.
- Resource report.
- Loadable smoke adapter.
- Successful inference using production-equivalent chat formatting.

**Gate G5:** The complete training pipeline works without unacceptable resource
pressure.

---

## Milestone 6 — Main training and adapter selection

**Depends on:** G5.

### Coding tasks

- [ ] ML-060 through ML-067: staged experiments, checkpoint evaluation, error
  bundles, and selection record.

### Human tasks

- [ ] H-052 through H-054: authorize and supervise runs.
- [ ] H-045: approve the selected adapter.
- [ ] H-070 through H-073: conduct model error analysis and approve corrections.

### Exit evidence

- Comparable experiment manifests.
- Base-versus-adapter report.
- Per-class and complete-document metrics.
- Human-approved model-selection record.

**Gate G6:** Fine-tuned model materially improves the task. If not, return to
Milestone 2 or 3 rather than continuing.

---

## Milestone 7 — Fusion, GGUF, and quantization

**Depends on:** G6.

### Coding tasks

- [ ] ML-070 through ML-075: fusion, conversion, metadata, and parity.
- [ ] ML-080 through ML-086: Q8/Q5/Q4/imatrix generation and evaluation.

### Human tasks

- [ ] Review disk requirements and authorize `llama.cpp` setup.
- [ ] H-042: apply the approved quantization tolerance.

### Exit evidence

- Fused MLX and BF16 GGUF manifests/checksums.
- Conversion-parity report.
- Quantization comparison report.
- Q4 and Q5 Pi candidates.

**Gate G7:** No unexplained conversion regression; selected candidates meet the
quality threshold.

---

## Milestone 8 — Deterministic document engine

**Depends on:** G2 for schema/policy. This milestone may run in parallel with
Milestones 5–7 using fake tokenizer/model clients.

### Coding tasks

- [ ] DOC-001 through DOC-005: tokenizer abstraction and parity.
- [ ] DOC-010 through DOC-026: lossless segmentation and chunking.
- [ ] DOC-030 through DOC-037: alignment.
- [ ] DOC-040 through DOC-047: reconciliation.
- [ ] DOC-050 through DOC-055: deterministic redaction.
- [ ] DOC-060 through DOC-066: complete sequential processor.
- [ ] QA-020 through QA-039: unit and property tests.

### Human tasks

- [ ] Approve overlap, conflict, and placeholder policies.

### Exit evidence

- Passing deterministic unit/property tests.
- Fake-client end-to-end result.
- Proof that only approved spans change.

**Gate G8:** Document engine is correct independently of model quality.

---

## Milestone 9 — Long-document evaluation

**Depends on:** G7 and G8.

### Coding tasks

- [ ] DOC-070 through DOC-074: generators, boundary cases, chunk matrix, and
  metrics.
- [ ] QA-060 through QA-063: corpus validation and benchmark execution.

### Human tasks

- [ ] H-043: apply boundary-recall tolerance.
- [ ] H-047: select chunk and overlap defaults.
- [ ] H-070 through H-073: review boundary and application failures.

### Exit evidence

- Boundary recall report.
- Chunking matrix with calls, latency, duplicates, conflicts, and recall.
- Human-approved production defaults.

**Gate G9:** Long-document behavior meets approved quality and cost trade-offs.

---

## Milestone 10 — Raspberry Pi CLI deployment

**Depends on:** G7; may begin before G9 is complete.

### Coding tasks

- [ ] PI-001 through PI-024: diagnostics, dependencies, build, and environment.
- [ ] PI-030 through PI-044: transfer, checksum, and CLI validation.
- [ ] PI-050 through PI-055: Q4/Q5 comparison and selection report.
- [ ] QA-080 through QA-086: Pi benchmark framework.

### Human tasks

- [ ] H-023, H-024, and H-060 through H-064: hardware/OS preparation and tests.
- [ ] H-046: approve the deployment quantization.

### Exit evidence

- Pi environment report.
- Working Q4 and Q5 CLI inference.
- Pi quality/performance/thermal comparison.
- Approved production model and checksum.

**Gate G10:** Selected model is stable and acceptable on the actual Pi.

---

## Milestone 11 — Model server and API

**Depends on:** G8, G9, and G10.

### Coding tasks

- [ ] APP-001 through APP-016: structured model/tokenizer clients.
- [ ] APP-020 through APP-036: FastAPI, progress, cancellation, and limits.
- [ ] APP-040 through APP-047: privacy and network controls.
- [ ] APP-060 through APP-062: safe observability.
- [ ] QA-050 through QA-058: API and privacy tests.

### Human tasks

- [ ] H-010 through H-016: reconfirm production privacy/network decisions.

### Exit evidence

- Passing fake- and real-server integration tests.
- Readiness correctly reflects the model service.
- Canary content absent from logs.
- Long request progress and failure behavior verified.

**Gate G11:** Backend is functional, bounded, and privacy-reviewed.

---

## Milestone 12 — Web UI

**Depends on:** G11 API contracts.

### Coding tasks

- [ ] APP-050 through APP-058: paste/upload, progress, result, errors,
  accessibility, and local assets.

### Human tasks

- [ ] Approve UI wording and responsive behavior.
- [ ] H-015 and H-016: approve entity display and warnings.

### Exit evidence

- Desktop/mobile manual checks.
- No remote frontend requests or browser content persistence.
- Successful paste, upload, progress, copy, and download flows.

**Gate G12:** Offline UI is usable and communicates residual risk.

---

## Milestone 13 — systemd and operational deployment

**Depends on:** G11 and G12.

### Coding tasks

- [ ] PI-060 through PI-075: LLM and app units.
- [ ] PI-080 through PI-084: install, update, rollback, uninstall, and recovery.
- [ ] PI-090 through PI-096: reboot, isolation, offline, recovery, sustained, and
  remanence tests.

### Human tasks

- [ ] H-065 through H-067: install services, cold reboot, and LAN tests.

### Exit evidence

- Cold-reboot success.
- Offline redaction success.
- Model port isolated from LAN.
- Recovery/update/rollback instructions validated.

**Gate G13:** The Pi behaves as an appliance rather than a manual demo.

---

## Milestone 14 — Final experimental evaluation

**Depends on:** G13.

### Coding tasks

- [ ] QA-070 through QA-086: full model, chunking, end-to-end, and Pi benchmark.
- [ ] Generate final machine-readable and Markdown reports.

### Human tasks

- [ ] H-070 through H-073: final false-negative/false-positive review.
- [ ] Confirm all approved thresholds are satisfied or explicitly document a
  release exception.

### Exit evidence

- Final model comparison.
- Final chunking comparison.
- Final Pi performance and sustained-run report.
- Categorized error analysis.

**Gate G14:** Final claims are directly supported by reproducible evidence.

---

## Milestone 15 — Documentation and release

**Depends on:** G14.

### Coding/documentation tasks

- [ ] C-090 through C-095: workflows, architecture, troubleshooting, checksums,
  and validation.
- [ ] REL-001 through REL-012: project documentation.
- [ ] REL-020 through REL-028: model card.
- [ ] REL-030 through REL-038: release validation and tagging.

### Human tasks

- [ ] H-080 through H-085: license, history, artifact, documentation, and final
  publication approvals.

### Exit evidence

- Source repository release.
- Adapter release with model card.
- GGUF release with checksums and Pi recommendation.
- Complete limitations and reproducibility documentation.

**Gate G15:** Project is publicly releasable and reproducible.

---

## Summary dependency graph

```text
G0 policy
 └─ G1 environment
     └─ G2 source mapping
         └─ G3 dataset
             └─ G4 evaluator/baselines
                 └─ G5 smoke QLoRA
                     └─ G6 selected adapter
                         └─ G7 GGUF/quantization ─────┐
                                                    │
G2 ── G8 deterministic document engine ── G9 long docs
                                                    │
G7 ── G10 Pi CLI/model selection ───────────────────┤
                                                    ▼
                                          G11 API/server
                                                    │
                                                    ▼
                                                G12 UI
                                                    │
                                                    ▼
                                           G13 deployment
                                                    │
                                                    ▼
                                           G14 final report
                                                    │
                                                    ▼
                                             G15 release
```

## Suggested execution rhythm

At the end of every milestone:

1. Save the required evidence under a uniquely identified run or report path.
2. Run the milestone's automated checks.
3. Review failures and update the risk register.
4. Record decisions that change policy, configuration, or acceptance thresholds.
5. Obtain the required human approval.
6. Mark the gate complete before starting dependent work.
