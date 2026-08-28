# Release and risk plan

## 1. Release artifacts

### Source-code repository

Include:

- dataset preparation and validation scripts;
- annotation and privacy policies;
- training configurations and run-manifest format;
- evaluation and benchmark scripts;
- deterministic document-processing code;
- FastAPI backend and local frontend;
- Raspberry Pi setup documentation and systemd units;
- architecture diagrams, benchmark summaries, and limitations;
- tests that do not require proprietary or unpublished data.

Exclude:

- raw or generated large datasets unless deliberately released separately;
- user documents or real PII;
- credentials, tokens, hostnames, or private network details;
- MLX model downloads, adapters, fused models, and GGUFs;
- local virtual environments, caches, and benchmark scratch files.

### Adapter repository

Publish separately:

- LoRA adapter weights and configuration;
- exact base model and revision;
- prompt version;
- training configuration and dataset description;
- metrics and known limitations;
- license and attribution;
- example MLX loading command;
- checksums.

### GGUF repository

Publish selected variants rather than every experimental artifact:

- recommended Pi quantization;
- one higher-quality comparison quantization if useful;
- optional importance-matrix variant when it demonstrates value;
- model source and conversion lineage;
- exact `llama.cpp` conversion/quantization commit and commands;
- Pi context/thread recommendation and benchmark table;
- checksums and file sizes.

## 2. Documentation backlog

- [ ] **REL-001 — Write the main README.** Lead with the working application,
  then architecture, quick start, results, limitations, and development links.
- [ ] **REL-002 — Write the annotation policy.** Include positive, negative,
  ambiguous, nested, and repeated examples.
- [ ] **REL-003 — Write the privacy model.** Explain local processing, trust
  boundaries, retention, logging, LAN exposure, and residual risks.
- [ ] **REL-004 — Write dataset documentation.** Cover source, license,
  transformations, custom templates, split isolation, statistics, and hashes.
- [ ] **REL-005 — Write training documentation.** Cover Mac requirements,
  QLoRA rationale, configs, memory controls, checkpoint selection, and resume.
- [ ] **REL-006 — Write model-packaging documentation.** Explain fusion,
  dequantization limits, GGUF conversion, parity checks, quantization, and
  importance-matrix calibration.
- [ ] **REL-007 — Write application documentation.** Cover chunking, alignment,
  reconciliation, API, UI, configuration, and safe operational behavior.
- [ ] **REL-008 — Write Pi deployment documentation.** Cover OS, build, model
  transfer, services, reboot test, update, rollback, and troubleshooting.
- [ ] **REL-009 — Publish benchmark methodology.** Include data versions,
  settings, hardware, warm-up, repetitions, metrics, and raw-result layout.
- [ ] **REL-010 — Write explicit limitations.** State that missed PII is possible
  and the tool must not be the sole control for production-sensitive data.
- [ ] **REL-011 — Create contributor guidance.** Explain safe test data, artifact
  exclusions, tests, style, and reporting privacy defects.
- [ ] **REL-012 — Create security reporting guidance.** Provide a private contact
  path if the project is published publicly.

## 3. Model-card checklist

- [ ] **REL-020 — Identify the base model.** Include official and MLX training
  checkpoints plus revisions.
- [ ] **REL-021 — Describe the task.** Structured English PII extraction with
  exact source strings and canonical labels.
- [ ] **REL-022 — Describe training.** QLoRA, Apple Silicon hardware, dataset
  versions, sample counts, sequence lengths, and selected configuration.
- [ ] **REL-023 — Describe evaluation.** Frozen test set, strict matching,
  complete-document recall, per-class results, and quantization deltas.
- [ ] **REL-024 — Describe intended use.** Local experimentation and assisted
  redaction with human awareness of errors.
- [ ] **REL-025 — Describe limitations.** English-only V1, synthetic-data bias,
  false negatives/positives, context/chunking limitations, and no compliance
  guarantee.
- [ ] **REL-026 — Describe deployment.** `llama.cpp`, supported quantizations,
  Pi settings, memory, latency, and thermal results.
- [ ] **REL-027 — Document lineage.** Base → QLoRA adapter → fused checkpoint →
  BF16 GGUF → deployment quantizations.
- [ ] **REL-028 — Document licenses.** Verify the terms as they exist at release
  time rather than relying only on the original project brief.

## 4. Release validation

- [ ] **REL-030 — Run the complete test suite.** Include explicitly requested
  real-model and Pi tests; do not rely only on default CI markers.
- [ ] **REL-031 — Regenerate final reports.** Use the release-candidate commit and
  final artifact hashes.
- [ ] **REL-032 — Validate documentation links and commands.** Test quick starts
  from clean environments where practical.
- [ ] **REL-033 — Scan Git history.** Check for credentials, private text, large
  binaries, model files, and generated datasets.
- [ ] **REL-034 — Verify artifact checksums.** Test downloaded copies, not only
  local originals.
- [ ] **REL-035 — Verify offline operation.** Run the published deployment without
  internet access after dependencies and models are installed.
- [ ] **REL-036 — Verify license and notice files.** Include required third-party
  attribution.
- [ ] **REL-037 — Tag the release.** Record source commit, adapter hash, model
  hashes, data manifest, and result manifest in release notes.
- [ ] **REL-038 — Preserve reproducibility records.** Keep immutable copies of
  final configuration and benchmark evidence.

## 5. Risk register

| ID | Risk | Likelihood | Impact | Mitigation | Trigger/owner |
| --- | --- | --- | --- | --- | --- |
| R-01 | M1 8 GB runs out of memory | High | High | Batch 1, checkpointing, fewer layers, shorter sequence, staged runs | Memory pressure or swap; coding + human |
| R-02 | Fine-tuning does not beat base Qwen | Medium | High | Freeze baselines, inspect mapping/data/prompt, balance classes, stop before packaging | Adapter gate fails; ML |
| R-03 | Generative model underperforms conventional NER | Medium | Medium | Preserve regex baseline, optionally add GLiNER, report honestly, consider hybrid V2 | Final comparison; human decision |
| R-04 | Synthetic template leakage inflates metrics | High | High | Group by template, check near duplicates, freeze manifests | Split validator warning; ML |
| R-05 | Rare labels have weak recall | High | High | Per-class metrics, targeted controlled examples, manual review | Per-class report; ML + human |
| R-06 | Model emits malformed or extra output | Medium | High | Stable prompt, thinking disabled, schema-constrained decoding, Pydantic validation | Validity threshold fails; APP |
| R-07 | Model hallucinates a PII value | Medium | High | Exact substring alignment; reject absent text | Alignment diagnostic; DOC |
| R-08 | Repeated identical text is misaligned | Medium | High | Source-order training, occurrence tracking, ambiguity diagnostics, targeted tests | Repeated fixture failure; DOC |
| R-09 | Chunk boundary causes a missed entity | High | High | Overlap, sentence-aware packing, torture tests, boundary metrics | Boundary recall gate; DOC/QA |
| R-10 | Reconciliation redacts the wrong overlap | Medium | High | Deterministic IoU/context policy, provenance, non-overlap assertion | Conflict regression; DOC |
| R-11 | Quantization damages recall | Medium | High | Compare all formats from common BF16 parent; prefer Q5 when justified | Tolerance gate fails; ML/human |
| R-12 | Conversion changes tokenizer/chat behavior | Medium | High | Pin llama commit, inspect metadata, parity evaluation | BF16 parity gate fails; ML |
| R-13 | Pi model does not fit or swaps heavily | Medium | High | Carry Q4/Q5, reduce context, Pi-only dependencies, measure actual RAM | CLI/load benchmark; PI |
| R-14 | Pi throttles during long documents | High | Medium | Cooling, sustained thermal test, sequential inference | Throttling flags/temperature; human/PI |
| R-15 | Multiple requests exhaust the Pi | Medium | High | Semaphore one, bounded queue, request limits, busy response | Load test; APP |
| R-16 | PII leaks into logs or temp files | Medium | Critical | Allowlisted logs, no content logging, memory processing, canary tests | Privacy test; APP/QA |
| R-17 | Unauthenticated LAN user submits data | Medium | High | Human-approved trust model, firewall/auth/TLS options | Network review; human |
| R-18 | Model endpoint is exposed directly | Low | High | Bind `127.0.0.1`, port-isolation test | Deployment test; PI |
| R-19 | Version drift breaks commands/endpoints | High | Medium | Pin revisions, capture manifests, revalidate at implementation/release | Dependency update; coding |
| R-20 | License assumptions are stale or incomplete | Medium | High | Recheck primary licenses and redistribution terms at release | Release review; human |
| R-21 | Users mistake output for guaranteed anonymization | Medium | Critical | Prominent warning, limitations, intended-use language | UI/model-card review; human/REL |
| R-22 | Huge input causes denial of service | Medium | High | File limits, queue bounds, timeout, future streaming | API load test; APP |
| R-23 | Generated results overwrite baselines | Medium | Medium | Immutable run directories and manifest hashes | Evaluator test; QA |
| R-24 | Real PII enters development fixtures | Low | Critical | Synthetic-only fixtures, contributor policy, secret/content scan | Review or scan; human/REL |

## 6. Release decision rules

Do not publish a production recommendation when:

- required license checks are incomplete;
- model/data lineage cannot be reproduced;
- the selected quantization exceeds the approved recall loss;
- privacy canary values appear in logs or persistent browser/server state;
- the Pi sustained benchmark shows crashes, uncontrolled swap, or unacceptable
  thermal throttling;
- the UI or README implies guaranteed anonymization;
- test or calibration leakage is suspected;
- conversion-parity failure remains unexplained.

It is acceptable to publish an explicitly experimental result with weak model
metrics if the measurements and limitations are accurate. It is not acceptable
to hide or relabel a failed gate.

## 7. V2 backlog

- Conventional NER/GLiNER production comparison.
- Hybrid regex and LLM merge policies.
- Cheap prefilter to skip clearly safe chunks.
- True generator-based constant-memory streaming.
- Multilingual datasets and per-language metrics.
- PDF/DOCX reconstruction and redaction.
- OCR and image support.
- Pseudonymization and reversible mappings.
- Contextual second-pass conflict adjudication.
- Authenticated multi-user scheduling.
