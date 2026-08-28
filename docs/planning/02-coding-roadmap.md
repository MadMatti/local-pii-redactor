# Coding roadmap

This is the repository-wide implementation backlog. Detailed ML, document,
application, deployment, and test requirements live in their respective plans.
Tasks here describe the build order, common infrastructure, and component
boundaries.

## 1. Repository foundation

- [x] **C-001 — Create the source layout.** Add package directories for schema,
  inference, baselines, document processing, redaction, API, and web assets.
- [x] **C-002 — Create data and artifact directories.** Add placeholders or
  README files for raw, intermediate, processed, calibration, evaluation,
  adapter, fused, and GGUF artifacts while keeping large outputs ignored.
- [ ] **C-003 — Add project metadata.** Create `pyproject.toml`, package metadata,
  Python-version constraints, and console-script entry points where useful.
- [ ] **C-004 — Split dependencies.** Maintain separate Mac training, Pi
  runtime, and development dependencies. Avoid MLX and training-only packages
  in the Pi environment.
- [x] **C-005 — Strengthen `.gitignore`.** Exclude virtual environments, caches,
  raw/generated data, adapters, fused models, GGUFs, logs, local configuration,
  and common editor files.
- [ ] **C-006 — Add task automation.** Provide documented commands for setup,
  formatting, linting, unit tests, data validation, evaluation, and local app
  startup.
- [ ] **C-007 — Add CI.** Run format/lint/type checks and tests that do not need
  Apple Silicon, a downloaded model, or a Raspberry Pi.
- [ ] **C-008 — Add large-file and secret checks.** Fail CI or pre-commit checks
  if model binaries, credentials, or likely private inputs are staged.

### Foundation acceptance criteria

- A clean clone can create a development environment using documented commands.
- Importing `pii_redactor` succeeds.
- CI can run without model downloads.
- Large generated artifacts cannot be accidentally committed through normal
  workflows.

## 2. Configuration and reproducibility

- [ ] **C-010 — Centralize immutable task constants.** Store the canonical system
  prompt, allowed entity types, output JSON schema, and prompt version.
- [ ] **C-011 — Create application settings.** Define model-server URL, context
  size, chunk target, overlap, timeouts, concurrency, upload limits, and logging
  settings in typed configuration.
- [ ] **C-012 — Support environment overrides.** Use documented environment
  variables without requiring secrets for a normal offline installation.
- [ ] **C-013 — Add run manifests.** Every dataset build, evaluation, training
  run, conversion, quantization, and benchmark records configuration, source
  revisions, Git commit, seed, timestamp, and artifact hashes.
- [ ] **C-014 — Add environment diagnostics.** Report platform, architecture,
  Python and package versions, MLX availability, disk, memory, and configured
  artifact paths.
- [ ] **C-015 — Pin external revisions.** Record Hugging Face model/dataset
  revisions and the exact `llama.cpp` commit.

## 3. Core domain schema

- [x] **C-020 — Implement `EntityType`.** Use a string enum containing only the
  approved V1 labels.
- [ ] **C-021 — Implement model response types.** Validate a list of `{type,
  text}` objects with nonempty text.
- [ ] **C-022 — Implement chunk types.** Store chunk ID, character start/end,
  exact source slice, and token count.
- [ ] **C-023 — Implement detection types.** Store entity type, exact text,
  global start/end, source chunk, and optional boundary-quality metadata.
- [ ] **C-024 — Implement result types.** Represent redacted text, reconciled
  detections, warnings, chunk counts, timings, and non-sensitive diagnostics.
- [ ] **C-025 — Add schema serialization tests.** Verify stable JSON output and
  rejection of invalid labels, offsets, and empty entity values.

## 4. Data and model implementation

Execute the detailed tasks in [03-data-and-ml-plan.md](03-data-and-ml-plan.md):

- dataset download and inspection;
- label normalization and controlled synthetic examples;
- deterministic dataset validation;
- regex and base-Qwen baselines;
- smoke and staged QLoRA training;
- adapter evaluation and selection;
- fusion, GGUF conversion, parity validation;
- Q8/Q5/Q4 and importance-matrix quantization.

Model work must not bypass dataset or evaluation gates.

## 5. Deterministic document pipeline

Execute [04-document-processing-plan.md](04-document-processing-plan.md) using a
fake model client before integrating the final model:

- tokenizer abstraction;
- paragraph/sentence segmentation with exact character offsets;
- token-aware overlapping chunking;
- exact substring alignment;
- local-to-global span conversion;
- duplicate/conflict reconciliation;
- immutable right-to-left redaction;
- sequential end-to-end processing and progress callbacks.

The pipeline must remain independent of MLX and `llama.cpp` implementation
details.

## 6. Serving and user interface

Execute [05-api-ui-privacy-plan.md](05-api-ui-privacy-plan.md):

- typed `llama-server` client;
- tokenizer endpoint client;
- one-generation-at-a-time concurrency control;
- FastAPI health, readiness, redaction, upload, and progress endpoints;
- Jinja2/HTML/CSS/JavaScript UI;
- request limits, safe errors, and content-free operational logging;
- no third-party browser dependencies.

## 7. Raspberry Pi packaging

Execute [06-raspberry-pi-deployment-plan.md](06-raspberry-pi-deployment-plan.md):

- Pi-only requirements;
- repeatable `llama.cpp` build instructions;
- configuration templates;
- systemd services;
- model checksum verification;
- deployment verification and rollback instructions.

## 8. Test and benchmark automation

Execute [07-testing-and-benchmark-plan.md](07-testing-and-benchmark-plan.md):

- deterministic unit fixtures;
- fake-server integration tests;
- optional real-model markers;
- long-document and chunk-boundary generators;
- model-quality comparison;
- Pi performance and thermal measurements;
- machine-readable and Markdown reports.

## 9. Documentation and release preparation

- [ ] **C-090 — Document all normal workflows.** A contributor should not need
  to reconstruct commands from shell history.
- [ ] **C-091 — Generate example configuration.** Include safe defaults and
  explain every setting that affects privacy, memory, or quality.
- [ ] **C-092 — Create architecture diagrams.** Cover training, packaging,
  runtime data flow, trust boundaries, and long-document processing.
- [ ] **C-093 — Create troubleshooting guidance.** Cover MLX memory failures,
  invalid model output, conversion divergence, Pi out-of-memory, slow startup,
  thermal throttling, and service startup failures.
- [ ] **C-094 — Add artifact verification.** Publish checksums and a script or
  command for verifying downloads.
- [ ] **C-095 — Add release validation.** Check documentation links, ignored
  artifacts, licenses, required files, tests, and benchmark-result presence.

## Recommended implementation order

```text
C-001..C-015  repository and reproducibility
       ↓
C-020..C-025  shared schemas
       ↓
ML data construction + QA evaluation framework
       ↓
base-model baseline
       ↓
smoke QLoRA → staged QLoRA → adapter selection
       ↓
fusion → BF16 parity → quantization
       ↓
document engine with fake client
       ↓
Pi CLI benchmark and deployment-model selection
       ↓
llama-server client → FastAPI → UI
       ↓
systemd deployment → reboot test
       ↓
final benchmark → release validation
```

The document engine may be developed in parallel with model experiments once
the schema and annotation policy are frozen. UI work should wait until the API
response and progress contracts are stable.

## Coding definition of done

- Public modules have clear contracts and type hints.
- Deterministic logic is covered by fast unit tests.
- Model-dependent tests are explicitly marked and never silently skipped in a
  release-validation run.
- Generated artifacts carry manifests and checksums.
- Error messages do not expose source document text.
- Logs do not contain user content or model output.
- All configuration affecting quality or privacy is documented.
- The codebase can be installed separately for Mac training and Pi runtime.
