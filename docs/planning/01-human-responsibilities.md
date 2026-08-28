# Human responsibilities plan

This file lists work that requires the project owner's judgment, physical
access, credentials, privileged system operations, or explicit approval. Coding
tasks may prepare commands and evidence, but should not silently make these
decisions.

## 1. Product and annotation decisions

- [ ] **H-001 — Approve the V1 entity taxonomy.** Confirm every included label
  and explicitly list excluded labels such as generic city, country, date,
  company, and job title.
- [ ] **H-002 — Decide name boundaries.** Decide whether titles, initials,
  suffixes, partial names, and signatures are included.
- [ ] **H-003 — Decide address boundaries.** Decide whether an address includes
  recipient name, unit, postcode, city, region, and country when they appear in
  one continuous phrase.
- [ ] **H-004 — Decide identifier context rules.** Define when a generic number
  becomes a bank, customer, employee, medical, national, or vehicle identifier.
- [ ] **H-005 — Decide nested-span policy.** Choose whether the widest span,
  narrowest span, or a label-priority rule wins when annotations overlap.
- [ ] **H-006 — Approve exact-text rules.** Confirm that source punctuation,
  capitalization, spacing, and repeated occurrences are preserved exactly.
- [ ] **H-007 — Approve the canonical system prompt.** Changes after dataset
  generation require regenerating affected artifacts or a documented exception.
- [ ] **H-008 — Approve placeholder style.** The V1 recommendation is typed
  placeholders such as `[EMAIL]` and `[PERSON_NAME]`.

Evidence to save:

- `docs/annotation-policy.md`
- label-mapping table
- at least one positive and one hard-negative example for every label
- dated decision entries for ambiguous cases

## 2. Privacy and access decisions

- [ ] **H-010 — Approve the data-retention policy.** Default: no documents,
  prompts, model outputs, or detected PII are persisted.
- [ ] **H-011 — Decide network scope.** Choose localhost-only or trusted-LAN
  access for FastAPI. Keep `llama-server` on `127.0.0.1`.
- [ ] **H-012 — Decide authentication.** If the UI is available to the LAN,
  explicitly accept an unauthenticated trusted-LAN model or require an
  authentication mechanism.
- [ ] **H-013 — Decide transport protection.** Determine whether plain local
  HTTP is acceptable or whether a local TLS reverse proxy is required.
- [ ] **H-014 — Set upload limits.** Choose maximum file size, maximum document
  characters/tokens, timeout behavior, and queue limits.
- [ ] **H-015 — Decide entity-display behavior.** Choose full values, masked
  previews, or reveal-on-demand in the browser.
- [ ] **H-016 — Approve privacy and limitation notices.** The UI and README must
  state that false positives and false negatives are possible.

Evidence to save:

- `docs/privacy-model.md`
- approved network diagram
- approved request and file limits
- final warning text

## 3. Hardware and account preparation

- [ ] **H-020 — Verify the Mac.** Record macOS version, `arm64` architecture,
  free storage, Python version, and available memory.
- [ ] **H-021 — Install Mac prerequisites.** Install Xcode command-line tools,
  Homebrew if required, Git, Python, and CMake.
- [ ] **H-022 — Verify Hugging Face access.** Log in if needed and review model
  and dataset terms before download or publication.
- [ ] **H-023 — Inventory the Raspberry Pi.** Record Pi revision, RAM, storage,
  power supply, cooling, and network connection.
- [ ] **H-024 — Install 64-bit Raspberry Pi OS.** Configure hostname, user,
  locale, time zone, SSH, and network access.
- [ ] **H-025 — Prepare release accounts.** Confirm GitHub and Hugging Face
  repositories and who owns them.

## 4. Dataset review and approval

- [x] **H-030 — Review raw dataset examples.** Review at least 100 random
  records plus samples from every rare source label.
- [x] **H-031 — Review excluded labels.** Verify that removal of generic dates,
  cities, countries, companies, and titles matches the policy.
- [x] **H-032 — Review negative templates.** Verify that templates are genuinely
  PII-free and use only fictitious information.
- [x] **H-033 — Review hard-negative pairs.** Pay special attention to dates,
  cities, usernames, IP terminology, and context-dependent numbers.
- [x] **H-034 — Review long-context samples.** Confirm that PII occurs at varied
  positions and that documents resemble intended inputs.
- [x] **H-035 — Approve dataset statistics.** Confirm class distribution,
  negative ratio, token lengths, and rare-class coverage.
- [x] **H-036 — Approve split integrity.** Review the template-grouping strategy
  and the report of duplicate or near-duplicate records.

Suggested minimum review sample:

| Review group | Minimum human sample |
| --- | ---: |
| Random converted examples | 100 |
| Every target label | 20 per label where available |
| Negative examples | 50 |
| Hard-negative pairs | 50 pairs |
| Long-context examples | 30 |
| Repeated-entity examples | 20 |

## 5. Metric and model-selection decisions

- [ ] **H-040 — Approve the metric hierarchy.** Recommended order: exact
  recall, complete-document recall, precision, validity, hardware performance,
  then size.
- [ ] **H-041 — Set a baseline-relative training gate.** Define how much the
  adapter must improve over untouched Qwen before model packaging begins.
- [ ] **H-042 — Set quantization tolerance.** Define the largest acceptable
  recall and complete-document-recall loss relative to BF16.
- [ ] **H-043 — Set boundary-recall tolerance.** Define acceptable loss for
  entities near or across chunk boundaries.
- [ ] **H-044 — Approve the first real training configuration.** Review the
  smoke-run evidence before starting a long run.
- [ ] **H-045 — Approve the final adapter.** Base the decision on frozen
  validation/test metrics and error analysis, not training loss alone.
- [ ] **H-046 — Approve Pi quantization.** Select Q4 or Q5 only after actual Pi
  memory, speed, temperature, and quality measurements.
- [ ] **H-047 — Approve final chunking defaults.** Review recall, calls, latency,
  duplicate rate, and conflict rate.

## 6. Training supervision

- [ ] **H-050 — Prepare the Mac for the smoke run.** Connect power, close
  memory-heavy applications, and confirm sufficient disk space.
- [ ] **H-051 — Observe smoke training.** Confirm finite loss, manageable memory
  pressure, validation execution, and adapter creation.
- [ ] **H-052 — Authorize staged experiments.** Do not begin the largest run
  until smaller experiments demonstrate value.
- [ ] **H-053 — Preserve experiment evidence.** Do not overwrite adapters,
  predictions, logs, or manifests from earlier runs.
- [ ] **H-054 — Stop unsafe or invalid runs.** Stop on non-finite loss, severe
  memory pressure, thermal instability, or a discovered dataset defect.

## 7. Raspberry Pi operations

- [ ] **H-060 — Update the Pi and install build tools.** These operations require
  system administration access.
- [ ] **H-061 — Compile the pinned `llama.cpp` revision.** Save the commit and
  build output.
- [ ] **H-062 — Transfer Q4 and Q5 candidates.** Verify checksums after transfer.
- [ ] **H-063 — Run initial CLI inference.** Confirm model loading and structured
  output before starting the server.
- [ ] **H-064 — Run sustained hardware benchmarks.** Monitor temperature,
  throttling, swap, RAM, and responsiveness.
- [ ] **H-065 — Install and enable systemd services.** Review generated unit
  files before copying them into `/etc/systemd/system`.
- [ ] **H-066 — Perform a cold-reboot test.** Do not use SSH intervention before
  checking application availability.
- [ ] **H-067 — Test from another approved LAN device.** Confirm that FastAPI is
  reachable and the model endpoint is not.

## 8. Manual error analysis

- [ ] **H-070 — Review false negatives first.** Classify unusual formatting,
  ambiguity, rare class, repeated omission, quantization regression, and chunk
  boundary failures.
- [ ] **H-071 — Review false positives.** Classify generic dates, cities,
  companies, roles, and random numbers incorrectly treated as PII.
- [ ] **H-072 — Separate model and application errors.** Determine whether each
  failure originated in model detection, alignment, chunking, reconciliation,
  or annotation policy.
- [ ] **H-073 — Approve corrective work.** Dataset changes require a new dataset
  version and new training; rule changes require affected tests and reports to
  be rerun.

## 9. Release approval

- [ ] **H-080 — Confirm licenses and attribution.** Review code, base-model,
  dataset, adapter, and GGUF redistribution terms at release time.
- [ ] **H-081 — Review repository history.** Confirm it contains no credentials,
  user documents, real PII, model binaries, or unintended generated data.
- [ ] **H-082 — Approve the model card.** Confirm training data, limitations,
  evaluation, intended use, and prohibited claims.
- [ ] **H-083 — Approve the application README.** Verify installation,
  architecture, benchmarks, security posture, and limitations.
- [ ] **H-084 — Approve public artifacts.** Verify filenames, checksums, sizes,
  and recommended deployment configuration.
- [ ] **H-085 — Make the final release decision.** Publish only after all release
  gates in `09-milestones-and-task-register.md` are satisfied.

## Human stop/go checkpoints

| Checkpoint | Required evidence | Human decision |
| --- | --- | --- |
| Policy freeze | Annotation and privacy policies | Permit dataset construction |
| Dataset freeze | Validation, statistics, and review sample | Permit baseline evaluation |
| Training start | Smoke-run evidence and baseline report | Permit long QLoRA runs |
| Model freeze | Adapter metrics and error analysis | Permit fusion/conversion |
| Quantization freeze | BF16/Q8/Q5/Q4 comparison | Permit Pi candidate transfer |
| Deployment choice | Pi quality and performance report | Select production GGUF |
| Release candidate | Full tests, benchmarks, docs, licenses | Permit public release |
