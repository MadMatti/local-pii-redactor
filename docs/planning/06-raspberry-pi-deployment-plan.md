# Raspberry Pi deployment plan

## Objective

Deploy the selected GGUF on a Raspberry Pi 4 with reproducible configuration,
local-only model serving, safe application exposure, automatic startup, and
measured resource behavior.

## 1. Hardware readiness

Human owner tasks `H-023` and `H-024` must record:

- Pi model and RAM capacity;
- power-supply rating;
- passive or active cooling;
- storage type, capacity, and free space;
- Ethernet or Wi-Fi connection;
- Raspberry Pi OS release and 64-bit architecture;
- hostname, user, and intended LAN exposure.

- [ ] **PI-001 — Define minimum free resources.** Document disk space required
  for two candidate models, application environment, build tree, and logs.
- [ ] **PI-002 — Add a Pi diagnostic script.** Report architecture, OS, CPU,
  cores, RAM, swap, disk, temperature capability, and throttling status.
- [ ] **PI-003 — Verify 64-bit architecture.** Fail the deployment check unless
  `uname -m` is compatible with the chosen build.

## 2. System prerequisites and `llama.cpp`

- [ ] **PI-010 — Document OS package installation.** Include Git, CMake,
  build-essential, Python, venv, and pip.
- [ ] **PI-011 — Pin the `llama.cpp` commit.** Use the same compatible revision
  tested during conversion unless a newer revision is explicitly revalidated.
- [ ] **PI-012 — Document the build.** Start with a Release build and four build
  jobs; record any Pi-specific CMake options.
- [ ] **PI-013 — Verify binaries.** Confirm `llama-cli`, `llama-server`, and
  tokenizer/metadata behavior expected by the application.
- [ ] **PI-014 — Record build evidence.** Save commit, compiler, CMake version,
  flags, duration, and resulting binary hashes or versions.

## 3. Application environment

- [ ] **PI-020 — Create `requirements-pi.txt`.** Include only runtime packages
  such as FastAPI, Uvicorn, HTTPX, Pydantic, Jinja2, and multipart support.
- [ ] **PI-021 — Add package-install instructions.** Install the application in a
  dedicated virtual environment without MLX or PyTorch.
- [ ] **PI-022 — Add typed production configuration.** Configure model path,
  ports, context, threads, chunking, upload limits, concurrency, and logging.
- [ ] **PI-023 — Validate permissions.** The service user must read code/model
  files and write only to explicitly required operational paths.
- [ ] **PI-024 — Add a deployment verification command.** Check imports,
  configuration, model file/checksum, and server binary before systemd install.

## 4. Model transfer and verification

- [ ] **PI-030 — Publish candidate checksums.** Record Mac-side SHA-256 for Q4
  and Q5 candidates.
- [ ] **PI-031 — Document secure transfer.** Use `scp`, `rsync` over SSH, or a
  trusted offline medium.
- [ ] **PI-032 — Verify Pi-side checksums.** Do not benchmark a mismatched file.
- [ ] **PI-033 — Use explicit model paths.** Avoid ambiguous globs or environment
  expansion in service definitions.
- [ ] **PI-034 — Keep model files out of Git.** Document separate artifact
  acquisition and verification.

## 5. Command-line inference gate

- [ ] **PI-040 — Run Q4 CLI inference.** Start with context 2,048 and four
  threads; test positive, negative, repeated, and hard-negative examples.
- [ ] **PI-041 — Run Q5 CLI inference.** Use identical prompts and settings.
- [ ] **PI-042 — Check structured behavior.** Confirm valid JSON-like extraction,
  absence of thinking output, and acceptable latency.
- [ ] **PI-043 — Measure load and memory.** Record model load time, resident and
  peak RAM, and swap use.
- [ ] **PI-044 — Measure thermals.** Record idle and sustained temperature plus
  throttling flags.

No server or UI work is considered deployable until CLI inference works.

## 6. Pi quantization selection

- [ ] **PI-050 — Benchmark Q4 and Q5 on identical data.** Use warm-up plus
  repeated measured runs.
- [ ] **PI-051 — Record prompt and generation throughput.** Keep these separate.
- [ ] **PI-052 — Run the frozen quality evaluation.** Do not choose using size or
  speed alone.
- [ ] **PI-053 — Run a sustained workload.** Detect thermal throttling, memory
  growth, swap thrashing, or process failure.
- [ ] **PI-054 — Produce a selection report.** Include quality, complete-document
  recall, size, RAM, speed, temperature, and recommendation.
- [ ] **PI-055 — Obtain human approval.** `H-046` selects the production GGUF.

## 7. `llama-server` service

- [ ] **PI-060 — Create `pii-llm.service`.** Use the actual service user and
  explicit executable/model paths.
- [ ] **PI-061 — Bind to localhost.** Use `127.0.0.1:8080` and verify that LAN
  devices cannot reach it.
- [ ] **PI-062 — Configure resources.** Start with context 2,048 and four threads;
  document all server flags.
- [ ] **PI-063 — Configure restart behavior.** Restart on failure with bounded
  delay and avoid a tight crash loop.
- [ ] **PI-064 — Configure shutdown.** Allow enough time for clean process
  termination while keeping reboot reliable.
- [ ] **PI-065 — Verify endpoints.** Test readiness, chat completion, structured
  output, and tokenization from localhost.

## 8. FastAPI service

- [ ] **PI-070 — Create `pii-app.service`.** Set the working directory, virtual
  environment executable, user, and production configuration explicitly.
- [ ] **PI-071 — Declare service ordering.** Start after and require the LLM
  service, while still using application readiness checks.
- [ ] **PI-072 — Configure network binding.** Follow `H-011`; default LAN service
  binding is `0.0.0.0:8000`, but localhost-only remains an allowed safer mode.
- [ ] **PI-073 — Harden the unit.** Apply compatible filesystem, privilege, and
  process restrictions after verifying model/application needs.
- [ ] **PI-074 — Configure safe logging.** Use the journal for content-free
  operational logs and document retention limits.
- [ ] **PI-075 — Configure restart behavior.** Handle transient startup delay
  without creating an infinite high-frequency restart loop.

## 9. Installation and lifecycle operations

- [ ] **PI-080 — Create a reviewed install procedure.** Copy units, reload
  systemd, enable services, and start them through explicit commands.
- [ ] **PI-081 — Create an update procedure.** Stop the app, verify new code and
  model checksums, migrate configuration if necessary, and restart safely.
- [ ] **PI-082 — Create a rollback procedure.** Keep the prior code revision,
  model alias, and configuration available until the new deployment passes.
- [ ] **PI-083 — Create an uninstall procedure.** Disable units and identify
  exactly which files are removed; model deletion should remain an explicit
  human choice.
- [ ] **PI-084 — Document status and recovery commands.** Include service status,
  safe journal inspection, restart, port checks, and readiness tests.

## 10. Deployment acceptance tests

- [ ] **PI-090 — Cold-reboot test.** Both services start without SSH activity.
- [ ] **PI-091 — LAN UI test.** The approved device reaches FastAPI on port 8000.
- [ ] **PI-092 — Model-port isolation test.** Port 8080 is unreachable from the
  LAN and available only locally.
- [ ] **PI-093 — Offline test.** Disconnect internet access while preserving the
  approved local network and complete a redaction.
- [ ] **PI-094 — Failure-recovery test.** Stop or crash the model server and
  confirm readiness and recovery behavior.
- [ ] **PI-095 — Sustained-document test.** Process a long document while
  monitoring memory, swap, temperature, and progress.
- [ ] **PI-096 — Data-remanence review.** Confirm documents and entity values are
  absent from repository files, application storage, browser persistence, and
  expected logs.

## Pi deployment exit gate

- Production model is selected from actual Pi measurements.
- Model and application services survive a cold reboot.
- The raw model port is localhost-only.
- A supported document is processed without internet access.
- No unacceptable throttling, swap behavior, or memory growth appears in the
  sustained benchmark.
- Update, rollback, recovery, and uninstall procedures are documented.
