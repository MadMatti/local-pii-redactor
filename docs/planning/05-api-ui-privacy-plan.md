# API, UI, serving, and privacy plan

## Runtime architecture

```text
Browser on localhost or approved LAN device
                    │
                    ▼
             FastAPI :8000
                    │
       validation, queue, progress
                    │
                    ▼
    document processor, concurrency = 1
                    │
                    ▼
      llama-server 127.0.0.1:8080
                    │
                    ▼
          selected Q4/Q5 GGUF
```

Only FastAPI may be exposed to the LAN. The raw model endpoint remains local to
the Pi.

## 1. `llama-server` request contract

- [ ] **APP-001 — Define canonical messages.** Reuse the exact prompt version
  used for training and evaluation.
- [ ] **APP-002 — Disable Qwen thinking.** Send the supported chat-template
  arguments on every inference request and test that reasoning text is absent.
- [ ] **APP-003 — Request structured output.** Supply the minimal JSON schema for
  an object containing an `entities` array of `type` and `text` objects.
- [ ] **APP-004 — Set deterministic decoding.** Record temperature, top-p,
  maximum output tokens, stop behavior, and seed support where available.
- [ ] **APP-005 — Reserve output capacity.** Reject or reduce an input chunk if
  prompt plus content plus output allowance could exceed server context.
- [ ] **APP-006 — Validate semantics after decoding.** JSON-schema validity does
  not replace Pydantic label and substring validation.

## 2. Inference and tokenizer clients

- [ ] **APP-010 — Implement a typed asynchronous HTTP client.** Encapsulate URL,
  timeouts, request construction, response parsing, and errors.
- [ ] **APP-011 — Separate model and transport errors.** Distinguish unavailable
  server, timeout, invalid HTTP response, invalid JSON, invalid schema, unknown
  label, and hallucinated substring.
- [ ] **APP-012 — Add bounded retry.** Retry only idempotent transient failures,
  use short bounded backoff, and never retry malformed semantic output forever.
- [ ] **APP-013 — Add concurrency control.** Use a process-local semaphore of one
  initially; benchmark two only after sequential behavior is stable.
- [ ] **APP-014 — Implement `/tokenize` access.** Hide HTTP details behind the
  tokenizer protocol used by the chunker.
- [ ] **APP-015 — Add readiness probing.** Confirm model metadata or a lightweight
  server endpoint before accepting work.
- [ ] **APP-016 — Sanitize diagnostics.** Exceptions and logs may contain sample
  IDs, counts, and error classes, but not prompts or entity values.

## 3. FastAPI endpoints

- [ ] **APP-020 — Implement `GET /health`.** Report application liveness without
  forcing model inference.
- [ ] **APP-021 — Implement `GET /ready`.** Verify configuration and model-server
  readiness; return a non-success status when jobs should not be accepted.
- [ ] **APP-022 — Implement `GET /api/model`.** Return safe metadata such as model
  alias, quantization, context, chunk defaults, and application version.
- [ ] **APP-023 — Implement `POST /api/redact`.** Accept validated text, run the
  processor, and return redacted text, spans, safe statistics, and warnings.
- [ ] **APP-024 — Implement file upload.** Support `.txt`, `.md`, `.csv`, and
  `.json` as textual content with encoding checks and bounded size.
- [ ] **APP-025 — Implement long-job progress.** Use Server-Sent Events or a
  similarly simple one-way stream for chunk progress.
- [ ] **APP-026 — Implement cancellation semantics.** Define what happens when a
  browser disconnects and ensure queued work does not run indefinitely.
- [ ] **APP-027 — Add consistent error responses.** Use stable error codes and
  safe user messages without returning document content.
- [ ] **APP-028 — Add request/job identifiers.** Use random IDs that cannot reveal
  content and expire them promptly.

### API response contract

The response should include:

- `redacted_text`;
- reconciled entities with type/start/end and optional display-safe text;
- number of chunks;
- processing duration;
- counts of rejected hallucinations and conflicts;
- model/application version identifiers.

It must not include hidden prompts, raw model reasoning, stack traces, or
internal filesystem paths.

## 4. Input safety and resource controls

- [ ] **APP-030 — Enforce body and upload limits.** Reject oversized content
  before expensive tokenization or inference.
- [ ] **APP-031 — Validate text decoding.** Use a documented encoding policy and
  return a clear error for undecodable input.
- [ ] **APP-032 — Treat structured uploads as text.** Do not evaluate templates,
  deserialize arbitrary objects, expand archives, or execute CSV formulas.
- [ ] **APP-033 — Bound active and queued jobs.** Return a busy response rather
  than exhausting RAM.
- [ ] **APP-034 — Set request and chunk timeouts.** Make values configurable and
  reflect them in the UI.
- [ ] **APP-035 — Handle partial failures safely.** Do not label a document fully
  redacted if any required chunk failed.
- [ ] **APP-036 — Avoid unbounded result retention.** Remove in-memory job state
  after completion or a short configurable expiry.

## 5. Privacy controls

- [ ] **APP-040 — Disable content logging.** Never log request bodies, prompts,
  raw outputs, entity values, or redacted documents.
- [ ] **APP-041 — Review access logging.** Ensure URLs contain no content and
  disable or minimize logs where they provide no operational value.
- [ ] **APP-042 — Avoid persistence.** Process text in memory. If the framework
  spools a large upload, use a controlled location, restrictive permissions,
  prompt cleanup, and explicit documentation.
- [ ] **APP-043 — Avoid browser persistence.** Do not use analytics, remote
  scripts, service-worker caches, `localStorage`, or auto-filled history for
  document content.
- [ ] **APP-044 — Keep the model server private.** Bind it to `127.0.0.1` and
  verify from another device that port 8080 is unreachable.
- [ ] **APP-045 — Use restrictive CORS.** Same-origin only unless a human-approved
  client requires a precise allowlist.
- [ ] **APP-046 — Add security headers.** Use a restrictive Content Security
  Policy compatible with locally served assets, clickjacking protection, and
  safe content-type handling.
- [ ] **APP-047 — Document residual risk.** LAN observers, a compromised Pi, a
  compromised browser, swap, and false negatives remain relevant risks.

## 6. Web user interface

- [ ] **APP-050 — Create a server-rendered shell.** Use FastAPI, Jinja2, HTML,
  CSS, and small JavaScript with no frontend compilation requirement.
- [ ] **APP-051 — Add paste and upload inputs.** Display supported formats and
  size limits before submission.
- [ ] **APP-052 — Add progress state.** Show current chunk, total chunks, elapsed
  time, and a non-sensitive count of detections so far.
- [ ] **APP-053 — Add result presentation.** Show redacted text and a type/count
  summary; display raw entity values only according to `H-015`.
- [ ] **APP-054 — Add copy and download.** Generate downloads client-side or as a
  non-persisted response and give them safe filenames.
- [ ] **APP-055 — Add failure states.** Cover model loading, server unavailable,
  invalid input, timeout, cancellation, partial failure, and busy queue.
- [ ] **APP-056 — Add privacy notice.** State where processing happens, what is
  retained, and that detection is not guaranteed.
- [ ] **APP-057 — Add accessibility.** Provide labels, focus states, keyboard
  operation, status announcements, contrast, and responsive layout.
- [ ] **APP-058 — Keep assets local.** No CDN fonts, analytics, or remote runtime
  dependencies.

## 7. Operational observability

Allowed operational fields:

- application version;
- model alias and quantization;
- request/job ID;
- start/end time and duration;
- input size bucket rather than content;
- chunk count;
- success/error category;
- memory, CPU, and temperature when benchmarking.

Forbidden operational fields:

- source document text;
- prompts;
- raw model output;
- detected entity values;
- redacted document;
- filenames supplied by a user unless sanitized and specifically approved.

- [ ] **APP-060 — Implement structured safe logging.** Make the allowlist above
  explicit and test it.
- [ ] **APP-061 — Add counters and timing.** Keep observability local and avoid
  external telemetry.
- [ ] **APP-062 — Add log-retention instructions.** Document systemd journal
  limits and how to inspect non-sensitive errors.

## Application exit gate

- Short and long inputs work without internet access.
- Model unavailability prevents new work through readiness status.
- Long work visibly progresses and can fail without claiming full redaction.
- Supported uploads are bounded and safely decoded.
- No content appears in logs or browser persistence tests.
- The raw model server is unreachable from the LAN.
- UI warnings and network policy have human approval.
