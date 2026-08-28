# Project charter

## Mission

Build a completely local application that detects personally identifiable
information in text documents and redacts it without sending the document to an
external service. The project also demonstrates the complete lifecycle of a
small local language model: dataset preparation, QLoRA fine-tuning, evaluation,
adapter fusion, GGUF conversion, deployment quantization, CPU serving,
long-document processing, and edge deployment.

## V1 outcome

A user opens a local web page, pastes text or uploads a supported text file, and
receives:

- a redacted document whose non-redacted characters are unchanged;
- validated entity types and exact character spans;
- progress for long documents;
- non-sensitive timing and chunk statistics;
- a clear warning that machine-learning redaction can miss PII.

The application runs on a Raspberry Pi 4 and calls a Qwen3-1.7B-derived GGUF
through a localhost-only `llama-server` process.

## In scope

### Machine-learning lifecycle

- Inspect and transform a synthetic PII dataset.
- Define a stable canonical label taxonomy.
- Build negative, hard-negative, repeated-entity, and long-context examples.
- Establish regex and untouched-Qwen baselines.
- Fine-tune Qwen3-1.7B using QLoRA through MLX-LM.
- Select checkpoints using task metrics rather than training loss alone.
- Fuse the adapter and convert the fused model to GGUF.
- Evaluate BF16, Q8, Q5, Q4, and importance-matrix-assisted Q4 artifacts.
- Select the deployment model using measurements from the actual Pi.

### Application lifecycle

- Exact-token, sentence-aware, overlapping chunking.
- Character-offset preservation independent of token IDs.
- Exact substring alignment of model-returned entity text.
- Cross-chunk duplicate and conflict reconciliation.
- Deterministic right-to-left redaction.
- A typed HTTP client for `llama-server`.
- A FastAPI backend and lightweight local web interface.
- Sequential model inference with progress reporting.
- Automatic startup through systemd.

### Evaluation and release

- Reproducible single-chunk and long-document evaluations.
- Pi latency, memory, throughput, and thermal benchmarks.
- Model, dataset, and application documentation.
- Separate publication of source code, adapter weights, and GGUF artifacts.

## Out of scope for V1

- Guaranteed anonymization or formal compliance certification.
- PDF or DOCX layout-preserving redaction.
- OCR or image redaction.
- Cloud inference or telemetry.
- Arbitrary multi-user throughput.
- True constant-memory input streaming.
- Pseudonymization or reversible encryption.
- Multilingual quality claims.
- An automatic second model call to adjudicate conflicts.
- React or another frontend build pipeline.
- Using a model-generated rewritten document as the final result.

Deferred work must remain in a V2 backlog until V1 evidence shows it is needed.

## Canonical V1 entity taxonomy

| Label | Intended meaning | Important boundary |
| --- | --- | --- |
| `PERSON_NAME` | A person's first name, last name, or full name | Companies and generic roles are excluded |
| `EMAIL` | Email address | Preserve punctuation only when it is part of the address |
| `PHONE_NUMBER` | Telephone or mobile number | Preserve formatting exactly |
| `ADDRESS` | Physical or postal address | A city alone is not automatically an address |
| `DATE_OF_BIRTH` | A date explicitly referring to birth | Generic dates are excluded |
| `IP_ADDRESS` | IPv4 or IPv6 address | Discussion of the protocol is not an address |
| `USERNAME` | A person's account identifier | Generic field names are excluded |
| `CREDIT_CARD` | Credit/debit card number | Preserve spaces and punctuation |
| `BANK_ACCOUNT` | Account, routing, or banking identifier | Random numbers without banking context are excluded |
| `NATIONAL_ID` | National, tax, SSN-like identifier | Country-specific subclasses are collapsed in V1 |
| `MEDICAL_ID` | Medical record or beneficiary identifier | Medical topics without identifiers are excluded |
| `CUSTOMER_ID` | Customer or user-account identifier | Must be used as an identifier in context |
| `EMPLOYEE_ID` | Employee identifier | Job titles are excluded |
| `VEHICLE_ID` | License plate or vehicle identifier | Generic vehicle descriptions are excluded |

The human owner must finalize ambiguous annotation rules in `H-001` through
`H-006` before the production dataset is built.

## Architectural principles

1. The LLM detects semantic entities; deterministic code performs redaction.
2. The LLM never supplies trusted character offsets.
3. Every predicted value must exist exactly in its source chunk.
4. The original document remains immutable until all spans are reconciled.
5. Training and deployment quantization are separate experiments.
6. The frozen test set is not used for training, tuning, or imatrix calibration.
7. Model quality is measured before and after every representation change.
8. Long-document support means independence from the context-window limit, not
   constant processing time.
9. Privacy-sensitive content is not persisted or logged by default.
10. Version-dependent commands, endpoints, and model revisions are pinned and
    revalidated when implementation begins.

## Quality priorities

The default order for selecting a model or processing configuration is:

1. Exact entity recall.
2. Complete-document recall.
3. Precision and PII-free false-positive rate.
4. Schema validity and operational reliability.
5. Pi memory fit and thermal stability.
6. End-to-end latency.
7. Artifact size.

A smaller model is not preferred when it causes a privacy-relevant recall loss
that exceeds the human-approved tolerance.

## Definition of done

The project is complete only when all of the following are true:

- The label policy, privacy model, and limitations are documented.
- Dataset construction is deterministic and validation passes with no errors.
- Regex, base-Qwen, and fine-tuned-model baselines are preserved.
- Fine-tuning provides a measured improvement on the frozen test set.
- Fused MLX and BF16 GGUF outputs pass the conversion-parity gate.
- Quantized candidates are evaluated on identical examples.
- The selected GGUF fits and runs reliably on the Raspberry Pi 4.
- Chunking, alignment, reconciliation, and redaction pass unit and integration
  tests, including boundary torture tests.
- Non-redacted source text remains byte-for-byte equivalent after deterministic
  replacement, accounting only for the intended placeholders.
- The FastAPI application and UI process supported files without internet
  access and without persisting content.
- `llama-server` is only reachable on localhost unless an explicit decision
  changes that architecture.
- Both services start automatically and work after a cold reboot.
- Final accuracy, latency, memory, and thermal benchmarks are reproducible.
- Source code, model artifacts, licenses, model card, results, and limitations
  are ready for separate, appropriate publication.

## Provisional effort envelope

| Workstream | Active effort | Elapsed-time uncertainty |
| --- | ---: | --- |
| Policy, repository, and environment | 2–4 days | Tool installation and model download |
| Dataset and evaluation | 5–8 days | Dataset inspection and human review |
| Training and model packaging | 4–8 days | Training duration and memory constraints |
| Document-processing engine | 5–8 days | Edge cases and boundary evaluation |
| API, UI, and privacy hardening | 4–7 days | Long-job behavior and error handling |
| Pi deployment and benchmarking | 3–6 days | Compilation, transfer, and thermal tests |
| Release and documentation | 2–4 days | Human approval and artifact upload |

These estimates are planning ranges, not delivery promises. Experiment failures
should result in a documented decision and revised plan rather than bypassing a
quality gate.
