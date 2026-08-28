# Local PII Redactor planning pack

This directory is the execution plan for building, evaluating, deploying, and
publishing the Local PII Redactor. The plans deliberately separate decisions
and physical actions owned by the human project owner from implementation work
that can be assigned to a developer or coding agent.

## How to use these documents

1. Read and approve the [project charter](00-project-charter.md).
2. Complete the decisions in the
   [human responsibilities plan](01-human-responsibilities.md) before the
   dependent coding tasks begin.
3. Use the [milestone and task register](09-milestones-and-task-register.md) as
   the main progress tracker.
4. Use the workstream plans for implementation detail, tests, and exit gates.
5. Record changes to scope, labels, metrics, or architecture in a decision log
   rather than silently changing completed artifacts.

## Plan files

| Document | Purpose |
| --- | --- |
| [00-project-charter.md](00-project-charter.md) | Scope, outcomes, constraints, principles, and definition of done |
| [01-human-responsibilities.md](01-human-responsibilities.md) | Decisions, reviews, hardware operations, approvals, and publishing tasks that require the project owner |
| [02-coding-roadmap.md](02-coding-roadmap.md) | Repository-wide implementation backlog and recommended build order |
| [03-data-and-ml-plan.md](03-data-and-ml-plan.md) | Dataset, baselines, QLoRA, conversion, and quantization plan |
| [04-document-processing-plan.md](04-document-processing-plan.md) | Chunking, alignment, reconciliation, and deterministic redaction |
| [05-api-ui-privacy-plan.md](05-api-ui-privacy-plan.md) | Model client, FastAPI, progress reporting, UI, and privacy controls |
| [06-raspberry-pi-deployment-plan.md](06-raspberry-pi-deployment-plan.md) | Pi preparation, llama.cpp, services, operations, and hardware validation |
| [07-testing-and-benchmark-plan.md](07-testing-and-benchmark-plan.md) | Unit, integration, model-quality, long-document, and Pi benchmarks |
| [08-release-and-risk-plan.md](08-release-and-risk-plan.md) | Risk register, documentation, licensing, model cards, and release process |
| [09-milestones-and-task-register.md](09-milestones-and-task-register.md) | Dependency-ordered milestones, task IDs, gates, and progress checklist |

## Ownership labels

The plans use the following prefixes:

| Prefix | Owner or workstream |
| --- | --- |
| `H-*` | Human-only or human-approval task |
| `C-*` | General coding and repository task |
| `ML-*` | Dataset, training, model packaging, or quantization task |
| `DOC-*` | Document-processing task |
| `APP-*` | API, UI, serving, or privacy task |
| `PI-*` | Raspberry Pi and deployment task |
| `QA-*` | Test, evaluation, or benchmark task |
| `REL-*` | Documentation and release task |

## Status convention

- `[ ]` not started
- `[~]` in progress
- `[x]` complete
- `[!]` blocked; record the reason next to the task
- `N/A` deliberately excluded by an approved decision

Only mark a milestone complete when its exit gate has passed and the evidence
listed in the milestone has been saved.

## Planning assumptions

- Training computer: Apple Silicon MacBook Pro M1 with 8 GB unified memory.
- Deployment computer: Raspberry Pi 4 running a 64-bit OS, CPU inference only.
- Training model: `mlx-community/Qwen3-1.7B-4bit`.
- Deployment runtime: `llama.cpp` and `llama-server`.
- Initial model context on the Pi: 2,048 tokens.
- Initial content chunk target: 1,024 tokens with 128 tokens of overlap.
- V1 language: English.
- V1 file types: pasted text, `.txt`, `.md`, `.csv`, and `.json` treated as
  textual content.
- V1 loads a document into memory. True streaming is explicitly deferred.
- The model extracts entity type and exact source text. Python owns offsets,
  reconciliation, and redaction.
- User documents, prompts, detected values, and model outputs are not persisted
  or included in operational logs.

These assumptions are starting points, not permanent truths. A change requires
an explicit decision, updated acceptance criteria, and identification of all
affected completed tasks.
