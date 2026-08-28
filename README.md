# Local PII Redactor

An offline-first project for fine-tuning, evaluating, quantizing, and deploying
Qwen3-1.7B as a structured PII extractor. Dataset V1 was approved on 2026-08-28;
the current workstream is frozen evaluation followed by staged training.

## Current state

- Gretel source revision `e06eb1499ca8d54470f085021cd8e54f9efac7fd` is
  pinned and saved locally.
- All 60,000 source records were inspected.
- 59,987 records convert exactly; 13 defective train annotations are recorded
  and rejected without repair.
- Main (20k/2k/2k), V0 (5k/500/500), and smoke (500/100/100) MLX chat datasets
  are built with a 25% negative ratio.
- All 14 canonical labels are present in every prepared split.
- The strict prepared-data validator passes with zero errors.

Generated datasets, source snapshots, model weights, adapters, and evaluation
results are intentionally ignored by Git.

## Quick start

Python 3.11 is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[data,dev]"
python scripts/data/download_dataset.py --offline
python scripts/data/inspect_dataset.py
python scripts/data/build_dataset.py
python scripts/data/validate_dataset.py
pytest -m "not network and not mlx"
```

Use the downloader without `--offline` only for the first retrieval or an
explicit revision refresh. See [the data-preparation guide](docs/data-preparation.md)
and [the project plans](docs/planning/README.md) for details and review gates.

## Repository map

```text
data/                   ignored source and generated dataset artifacts
deployment/             Raspberry Pi and systemd packaging
docs/                   workflow documentation and detailed plans
evaluation/             baselines and result artifacts
models/                 local bases, adapters, fused models, and GGUFs
scripts/data/            acquisition, inspection, build, and validation CLIs
scripts/evaluation/      future quality evaluation commands
scripts/model/           future training/conversion commands
src/pii_redactor/        installable Python package
tests/                   offline unit and opt-in integration tests
training/configs/        versioned MLX-LM experiment configurations
```

Dataset approval is recorded in
[`docs/decisions/0001-dataset-v1-approved.md`](docs/decisions/0001-dataset-v1-approved.md).
Model selection and deployment remain separately review-gated.
