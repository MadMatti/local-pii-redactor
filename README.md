# Local PII Redactor

An offline-first project for fine-tuning, evaluating, quantizing, and deploying
Qwen3-1.7B as a structured PII extractor. Dataset V1 was approved on 2026-08-28.
Main training and checkpoint evaluation are complete. Checkpoint 19,000 was
approved on 2026-09-15 for fusion/parity and quantization trials.

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
- The main run completed 19,255 microbatch iterations with 2.485 GB peak MLX
  memory. Checkpoint 19,000 was selected on frozen validation task metrics.
- On the full 2,000-record test, that checkpoint achieves 88.82% exact recall,
  88.98% F1, 75.20% complete-document recall, and 99.80% schema validity.
- A deterministic 361-document error-review bundle is ready locally. Names,
  identifier confusion, and four invalid outputs still require review.
- Fused BF16 MLX passes the 100-record recall/validity parity gates, with one
  additional MEDICAL_ID false positive recorded. BF16 GGUF conversion and static
  metadata verification are complete; native generation parity gates quantization.

See [the main-adapter review](docs/decisions/0005-main-adapter-review.md) for
baseline comparisons, limitations, artifact paths, and the approved `H-045`
decision. These results do not establish deployment readiness.

The [model-packaging guide](docs/model-packaging.md) documents approved fusion,
tokenizer recovery, immutable toolchain evidence, and the generation-parity
gates that must pass before conversion and quantization.

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
scripts/evaluation/      frozen scoring, checkpoint comparisons, and error review
scripts/model/           training, approved fusion/recovery, and toolchain checks
src/pii_redactor/        installable Python package
tests/                   offline unit and opt-in integration tests
training/configs/        versioned MLX-LM experiment configurations
```

Dataset approval is recorded in
[`docs/decisions/0001-dataset-v1-approved.md`](docs/decisions/0001-dataset-v1-approved.md).
Model selection and deployment remain separately review-gated.
