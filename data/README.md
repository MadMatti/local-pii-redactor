# Data artifacts

This directory separates source, intermediate, prepared, calibration, and
long-document data. Generated content is ignored by Git because records may be
large or sensitive.

| Directory | Purpose |
| --- | --- |
| `raw/` | Hugging Face cache and immutable source snapshots |
| `intermediate/` | Source lock, inspection bundle, and review previews |
| `processed/` | MLX chat JSONL stages, manifests, rejects, and validation |
| `calibration/` | Test-independent text for later importance-matrix work |
| `long_documents/` | Specifications and future generated boundary corpus |

Rebuild generated outputs with `python scripts/data/build_dataset.py`, then run
`python scripts/data/validate_dataset.py` before training or evaluation.
