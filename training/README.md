# Training

MLX-LM experiment configurations are versioned in `configs/`. Adapters and logs
are written under ignored `models/adapters/` directories. Dataset validation
must pass and the human dataset gate must be approved before executing a run.

Install the Mac-only training dependencies with:

```bash
python -m pip install -e ".[data,dev,training]"
```

Example command after approval:

```bash
python scripts/model/run_training.py \
  --config training/configs/smoke.yaml \
  --run-dir training/runs/smoke-v1
```

The wrapper streams the normal MLX-LM progress output while recording the Git
commit, configuration and dataset hashes, duration, losses, throughput, peak
memory, logs, and adapter checksums. It refuses to overwrite prior evidence.
Before allocating Metal memory, it renders every chat record and fails if right
truncation could remove any assistant target.

The four V0 screening configurations each run for 625 iterations so layer,
rank, and context effects can be compared before a longer run. The 768-token
screens use the deterministic complete-record view at
`data/processed/length_views/v0-768`; the 1024-token screen uses all V0 rows.

After the frozen comparison selects an architecture,
`v0-selected-l16-r8-768-full.yaml` runs one pass over all 4,819 complete V0
training records. It writes to a new adapter directory and does not resume or
overwrite any screening adapter.
