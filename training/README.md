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
