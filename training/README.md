# Training

MLX-LM experiment configurations are versioned in `configs/`. Adapters and logs
are written under ignored `models/adapters/` directories. Dataset validation
must pass and the human dataset gate must be approved before executing a run.

Example command after approval:

```bash
python -m mlx_lm lora --config training/configs/smoke.yaml
```
