# MLX-LM configurations

- `smoke.yaml`: roughly 100 iterations on the 500-record training stage.
- `v0.yaml`: first meaningful run on the 5,000-record stage.
- `v1.yaml`: main candidate on the 20,000-record stage; values remain subject
  to evidence from smoke and V0.

These are starting configurations, not authorization to train. Copy a config
to a run-specific record and capture the dataset manifest hash before launch.
