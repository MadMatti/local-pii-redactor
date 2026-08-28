# MLX-LM configurations

- `smoke.yaml`: roughly 100 iterations on the 500-record training stage at
  1024 tokens. A tokenizer audit found assistant-target loss at 768 for 20
  smoke-train records, so 768 is not used for this pipeline check.
- `v0.yaml`: 8-layer/rank-8/768 first meaningful run on the 5,000-record stage.
- `v0-l16-r8-768.yaml`: holds rank and context fixed while testing 16 layers.
- `v0-l8-r16-768.yaml`: holds layers and context fixed while testing rank 16.
- `v0-l8-r8-1024.yaml`: holds layers and rank fixed while testing 1024 tokens.
- `v1.yaml`: main candidate on the 20,000-record stage; values remain subject
  to evidence from smoke and V0.

These are starting configurations, not authorization to train. Copy a config
to a run-specific record and capture the dataset manifest hash before launch.

The 768-token V0 configs use the deterministic
`data/processed/length_views/v0-768` view. It excludes complete chat sequences
that MLX-LM would truncate; the 1024 config uses the full V0 stage.
