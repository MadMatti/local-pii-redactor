# Model artifacts

Model binaries are local-only and ignored by Git.

- `base-qwen3-1.7b-4bit/`: local MLX base model and tokenizer.
- `adapters/`: QLoRA adapter runs.
- `fused/`: fused floating-point checkpoints.
- `gguf/`: BF16 and deployment quantizations.

Every produced model artifact must have a manifest and checksum before it is
used for evaluation or deployment.
