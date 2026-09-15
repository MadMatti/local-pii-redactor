# Model commands

`run_training.py` executes versioned MLX-LM configurations and writes run
manifests, resource metrics, input hashes, and adapter checksums. See
[the training guide](../../training/README.md) for usage and completed runs.

Adapter fusion, GGUF conversion, quantization, and artifact-verification entry
points are still planned. They must consume versioned configs, write run
manifests, and wait for the relevant human approvals.
