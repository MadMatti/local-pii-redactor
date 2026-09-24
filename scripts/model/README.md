# Model commands

`run_training.py` executes versioned MLX-LM configurations and writes run
manifests, resource metrics, input hashes, and adapter checksums. See
[the training guide](../../training/README.md) for usage and completed runs.

Fusion, conversion, quantization, and verification are implemented. See
[the packaging guide](../../docs/model-packaging.md) for the pinned toolchain,
provenance gates, and commands. `quantize_gguf.py` creates approved formats from
the verified BF16 parent; `build_imatrix.py` uses only approved calibration.
These commands refuse existing artifact directories and preserve local evidence.

`report_quantization.py` recomputes all frozen validation comparisons. Only Q8
passed the current gates; smaller formats must not be promoted.

`report_q8_test.py` summarizes the subsequently approved full frozen Q8 test
after all 2,000 predictions are complete. Its protocol is
[`q8-full-test-v1.json`](../../models/q8-full-test-v1.json). It verifies model,
data, runtime, prompt/EOS, and reference checksums before exporting aggregate
metrics only. The adapter comparison is descriptive, not test-based selection
or proof of full-test BF16/Q8 parity. Deployment remains unapproved.
