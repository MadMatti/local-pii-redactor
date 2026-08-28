# Calibration corpus

`pii_calibration.txt` is generated independently of the frozen test set. It
contains deterministic positive, non-PII, hard-negative, repeated-value, and
long-context text for later GGUF importance-matrix generation. Its manifest
records the seed and checksum.
