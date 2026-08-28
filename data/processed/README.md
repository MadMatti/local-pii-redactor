# Prepared datasets

Generated files in this directory are ignored. The main MLX-LM dataset uses
`train.jsonl`, `valid.jsonl`, and `test.jsonl`. Nested stages live at
`stages/v0/` and `stages/smoke/`; `full_source/` contains every safely
convertible Gretel record for audit and future experiments.

`manifest.json` pins source, policy, prompt, seed, selection rules, counts, and
artifact hashes. `rejected_source.jsonl` records source defects without source
repair. `validation/` contains the latest strict validation report.
`synthetic_review_samples.jsonl` is a deterministic, reason-tagged bundle for
the human dataset gate.
