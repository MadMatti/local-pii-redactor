# Approved model packaging trials

Checkpoint 19,000 is approved for local packaging trials under `H-045`, not
for deployment. The immutable inputs and tolerances are in
[`models/packaging-v1.json`](../models/packaging-v1.json). Do not substitute the
final checkpoint, change the frozen validation data, or relax the evaluator
to make a packaging candidate pass.

## Environment and local artifacts

The existing Python 3.11 `.venv` runs MLX and evaluation. The separate
`.venv-conversion` isolates the official GGUF converter's PyTorch and
Transformers requirements from that environment. The pinned llama.cpp checkout
and both virtual environments are ignored by Git, as are weights, generated
GGUFs, raw predictions, calibration text, and local manifests.

The pinned llama.cpp revision is
`38a5b42d9a3e82e0a586bcd1caed121f36c87a73`. Its `llama-server`,
`llama-quantize`, and `llama-imatrix` targets were built with Metal and
Accelerate support. A successful build is not yet evidence of converted-model
correctness or Raspberry Pi performance.

Record and verify the clean pinned checkout, compiler, build flags, native
binary/library hashes, and installed converter dependencies with:

```bash
.venv/bin/python scripts/model/record_toolchain.py
```

The resulting `models/gguf/toolchain-v1/manifest.json` is local evidence and is
never overwritten by the command. Converter dependencies are pinned in
`requirements-conversion.txt`; install them only into `.venv-conversion`.

Keep this project downloaded when it is stored in an iCloud-synced Documents
folder. Offloaded dependencies and weights can make ordinary reads appear to
hang. Use AC power and check free disk space before generating more artifacts.
Do not remove the approved base, adapter, training run, or evaluation evidence
to free space.

## 1. Fuse and verify

For a new, unused output/run pair in the packaging plan:

```bash
.venv/bin/python scripts/model/fuse_adapter.py
```

The wrapper verifies approved hashes, refuses existing output directories,
records inputs and implementation hashes, invokes native MLX-LM fusion with
`--dequantize`, and checks floating-point safetensors and all 100 frozen
validation prompts. It preserves the exact approved tokenizer files. This
dequantizes the trained 4-bit base and merges the adapter; it does **not**
recover the original prequantization base weights.

The first local fusion completed its native subprocess and produced 310 BF16
tensors with 1,720,574,976 parameters, but the wrapper rejected the generated
tokenizer. Serialization retained the vocabulary and embedded chat template,
while changing tokenizer JSON components and dropping additional-special-token
declarations from the tokenizer configuration. Do not treat this failed wrapper
manifest as a passing fusion.

The explicit recovery command for that already-generated model is:

```bash
.venv/bin/python scripts/model/verify_fusion.py --restore-approved-tokenizer
```

It checks that native fusion completed for the exact plan and unchanged input
files; checks the frozen validation hash before mutation; moves generated
tokenizer files to the run's `generated-tokenizer/` backup; and copies approved
tokenizer files byte-for-byte. It verifies the vocabulary, special-token map,
and rendered prompt IDs against the approved base. Weight hashes are checked
before and after recovery. The original failed `manifest.json` remains intact;
new evidence is written to `verified-fusion.json`.

A pre-existing backup or verification report is a hard stop, including after
interruption. Do not delete evidence to force a retry: inspect the saved state
and establish a separate recovery attempt if one is necessary. Recovery passing
does not imply that model generations pass parity.

Recovery completed on 2026-09-16. All 100 validation prompts passed vocabulary,
special-token, and exact token-ID checks. The fused weights remained unchanged:
`c672a11d7214e2d1a2b4765950c9552575d206cf07764bdf3845417e0ac5b56d`.
The prompt token-ID digest is
`0a6e124e710de8c78e7d9dbd3d7fc341528a08de818618eb457895cf6a8dee04`.

## 2. Frozen generation parity

Only after structural and tokenizer verification passes:

```bash
.venv/bin/python scripts/evaluation/run_model_baseline.py \
  --model models/fused/v1-ckpt19000-bf16 \
  --dataset data/processed/stages/smoke/valid.jsonl \
  --output-dir evaluation/results/v1-fused-bf16-smoke-valid \
  --expected-dataset-sha256 49f1df7ba0826f913a364aebf155441a7e5e58b574adb212ee4dec59a2d14042

.venv/bin/python scripts/evaluation/check_parity.py \
  --dataset data/processed/stages/smoke/valid.jsonl \
  --reference-predictions evaluation/results/v1-ckpt19000-smoke-valid/predictions.jsonl \
  --candidate-predictions evaluation/results/v1-fused-bf16-smoke-valid/predictions.jsonl \
  --expected-dataset-sha256 49f1df7ba0826f913a364aebf155441a7e5e58b574adb212ee4dec59a2d14042 \
  --output-dir evaluation/results/v1-fused-bf16-smoke-valid-parity
```

Generation uses seed 42, greedy decoding, thinking disabled, and a 384-token
output cap, matching the adapter reference. Use `--resume` on the generation
command only for an interrupted run with identical configuration and inputs.

Parity outcomes are `0` for passing task-score gates, `2` for completed reports
requiring review, and `1` for an operational/input failure. Fusion and BF16 GGUF
conversion require no loss in exact recall, complete-document recall, or schema
validity. Inspect changed outputs and per-label deltas even when aggregates
pass. Aggregate parity is not bitwise equivalence or deployment approval.

### Completed fused-MLX result

The 100-record comparison passed all three zero-drop gates. Both models find
215/246 gold entities (0.873984 recall), achieve 0.72 complete-document recall,
and return valid schema for every record. The fused model has 31 false positives
versus the adapter's 30; F1 is 0.873984 versus 0.875764. Exact-document match
remains 0.73 and negative-document false positives remain zero.

Only one output changed: a source-present 12-character identifier was additionally
labeled MEDICAL_ID despite having no matching or overlapping gold annotation.
No reference prediction was removed; the other 99 outputs are byte-identical.
This recorded extraction difference does not establish a numerical cause. The
approved recall/validity gates pass, but the models are not exactly equivalent.

Generation took 491.79 seconds with 3.787 GB peak MLX allocation. These are local
Mac diagnostics, not controlled cross-runtime or Pi performance results. Full
2,000-record fused-MLX test generation was not run. The commit-safe aggregate is
[`fusion-parity-v1.json`](../evaluation/baselines/fusion-parity-v1.json).

## 3. Conversion and quantization gates

Do not execute these stages until the preceding generation gate passes:

1. Convert verified fused weights to BF16 GGUF with the pinned official
   converter in `.venv-conversion`. Record source, toolchain, command, output
   hashes, and elapsed time. Preserve the common floating-point parent.
2. Check GGUF metadata, tokenizer IDs, non-thinking chat rendering, and frozen
   generations against the fused model using the same output budget and scorer.
3. Produce Q8_0, Q5_K_M, Q4_K_M, and Q4_K_M with an importance matrix. Use only
   the approved calibration corpus, never validation/test documents, for the
   importance matrix. Quantize each candidate from the BF16 parent, not another
   quantized model.
4. Compare each quantization to its floating-point parent. The approved maximum
   absolute loss is 0.01 in exact recall and 0.01 in complete-document recall;
   schema validity must not decline. Pass these tolerances explicitly to
   `check_parity.py`; do not use them for the fusion/conversion stages.
5. Select on validation evidence, then report the selected artifact's frozen
   test results and local performance. Raspberry Pi benchmarking and deployment
   remain separate work and approval gates.

No converted or quantized candidate is considered validated merely because its
file was created successfully. An unexplained regression stops downstream
packaging and requires diagnosis, not a changed threshold or checkpoint.

The guarded conversion command, after fusion parity and toolchain recording, is:

```bash
.venv/bin/python scripts/model/convert_gguf.py
```

It rechecks fusion, prediction, decoding, dependency, and converter-source
provenance; recomputes the frozen generation gate; and requires estimated output
space plus a 4 GiB disk reserve. It refuses an existing output directory. The
default output is `models/gguf/v1-ckpt19000-bf16/model-bf16.gguf`, with a manifest,
local converter log, and static metadata report alongside it. Static checks
cover architecture, tensor counts, complete approved vocabulary, special-token
IDs, and the embedded chat template. They do not replace native generation
parity or EOS-behavior review.

BF16 conversion produced a 3,447,348,928-byte file with SHA-256
`11f71a9ae53bfb60a9eed335c2936ada3f91e627875766c473e964fca91a21ca`.
It contains 197 BF16 tensors and 113 F32 tensors (the converter's floating-point
normalization parameters), totaling the same 1,720,574,976 parameters.
All 151,669 approved vocabulary entries, special IDs, and the chat template pass
static verification; the embedding vocabulary is padded to 151,936 entries.

The first post-conversion inspection failed to discover the editable `gguf`
package. The inspector now explicitly loads the pinned checkout, matching the
official converter. The existing GGUF was verified without rewriting it using:

```bash
.venv/bin/python scripts/model/convert_gguf.py --verify-existing
```

That command preserves the original `manifest.json` and `conversion.log`, checks
unchanged GGUF hashes, and writes `verified-conversion.json`, `metadata.json`,
and `verification.log`. It refuses an existing verification attempt. The native
conversion was not repeated, and its original failed wrapper status was not
silently rewritten.

### Prepared GGUF evaluator

`scripts/evaluation/run_gguf_baseline.py` is the GGUF generation runner. It
requires explicit model and dataset checksums and a recorded native toolchain.
It verifies every frozen prompt against the approved Hugging Face tokenizer
and GGUF chat template before generating. It refuses context truncation, uses
greedy decoding without repetition penalties or prompt-cache reuse, and calls
the native completion endpoint with exact token IDs.

The owned server binds only to `127.0.0.1`, uses an ephemeral API key, disables
the web UI and model downloads, and is terminated on completion or failure.
The client ignores proxy settings and refuses redirects. Raw predictions remain
local; stdout contains progress and aggregate metrics only. Resume requires
identical model, data, tokenizer, binary, dependency, and implementation hashes.
These safeguards have unit coverage; real GGUF validation is a separate pending
step, not established by mocked transport tests.

For the verified BF16 artifact:

```bash
.venv/bin/python scripts/evaluation/run_gguf_baseline.py \
  --model models/gguf/v1-ckpt19000-bf16/model-bf16.gguf \
  --expected-model-sha256 11f71a9ae53bfb60a9eed335c2936ada3f91e627875766c473e964fca91a21ca \
  --dataset data/processed/stages/smoke/valid.jsonl \
  --expected-dataset-sha256 49f1df7ba0826f913a364aebf155441a7e5e58b574adb212ee4dec59a2d14042 \
  --output-dir evaluation/results/v1-gguf-bf16-smoke-valid
```

The runner also verifies the model path, context size, disabled UI/proxy, and
native EOS mapping. It checks the terminal sampled token for every EOS-ended
response: llama.cpp recognizes more end-of-generation tokens than the approved
MLX tokenizer, so an unexpected terminal token must not pass silently. Token
IDs and native timing are saved locally; text-token counts and sampled-token
counts are kept distinct because the latter include EOS.

### Calibration audit

```bash
.venv/bin/python scripts/model/audit_calibration.py
```

The existing approved corpus regenerates byte-for-byte from the train-only
synthetic generator: 500 distinct documents, 250 positive-generator and 250
negative-generator records. There are zero exact or whitespace/case-normalized
document overlaps with frozen validation or test. There are 287 overlaps with
training, which are allowed and explicitly reported. No dataset was rewritten.
This is narrow synthetic train-style calibration, not an independent natural
document corpus; no claim of optimal calibration representativeness is made.
The local audit is `models/gguf/calibration-audit-v1/manifest.json`.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -m "not network and not mlx"
```

The unit tests do not require Metal, MLX, PyTorch, or network access. Actual
fusion and MLX generation require Metal access on the local Mac.
