# Decision 0004 — V0 promotion and main-run configuration

Status: accepted for main training

Date: 2026-08-31

## Decision

Promote the final checkpoint from the one-epoch V0 run and use its winning
architecture for the main experiment:

- 16 LoRA layers;
- rank 8, scale 20, dropout 0;
- complete-record 768-token data view;
- batch size 1 with 8-step gradient accumulation;
- prompt masking and gradient checkpointing;
- learning rate `1e-5`;
- seed 42;
- one pass over the main training view.

The main experiment starts from the pinned quantized base model. The V0 adapter
selects the architecture and training budget; its weights are not mixed into
the main run.

## Frozen V0 evidence

The final V0 adapter was compared with the lowest-validation-loss checkpoint at
iteration 4,400 on the frozen 100-record smoke test. The final adapter ranked
first:

| Candidate | Recall | Complete-doc recall | Precision | F1 | Schema valid | PII-free FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Final, iteration 4,819 | 0.7928 | 0.6267 | 0.8541 | 0.8223 | 0.9900 | 0.0000 |
| Checkpoint 4,400 | 0.7888 | 0.6133 | 0.8216 | 0.8049 | 0.9900 | 0.0800 |

On the complete frozen 481-record V0 768-token test view, the promoted final
adapter achieved:

- micro exact recall 0.7757;
- complete-document recall 0.5568;
- micro exact precision 0.8005;
- micro exact F1 0.7879;
- schema validity 0.9875;
- PII-free false-positive rate 0.0090;
- two hallucinated substrings.

This materially exceeds the untouched-base result on the same view (recall
0.3373, complete-document recall 0.1595, F1 0.3996) and the best short screen
(recall 0.6301, complete-document recall 0.3973, F1 0.6521).

## Main data view

The deterministic `main-768` view retains complete rendered chats only; it does
not truncate prompts or targets:

| Split | Source | Kept | Excluded over 768 | Negative ratio |
| --- | ---: | ---: | ---: | ---: |
| Train | 20,000 | 19,255 | 745 | 23.14% |
| Validation | 2,000 | 1,997 | 3 | 25.04% |
| Test | 2,000 | 1,924 | 76 | 23.08% |

All 14 canonical labels remain represented in every split. The generated view,
model weights, predictions, logs, and evaluation reports stay local and are
excluded from Git; manifests record their hashes.

## Gate after training

The final main checkpoint and the strongest validation-loss checkpoint must be
evaluated with the frozen task evaluator. Selection follows the approved order:
exact recall, complete-document recall, exact precision, schema validity, exact
F1, then lower PII-free false-positive rate. Final adapter approval remains the
human gate `H-045`; this decision authorizes the experiment, not fusion or
quantization.
