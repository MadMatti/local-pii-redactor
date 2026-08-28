# Decision 0003 — V0 QLoRA architecture selection

- **Status:** Approved for one-epoch V0 training
- **Date:** 2026-08-28
- **Owner:** Mattia Evangelisti
- **Dataset:** deterministic 768-token V0 complete-record view

## Decision

Promote the 16-layer, rank-8, 768-token QLoRA architecture to the one-epoch V0
run. The selected screening artifact is the iteration-600 checkpoint from
`v0-l16-r8-768-screen`; it is evidence for the architecture and is not itself
the final project adapter.

The one-epoch run starts from the pinned base model, consumes each of the 4,819
complete V0 training records once, and writes to the new
`models/adapters/v0-selected-l16-r8-768-full` directory. It must not overwrite
or resume a screening adapter.

## Frozen evidence

On the common 100-record smoke comparison, the selected checkpoint achieved
0.6892 exact recall, 0.4667 complete-document recall, 0.7208 precision, and
0.7047 F1. The next-best rank-16 candidate achieved 0.6375 recall and 0.6695
F1.

On the full 481-record V0 768-token test view, the selected checkpoint achieved
0.6301 exact recall, 0.3973 complete-document recall, 0.6758 precision, 0.6521
F1, and 0.9522 schema validity. Untouched Qwen achieved 0.3373 recall and
0.3996 F1 on the same records; the 8-layer/rank-8 adapter achieved 0.5434
recall and 0.5880 F1.

## Rejected screening alternatives

- Eight layers/rank 8/768: lower recall and complete-document recall.
- Eight layers/rank 16/768: faster training, but lower task recall and F1.
- Eight layers/rank 8/1024: safe at 2.834 GB peak memory, but lower recall,
  validity, and F1 despite retaining all long V0 records.

The 1024-token result does not prove that context is harmful; it changed both
context and the dataset view. It does show that the tested 8-layer 1024 setup
cannot displace the selected architecture.

## Promotion gate

The one-epoch V0 adapter must finish with finite loss, reload successfully, and
retain a material recall advantage on the frozen 481-record V0 view. Main-stage
training remains contingent on that result. Final adapter approval `H-045` is
not granted by this decision.
