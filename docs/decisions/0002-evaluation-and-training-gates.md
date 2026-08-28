# Decision 0002 — Evaluation order and staged-training gates

- **Status:** Approved for execution
- **Date:** 2026-08-28
- **Owner:** Mattia Evangelisti
- **Authority:** Request to proceed through evaluator, smoke, and staged training

## Metric order

Candidates are ranked in this privacy-first order:

1. exact occurrence-level micro recall;
2. complete-document recall on PII-containing records;
3. exact micro precision;
4. strict schema-valid output rate;
5. exact micro F1;
6. lower PII-free false-positive rate.

Relaxed overlap is diagnostic only and cannot replace exact metrics.

## Execution gates

- The oracle evaluator check must score one on every strict metric.
- Smoke training must finish with finite loss and produce a reloadable adapter.
- A staged adapter must materially improve exact recall over the untouched model
  and regex baseline before it can become the main-run configuration.
- The selected final adapter must be chosen from frozen comparable evaluation
  reports rather than training loss alone.
- Fusion or conversion must not introduce an unexplained exact-recall decline.
- Deployment quantization may lose at most 0.01 absolute exact recall or 0.01
  complete-document recall relative to the selected floating-point parent.

## Operational authorization

The smoke configuration and the bounded V0 layer/rank/context experiments are
authorized. The best V0 configuration may be promoted to the main dataset if
its run is stable and it improves the frozen metrics. All outputs must use new
directories; adapters and run evidence must not be overwritten.

Tokenizer inspection found that some long-context prompts exceed 768 tokens;
right truncation would remove their assistant targets. Smoke therefore uses
1024 tokens. The 768-token layer/rank experiments use a deterministic view that
retains only complete rendered chats at or below 768; the 1024 context
experiment uses full V0.
