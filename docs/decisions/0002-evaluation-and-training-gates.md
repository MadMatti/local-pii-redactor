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

## Execution evidence

Gate G4 passed on 2026-08-28. The oracle scored perfectly on all strict metrics.
On the frozen 2,000-record test set, regex achieved 0.3265 exact recall and
0.3994 F1; untouched Qwen achieved 0.3506 exact recall and 0.4196 F1. The
untouched model's 0.6405 schema-valid rate and 0.6580 PII-free false-positive
rate confirm that it is not deployable without adaptation.

Gate G5 passed on 2026-08-28. The 100-iteration smoke run completed with finite
0.324 train loss, 0.301 validation loss, and 2.766 GB peak memory. Its adapter
reloaded and generated the full frozen smoke set using the evaluation chat
template. The smoke adapter's 0.0637 exact recall does not pass the quality gate,
so it is retained only as pipeline evidence and cannot be promoted. The
authorized V0 screens will determine whether additional training restores
recall while preserving strict output validity.
