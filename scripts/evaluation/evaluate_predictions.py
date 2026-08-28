#!/usr/bin/env python3
"""Evaluate complete predictions against a frozen prepared JSONL split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.evaluation.metrics import (
    EvaluationError,
    evaluate_predictions,
    write_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--title", default="PII extraction evaluation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = evaluate_predictions(
            args.dataset,
            args.predictions,
            expected_dataset_sha256=args.expected_dataset_sha256,
        )
        metrics, samples, report = write_evaluation(
            result, args.output_dir, title=args.title
        )
    except (EvaluationError, OSError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Metrics: {metrics}")
    print(f"Sample results: {samples}")
    print(f"Report: {report}")
    print(f"Micro F1: {result.metrics['micro']['f1']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
