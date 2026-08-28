#!/usr/bin/env python3
"""Run a gold-output evaluator self-check; all strict metrics must equal one."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data/processed/test.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation/results/oracle-check",
    )
    parser.add_argument("--expected-dataset-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = args.output_dir / "predictions.jsonl"
        with args.dataset.open(encoding="utf-8") as source, predictions_path.open(
            "w", encoding="utf-8"
        ) as destination:
            for line in source:
                record = json.loads(line)
                destination.write(
                    json.dumps(
                        {
                            "sample_id": record["sample_id"],
                            "raw_output": record["messages"][2]["content"],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        result = evaluate_predictions(
            args.dataset,
            predictions_path,
            expected_dataset_sha256=args.expected_dataset_sha256,
        )
        write_evaluation(result, args.output_dir, title="Evaluator oracle check")
    except (EvaluationError, OSError, ValueError, KeyError) as exc:
        print(f"Oracle check failed: {exc}", file=sys.stderr)
        return 1
    required = (
        result.metrics["micro"]["f1"],
        result.metrics["documents"]["complete_document_recall"],
        result.metrics["documents"]["exact_document_match"],
        result.metrics["validity"]["json_valid_rate"],
        result.metrics["validity"]["schema_valid_rate"],
        result.metrics["validity"]["source_order_valid_rate"],
    )
    if any(value != 1.0 for value in required):
        print(f"Oracle check failed strict metrics: {required}", file=sys.stderr)
        return 2
    print(f"Oracle check passed for {result.metrics['dataset']['records']} records")
    print(f"Report: {args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
