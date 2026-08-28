#!/usr/bin/env python3
"""Run and evaluate the deterministic regex baseline on prepared JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.evaluation.metrics import (
    EvaluationError,
    evaluate_predictions,
    sha256_file,
    write_evaluation,
)
from pii_redactor.evaluation.regex_baseline import (
    REGEX_BASELINE_VERSION,
    regex_response,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data/processed/test.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation/results/regex-v1",
    )
    parser.add_argument("--expected-dataset-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = [json.loads(line) for line in args.dataset.open(encoding="utf-8")]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = args.output_dir / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                prediction = {
                    "sample_id": row["sample_id"],
                    "raw_output": regex_response(row["messages"][1]["content"]),
                }
                handle.write(
                    json.dumps(prediction, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
        result = evaluate_predictions(
            args.dataset,
            predictions_path,
            expected_dataset_sha256=args.expected_dataset_sha256,
        )
        write_evaluation(result, args.output_dir, title="Regex baseline V1")
        manifest = {
            "schema_version": 1,
            "baseline_version": REGEX_BASELINE_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset": {
                "path": str(args.dataset),
                "sha256": sha256_file(args.dataset),
            },
            "predictions": {
                "path": str(predictions_path),
                "sha256": sha256_file(predictions_path),
            },
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (EvaluationError, OSError, ValueError, KeyError) as exc:
        print(f"Regex baseline failed: {exc}", file=sys.stderr)
        return 1
    print(f"Predictions: {predictions_path}")
    print(f"Report: {args.output_dir / 'report.md'}")
    print(f"Micro F1: {result.metrics['micro']['f1']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
