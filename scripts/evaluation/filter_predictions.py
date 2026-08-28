#!/usr/bin/env python3
"""Select predictions for an exact prepared-dataset subset in dataset order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.evaluation.metrics import EvaluationError, load_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        predictions = load_predictions(args.source_predictions)
        dataset_rows = [
            json.loads(line) for line in args.dataset.open(encoding="utf-8")
        ]
        missing = [
            row["sample_id"]
            for row in dataset_rows
            if row["sample_id"] not in predictions
        ]
        if missing:
            raise EvaluationError(f"source predictions are missing {len(missing)} IDs")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for row in dataset_rows:
                handle.write(
                    json.dumps(
                        {
                            "sample_id": row["sample_id"],
                            "raw_output": predictions[row["sample_id"]],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    except (EvaluationError, OSError, ValueError, KeyError) as exc:
        print(f"Prediction filtering failed: {exc}", file=sys.stderr)
        return 1
    print(f"Selected predictions: {len(dataset_rows)}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
