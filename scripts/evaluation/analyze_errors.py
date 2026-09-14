#!/usr/bin/env python3
"""Write aggregate error groups and capped local review samples for frozen predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.evaluation.error_analysis import analyze_errors, write_error_analysis
from pii_redactor.evaluation.metrics import EvaluationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-group", type=int, default=20)
    parser.add_argument("--max-review-records", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = analyze_errors(
            args.dataset,
            args.predictions,
            expected_dataset_sha256=args.expected_dataset_sha256,
            seed=args.seed,
            samples_per_group=args.samples_per_group,
            max_review_records=args.max_review_records,
        )
        root = (PROJECT_ROOT / "evaluation" / "results").resolve()
        output_dir = args.output_dir or root / (
            f"error-analysis-{result.statistics['dataset']['sha256'][:12]}-"
            f"{result.statistics['predictions']['sha256'][:12]}-"
            f"s{args.seed}-g{args.samples_per_group}-n{args.max_review_records}"
        )
        if output_dir.resolve() == root or not output_dir.resolve().is_relative_to(root):
            raise EvaluationError("output must be a subdirectory of local evaluation/results")
        manifest = write_error_analysis(result, output_dir)
    except (EvaluationError, OSError, ValueError, KeyError, TypeError) as exc:
        # Some underlying coverage/parser errors contain IDs or values. Do not
        # echo exception details into logs; the local artifacts hold evidence.
        print(
            f"Error analysis failed ({type(exc).__name__}). Check the frozen input hash, "
            "complete unique prediction IDs, prepared schema, and a writable unused "
            "evaluation/results subdirectory.",
            file=sys.stderr,
        )
        return 1
    print(f"Records analyzed: {result.statistics['dataset']['records']}")
    print(f"False negatives: {result.statistics['totals']['fn']}")
    print(f"False positives: {result.statistics['totals']['fp']}")
    print(f"Review records: {result.statistics['totals']['review_records']}")
    print(f"Error analysis manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
