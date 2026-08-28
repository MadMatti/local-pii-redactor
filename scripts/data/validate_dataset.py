#!/usr/bin/env python3
"""Validate all prepared dataset stages and write statistics and a report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.data.constants import DEFAULT_PROCESSED_DIR, DEFAULT_TOKENIZER_PATH
from pii_redactor.data.validation import (
    DatasetValidationError,
    validate_prepared_datasets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_prepared_datasets(
            output_dir=args.output_dir, tokenizer_path=args.tokenizer
        )
    except (DatasetValidationError, OSError, ValueError) as exc:
        print(f"Dataset validation failed to run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Validation report: {result.report_path}")
    print(f"Statistics: {result.statistics_path}")
    print(f"Errors: {result.error_count}")
    print(f"Warnings: {result.warning_count}")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
