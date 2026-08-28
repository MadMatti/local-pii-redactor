#!/usr/bin/env python3
"""Build deterministic MLX chat datasets from the pinned Gretel snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.data.builder import DatasetBuildError, build_datasets
from pii_redactor.data.constants import (
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_LOCK_FILE,
    DEFAULT_PROCESSED_DIR,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_datasets(
            lock_file=args.lock_file,
            output_dir=args.output_dir,
            calibration_dir=args.calibration_dir,
        )
    except (DatasetBuildError, OSError, ValueError) as exc:
        print(f"Dataset build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset manifest: {result.manifest_path}")
    print(f"Rejected source records: {result.rejected_count}")
    print(f"Rejected-source report: {result.rejected_path}")
    for stage, split_counts in result.counts.items():
        rendered = ", ".join(
            f"{split}={count}" for split, count in split_counts.items()
        )
        print(f"{stage}: {rendered}")
    print("Next command: python scripts/data/validate_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
