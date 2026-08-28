#!/usr/bin/env python3
"""Inspect the pinned Gretel snapshot and generate a human-review bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Keep direct script execution usable even when an editable-install .pth file is
# disabled by a hardened Python environment.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.data.constants import DEFAULT_LOCK_FILE, DEFAULT_TOKENIZER_PATH
from pii_redactor.data.inspection import (
    InspectionConfig,
    InspectionError,
    inspect_source_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK_FILE)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer-batch-size", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.tokenizer_batch_size < 1:
        print("Inspection failed: tokenizer batch size must be positive", file=sys.stderr)
        return 1
    config = InspectionConfig(
        lock_file=args.lock_file,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        seed=args.seed,
        tokenizer_batch_size=args.tokenizer_batch_size,
    )
    try:
        outcome = inspect_source_dataset(config)
    except InspectionError as exc:
        print(f"Inspection failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # keep CLI output content-free
        print(f"Inspection failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Resolved revision: {outcome.resolved_revision}")
    print(f"Records inspected: {outcome.record_count}")
    print(f"Quality issues: {outcome.issue_count}")
    print(f"Review blockers: {outcome.blocker_count}")
    print(f"Review samples: {outcome.review_sample_count}")
    print(f"Review bundle: {outcome.output_dir}")
    if outcome.blocker_count:
        print("Human review is required before dataset conversion.")
    else:
        print("Inspection passed; human approval is still required by the project gate.")
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
