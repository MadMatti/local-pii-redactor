#!/usr/bin/env python3
"""Generate an auditable deterministic preview of synthetic negative examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.data.constants import DATASET_BUILD_SEED
from pii_redactor.data.synthetic import generate_negative_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DATASET_BUILD_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "intermediate" / "synthetic-negative-preview.jsonl",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count < 1:
        print("Negative generation failed: count must be positive", file=sys.stderr)
        return 1
    samples = generate_negative_samples(
        split=args.split, count=args.count, seed=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(
                sample.to_json_record(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for sample in samples
        ),
        encoding="utf-8",
    )
    print(f"Generated records: {len(samples)}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
