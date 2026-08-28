#!/usr/bin/env python3
"""Download and pin the Gretel source dataset without printing its content."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Keep direct script execution usable even when an editable-install .pth file is
# disabled by a hardened Python environment.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.data.constants import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATASET_ID,
    DEFAULT_LOCK_FILE,
    DEFAULT_REQUESTED_REVISION,
    DEFAULT_SNAPSHOT_ROOT,
)
from pii_redactor.data.download import DownloadConfig, DownloadError, download_source_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--revision", default=DEFAULT_REQUESTED_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK_FILE)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Validate and reuse the snapshot referenced by the existing lock.",
    )
    parser.add_argument(
        "--refresh-revision",
        action="store_true",
        help="Allow a changed remote SHA to create and select a new snapshot.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DownloadConfig(
        dataset_id=args.dataset_id,
        requested_revision=args.revision,
        cache_dir=args.cache_dir,
        snapshot_root=args.snapshot_root,
        lock_file=args.lock_file,
        offline=args.offline,
        refresh_revision=args.refresh_revision,
    )
    try:
        outcome = download_source_dataset(config)
    except DownloadError as exc:
        print(f"Dataset acquisition failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # keep the CLI failure content-free
        print(f"Dataset acquisition failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Resolved revision: {outcome.resolved_revision}")
    print(f"Snapshot: {outcome.snapshot_path}")
    print(f"Source lock: {outcome.lock_file}")
    print(
        "Split counts: "
        + ", ".join(f"{name}={count}" for name, count in outcome.split_counts.items())
    )
    print(f"Reused existing snapshot: {str(outcome.reused_snapshot).lower()}")
    print(f"Review blockers: {len(outcome.review_blockers)}")
    print("Next command: python scripts/data/inspect_dataset.py")
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
