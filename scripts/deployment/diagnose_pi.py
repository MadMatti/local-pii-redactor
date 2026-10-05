#!/usr/bin/env python3
"""Collect a read-only Pi inventory, without installing or starting anything."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.deployment.diagnostics import collect_diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-path", type=Path, default=PROJECT_ROOT,
                        help="Existing directory on the intended model-storage filesystem")
    parser.add_argument("--output", type=Path,
                        help="New JSON file under local evaluation/results; otherwise print JSON")
    args = parser.parse_args(argv)
    try:
        if args.output is not None:
            allowed = (PROJECT_ROOT / "evaluation/results").resolve()
            target = args.output.resolve()
            if (args.output.exists() or args.output.is_symlink() or args.output.suffix != ".json"
                    or target == allowed or not target.is_relative_to(allowed)
                    or not allowed.is_relative_to(PROJECT_ROOT.resolve())):
                raise ValueError("report requires a new local evaluation/results JSON file")
        report = collect_diagnostics(args.storage_path)
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output is None:
            print(payload, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(payload)
            print(f"Diagnostics: {report['status']}")
            print(f"Report: {args.output}")
            print("Inventory only; benchmark and deployment are not approved by this report.")
    except (OSError, ValueError, TypeError) as exc:
        print(f"Pi diagnostics failed ({type(exc).__name__}); check paths and local permissions.", file=sys.stderr)
        return 1
    return 0 if report["status"] == "diagnostics_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
