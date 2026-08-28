#!/usr/bin/env python3
"""Rank comparable evaluation reports using the approved privacy-first order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "evaluations",
        nargs="+",
        help="NAME=path/to/metrics.json entries evaluated on the same dataset",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        row["micro_recall"],
        row["complete_document_recall"],
        row["micro_precision"],
        row["schema_valid_rate"],
        row["micro_f1"],
        -row["pii_free_false_positive_rate"],
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows: list[dict[str, Any]] = []
    dataset_hash: str | None = None
    try:
        for value in args.evaluations:
            if "=" not in value:
                raise ValueError(f"evaluation must use NAME=PATH: {value}")
            name, raw_path = value.split("=", 1)
            path = Path(raw_path)
            metrics = json.loads(path.read_text(encoding="utf-8"))
            current_hash = metrics["dataset"]["sha256"]
            if dataset_hash is None:
                dataset_hash = current_hash
            elif current_hash != dataset_hash:
                raise ValueError("evaluation reports use different dataset hashes")
            rows.append(
                {
                    "name": name,
                    "metrics_path": str(path),
                    "micro_precision": metrics["micro"]["precision"],
                    "micro_recall": metrics["micro"]["recall"],
                    "micro_f1": metrics["micro"]["f1"],
                    "complete_document_recall": metrics["documents"][
                        "complete_document_recall"
                    ],
                    "pii_free_false_positive_rate": metrics["documents"][
                        "pii_free_false_positive_rate"
                    ],
                    "schema_valid_rate": metrics["validity"]["schema_valid_rate"],
                }
            )
        rows.sort(key=_selection_key, reverse=True)
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = {
            "schema_version": 1,
            "selection_order": [
                "micro exact recall",
                "complete-document recall",
                "micro exact precision",
                "schema validity",
                "micro exact F1",
                "lower PII-free false-positive rate",
            ],
            "dataset_sha256": dataset_hash,
            "ranking": rows,
        }
        (args.output_dir / "comparison.json").write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        lines = [
            "# Evaluation comparison",
            "",
            "| Rank | Candidate | Recall | Complete-doc recall | Precision | F1 | Schema valid | PII-free FP |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| {row['rank']} | {row['name']} | {row['micro_recall']:.4f} | "
                f"{row['complete_document_recall']:.4f} | "
                f"{row['micro_precision']:.4f} | {row['micro_f1']:.4f} | "
                f"{row['schema_valid_rate']:.4f} | "
                f"{row['pii_free_false_positive_rate']:.4f} |"
            )
        (args.output_dir / "comparison.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Evaluation comparison failed: {exc}", file=sys.stderr)
        return 1
    print(f"Selected candidate: {rows[0]['name']}")
    print(f"Comparison: {args.output_dir / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
