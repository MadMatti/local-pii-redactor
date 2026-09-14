#!/usr/bin/env python3
"""Partition a frozen test by prior selection or a length view, preserving order.

The complementary view is not an untouched holdout: earlier baseline evaluation
may have covered the full test. This command records only prior adapter selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


class PartitionError(ValueError):
    """Invalid frozen inputs or an existing output that cannot be reused."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PartitionError("duplicate JSON object field")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PartitionError("non-finite JSON constant")


def _read_rows(data: bytes, *, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise PartitionError(f"{description}: invalid UTF-8") from exc
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(
                line, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
            if not isinstance(row, dict):
                raise PartitionError("record must be a JSON object")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise PartitionError("sample_id must be a nonempty string")
            if sample_id in seen:
                raise PartitionError("duplicate sample_id")
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise PartitionError("messages must be a nonempty list")
            if any(
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)
                for message in messages
            ):
                raise PartitionError("invalid rendered message")
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                raise PartitionError("metadata must be an object")
            spans = metadata.get("entity_spans")
            if not isinstance(spans, list) or any(
                not isinstance(span, dict)
                or not isinstance(span.get("type"), str)
                or not span["type"]
                for span in spans
            ):
                raise PartitionError("invalid entity_spans")
            if (
                type(metadata.get("is_negative")) is not bool
                or metadata["is_negative"] != (not spans)
            ):
                raise PartitionError("is_negative must agree with entity_spans")
        except (json.JSONDecodeError, PartitionError) as exc:
            # Do not echo IDs, malformed JSON, document text, or entity values.
            detail = "invalid JSON" if isinstance(exc, json.JSONDecodeError) else str(exc)
            raise PartitionError(f"{description} line {line_number}: {detail}") from exc
        seen.add(sample_id)
        rows.append(row)
    return rows


def _json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent)
        + "\n"
    ).encode("utf-8")


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entities: Counter[str] = Counter()
    for row in rows:
        entities.update(span["type"] for span in row["metadata"]["entity_spans"])
    return {
        "record_count": len(rows),
        "negative_count": sum(row["metadata"]["is_negative"] for row in rows),
        "entity_counts": dict(sorted(entities.items())),
    }


def partition_test(
    dataset: Path,
    previously_used_dataset: Path,
    output_dir: Path,
    *,
    expected_dataset_sha256: str,
    partition_role: str = "prior_selection",
) -> dict[str, Any]:
    """Write exhaustive, disjoint views or verify an unchanged previous result."""
    if partition_role not in {"prior_selection", "length_filter"}:
        raise PartitionError("unknown partition role")
    data = dataset.read_bytes()
    dataset_hash = _sha256(data)
    if dataset_hash != expected_dataset_sha256:
        raise PartitionError("frozen dataset SHA-256 does not match expected value")
    prior_data = previously_used_dataset.read_bytes()
    rows = _read_rows(data, description="full dataset")
    prior_rows = _read_rows(prior_data, description="previously used dataset")
    if not rows:
        raise PartitionError("full dataset is empty")
    indexed_rows = {row["sample_id"]: row for row in rows}
    prior_ids = {row["sample_id"] for row in prior_rows}
    absent = prior_ids - indexed_rows.keys()
    if absent:
        raise PartitionError(
            f"previously used IDs are not a subset of full dataset ({len(absent)} missing)"
        )
    for row in prior_rows:
        if row != indexed_rows[row["sample_id"]]:
            raise PartitionError("shared sample_id has changed record content")

    included_name, excluded_name = (
        ("previously_used", "not_previously_used")
        if partition_role == "prior_selection" else ("retained", "excluded")
    )
    groups = {
        included_name: [row for row in rows if row["sample_id"] in prior_ids],
        excluded_name: [row for row in rows if row["sample_id"] not in prior_ids],
    }
    payloads = {
        f"{name}.jsonl": b"".join(_json_bytes(row) for row in group)
        for name, group in groups.items()
    }
    manifest = {
        "schema_version": 1,
        "purpose": (
            "Partition by prior adapter-selection exposure, in full-test order."
            if partition_role == "prior_selection"
            else "Partition by membership in the supplied length-filtered reference, in full-test order."
        ),
        "limitation": (
            "not_previously_used means not used for prior adapter selection; "
            "earlier baseline evaluations may cover these records. "
            "It is not an untouched holdout."
        ) if partition_role == "prior_selection" else (
            "retained/excluded indicate membership in the supplied length view; "
            "no tokenizer or length filtering is performed by this command."
        ),
        "inputs": {
            "dataset": {"sha256": dataset_hash, **_summary(rows)},
            ("previously_used_dataset" if partition_role == "prior_selection" else "reference_dataset"): {
                "sha256": _sha256(prior_data),
                **_summary(prior_rows),
            },
        },
        "outputs": {
            name: {
                "path": f"{name}.jsonl",
                "sha256": _sha256(payloads[f"{name}.jsonl"]),
                **_summary(group),
            }
            for name, group in groups.items()
        },
        "disjoint": True,
        "exhaustive": True,
    }
    if partition_role == "length_filter":
        manifest["partition_role"] = partition_role
    payloads["manifest.json"] = _json_bytes(manifest, indent=2)
    if output_dir.is_symlink():
        raise PartitionError("output directory must not be a symlink")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        if not output_dir.is_dir() or any(
            not (output_dir / name).is_file()
            or (output_dir / name).is_symlink()
            or (output_dir / name).read_bytes() != payload
            for name, payload in payloads.items()
        ):
            raise PartitionError("existing output differs from frozen inputs or was modified")
        return manifest

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".test-partition-", dir=output_dir.parent) as temp:
        staged = Path(temp) / "partition"
        staged.mkdir()
        for name, payload in payloads.items():
            (staged / name).write_bytes(payload)
        os.rename(staged, output_dir)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--previously-used-dataset", "--reference-dataset", type=Path, required=True)
    parser.add_argument("--partition-role", choices=("prior_selection", "length_filter"), default="prior_selection")
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = partition_test(
            args.dataset,
            args.previously_used_dataset,
            args.output_dir,
            expected_dataset_sha256=args.expected_dataset_sha256,
            partition_role=args.partition_role,
        )
    except (OSError, ValueError) as exc:
        print(f"Test partition failed: {exc}", file=sys.stderr)
        return 1
    for name, item in manifest["outputs"].items():
        print(f"{name}: {item['record_count']} records, {item['negative_count']} negatives")
    print(f"Manifest: {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
