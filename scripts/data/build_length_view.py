#!/usr/bin/env python3
"""Build a deterministic prepared-data view that cannot truncate MLX targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.data.constants import DEFAULT_TOKENIZER_PATH
from pii_redactor.schema import EntityType


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_seq_length < 1:
        print("Length-view build failed: max length must be positive", file=sys.stderr)
        return 1
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer), local_files_only=True, trust_remote_code=False
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        split_stats: dict[str, Any] = {}
        artifacts: dict[str, Any] = {}
        for split in ("train", "valid", "test"):
            source_path = args.source_dir / f"{split}.jsonl"
            output_path = args.output_dir / f"{split}.jsonl"
            kept: list[str] = []
            rejected = 0
            labels: Counter[str] = Counter()
            negatives = 0
            maximum = 0
            for line in source_path.open(encoding="utf-8"):
                record = json.loads(line)
                token_ids = tokenizer.apply_chat_template(
                    record["messages"], tokenize=True, return_dict=False
                )
                length = len(token_ids)
                maximum = max(maximum, length)
                if length > args.max_seq_length:
                    rejected += 1
                    continue
                kept.append(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                spans = record["metadata"]["entity_spans"]
                negatives += not spans
                labels.update(item["type"] for item in spans)
            output_path.write_text("".join(kept), encoding="utf-8")
            if set(labels) != {label.value for label in EntityType}:
                raise ValueError(f"{split}: filtered view lost canonical label coverage")
            ratio = negatives / len(kept)
            if not 0.20 <= ratio <= 0.30:
                raise ValueError(f"{split}: filtered negative ratio is {ratio:.4f}")
            split_stats[split] = {
                "source_records": len(kept) + rejected,
                "kept_records": len(kept),
                "rejected_records": rejected,
                "negative_records": negatives,
                "negative_ratio": round(ratio, 8),
                "source_max_tokens": maximum,
                "label_occurrences": dict(sorted(labels.items())),
            }
            artifacts[output_path.name] = {
                "bytes": output_path.stat().st_size,
                "sha256": _sha256(output_path),
            }
        manifest = {
            "schema_version": 1,
            "source_dir": str(args.source_dir),
            "source_hashes": {
                split: _sha256(args.source_dir / f"{split}.jsonl")
                for split in ("train", "valid", "test")
            },
            "tokenizer": str(args.tokenizer),
            "max_seq_length": args.max_seq_length,
            "rule": "retain only records whose complete rendered chat length is <= max_seq_length",
            "splits": split_stats,
            "artifacts": artifacts,
        }
        _write_json(args.output_dir / "manifest.json", manifest)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Length-view build failed: {exc}", file=sys.stderr)
        return 1
    print(f"Length view: {args.output_dir}")
    for split, stats in split_stats.items():
        print(
            f"{split}: kept={stats['kept_records']}, "
            f"rejected={stats['rejected_records']}, "
            f"negative_ratio={stats['negative_ratio']:.2%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
