#!/usr/bin/env python3
"""Generate deterministic MLX predictions and evaluate them on a frozen split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.evaluation.metrics import (
    EvaluationError,
    evaluate_predictions,
    sha256_file,
    write_evaluation,
)


def _directory_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    included = ("*.json", "*.safetensors", "*.txt", "*.model")
    files = sorted({item for pattern in included for item in path.glob(pattern)})
    if not files:
        raise EvaluationError(f"no model fingerprint files found in {path}")
    for file_path in files:
        digest.update(file_path.name.encode("utf-8"))
        digest.update(sha256_file(file_path).encode("ascii"))
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "models/base-qwen3-1.7b-4bit"
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data/processed/test.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_tokens < 1:
        print("Model baseline failed: max-tokens must be positive", file=sys.stderr)
        return 1
    try:
        dataset_hash = sha256_file(args.dataset)
        if args.expected_dataset_sha256 and dataset_hash != args.expected_dataset_sha256:
            raise EvaluationError("dataset hash differs from the frozen hash")
        model_fingerprint = _directory_fingerprint(args.model)
        adapter_fingerprint = (
            _directory_fingerprint(args.adapter) if args.adapter else None
        )
        run_config = {
            "schema_version": 1,
            "model": str(args.model.resolve()),
            "model_fingerprint": model_fingerprint,
            "adapter": str(args.adapter.resolve()) if args.adapter else None,
            "adapter_fingerprint": adapter_fingerprint,
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": dataset_hash,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "temperature": 0.0,
            "enable_thinking": False,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config_path = args.output_dir / "run_config.json"
        predictions_path = args.output_dir / "predictions.jsonl"
        if config_path.exists():
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
            if existing_config != run_config:
                raise EvaluationError("existing run configuration differs")
        else:
            _write_json(config_path, run_config)
        if predictions_path.exists() and not args.resume:
            raise EvaluationError(
                f"predictions already exist; pass --resume: {predictions_path}"
            )
        completed: set[str] = set()
        if predictions_path.exists():
            for line in predictions_path.open(encoding="utf-8"):
                row = json.loads(line)
                sample_id = row["sample_id"]
                if sample_id in completed:
                    raise EvaluationError(f"duplicate resumed prediction: {sample_id}")
                completed.add(sample_id)
        dataset_rows = [
            json.loads(line) for line in args.dataset.open(encoding="utf-8")
        ]
        expected_ids = {row["sample_id"] for row in dataset_rows}
        if not completed <= expected_ids:
            raise EvaluationError("resumed predictions contain unknown sample IDs")

        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(args.seed)
        model, tokenizer = load(
            str(args.model),
            adapter_path=str(args.adapter) if args.adapter else None,
        )
        sampler = make_sampler(temp=0.0)
        start_time = time.perf_counter()
        generated = 0
        with predictions_path.open("a", encoding="utf-8") as destination:
            for index, row in enumerate(dataset_rows, 1):
                if row["sample_id"] in completed:
                    continue
                messages = row["messages"][:2]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                sample_start = time.perf_counter()
                raw_output = generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                    verbose=False,
                )
                elapsed = time.perf_counter() - sample_start
                prediction = {
                    "sample_id": row["sample_id"],
                    "raw_output": raw_output,
                    "prompt_tokens": len(prompt),
                    "generated_tokens": len(
                        tokenizer.encode(raw_output, add_special_tokens=False)
                    ),
                    "elapsed_seconds": round(elapsed, 6),
                }
                destination.write(
                    json.dumps(
                        prediction,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                destination.flush()
                generated += 1
                if generated % 25 == 0 or index == len(dataset_rows):
                    total_done = len(completed) + generated
                    print(
                        f"Generated {total_done}/{len(dataset_rows)} records; "
                        f"elapsed={time.perf_counter() - start_time:.1f}s",
                        flush=True,
                    )
        result = evaluate_predictions(
            args.dataset,
            predictions_path,
            expected_dataset_sha256=args.expected_dataset_sha256,
        )
        write_evaluation(result, args.output_dir, title="MLX model baseline")
        prediction_rows = [
            json.loads(line) for line in predictions_path.open(encoding="utf-8")
        ]
        total_elapsed = sum(row["elapsed_seconds"] for row in prediction_rows)
        total_generated = sum(row["generated_tokens"] for row in prediction_rows)
        manifest = {
            **run_config,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "predictions_sha256": sha256_file(predictions_path),
            "records": len(prediction_rows),
            "total_generation_seconds": round(total_elapsed, 6),
            "generated_tokens": total_generated,
            "generation_tokens_per_second": round(
                total_generated / total_elapsed if total_elapsed else 0.0, 6
            ),
            "peak_memory_gb": round(float(mx.get_peak_memory()) / 1e9, 6),
        }
        _write_json(args.output_dir / "manifest.json", manifest)
    except (EvaluationError, OSError, ValueError, KeyError) as exc:
        print(f"Model baseline failed: {exc}", file=sys.stderr)
        return 1
    print(f"Report: {args.output_dir / 'report.md'}")
    print(f"Micro F1: {result.metrics['micro']['f1']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
