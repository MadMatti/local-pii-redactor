#!/usr/bin/env python3
"""Run one MLX-LM training config with immutable local evidence and checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TrainingRunError(RuntimeError):
    pass


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


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _parse_log(log_text: str) -> dict[str, Any]:
    train_rows = []
    validation_rows = []
    for line in log_text.splitlines():
        train_match = re.search(
            r"Iter (\d+): Train loss ([0-9.eE+-]+).*?It/sec ([0-9.eE+-]+).*?"
            r"Tokens/sec ([0-9.eE+-]+).*?Peak mem ([0-9.eE+-]+) GB",
            line,
        )
        if train_match:
            train_rows.append(
                {
                    "iteration": int(train_match.group(1)),
                    "loss": float(train_match.group(2)),
                    "iterations_per_second": float(train_match.group(3)),
                    "tokens_per_second": float(train_match.group(4)),
                    "peak_memory_gb": float(train_match.group(5)),
                }
            )
        validation_match = re.search(
            r"Iter (\d+): Val loss ([0-9.eE+-]+)", line
        )
        if validation_match:
            validation_rows.append(
                {
                    "iteration": int(validation_match.group(1)),
                    "loss": float(validation_match.group(2)),
                }
            )
    return {
        "train_reports": train_rows,
        "validation_reports": validation_rows,
        "final_train_loss": train_rows[-1]["loss"] if train_rows else None,
        "final_validation_loss": (
            validation_rows[-1]["loss"] if validation_rows else None
        ),
        "peak_memory_gb": max(
            (row["peak_memory_gb"] for row in train_rows), default=None
        ),
    }


def _validate_inputs(config: dict[str, Any], run_dir: Path) -> tuple[Path, Path]:
    required = {
        "model",
        "data",
        "train",
        "fine_tune_type",
        "adapter_path",
        "iters",
        "max_seq_length",
        "mask_prompt",
    }
    missing = required - set(config)
    if missing:
        raise TrainingRunError(f"configuration is missing: {sorted(missing)}")
    if config["train"] is not True or config["mask_prompt"] is not True:
        raise TrainingRunError("training and prompt masking must both be enabled")
    model_path = PROJECT_ROOT / config["model"]
    data_path = PROJECT_ROOT / config["data"]
    adapter_path = PROJECT_ROOT / config["adapter_path"]
    if not model_path.is_dir():
        raise TrainingRunError(f"model directory does not exist: {model_path}")
    for split in ("train", "valid", "test"):
        if not (data_path / f"{split}.jsonl").is_file():
            raise TrainingRunError(f"dataset split does not exist: {split}")
    validation_stats = PROJECT_ROOT / "data/processed/validation/statistics.json"
    if not validation_stats.is_file():
        raise TrainingRunError("prepared-data validation statistics are missing")
    validation = json.loads(validation_stats.read_text(encoding="utf-8"))
    if validation.get("validation", {}).get("status") != "passed":
        raise TrainingRunError("prepared-data validation status is not passed")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise TrainingRunError(f"run directory is not empty: {run_dir}")
    if adapter_path.exists() and any(adapter_path.iterdir()):
        raise TrainingRunError(f"adapter directory is not empty: {adapter_path}")
    return data_path, adapter_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise TrainingRunError("configuration must be a YAML object")
        data_path, adapter_path = _validate_inputs(config, run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        started = datetime.now(UTC)
        dataset_hashes = {
            split: _sha256(data_path / f"{split}.jsonl")
            for split in ("train", "valid", "test")
        }
        start_manifest = {
            "schema_version": 1,
            "status": "running",
            "started_at_utc": started.isoformat(),
            "git_commit": _git_head(),
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "config": config,
            "dataset_hashes": dataset_hashes,
            "dataset_manifest_sha256": _sha256(
                PROJECT_ROOT / "data/processed/manifest.json"
            ),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        }
        _write_json(run_dir / "manifest.json", start_manifest)
        (run_dir / "config.yaml").write_text(
            config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        command = [
            sys.executable,
            "-m",
            "mlx_lm",
            "lora",
            "--config",
            str(config_path),
        ]
        log_path = run_dir / "training.log"
        captured: list[str] = []
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                captured.append(line)
                log_handle.write(line)
                log_handle.flush()
                print(line, end="", flush=True)
            return_code = process.wait()
        completed = datetime.now(UTC)
        log_text = "".join(captured)
        training_metrics = _parse_log(log_text)
        adapter_files = {}
        if adapter_path.is_dir():
            adapter_files = {
                str(path.relative_to(PROJECT_ROOT)): {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in sorted(adapter_path.rglob("*"))
                if path.is_file()
            }
        status = "completed" if return_code == 0 else "failed"
        manifest = {
            **start_manifest,
            "status": status,
            "completed_at_utc": completed.isoformat(),
            "duration_seconds": round((completed - started).total_seconds(), 3),
            "return_code": return_code,
            "training_metrics": training_metrics,
            "log_sha256": _sha256(log_path),
            "adapter_files": adapter_files,
        }
        _write_json(run_dir / "manifest.json", manifest)
        if return_code != 0:
            raise TrainingRunError(f"MLX-LM exited with status {return_code}")
        required_adapter = adapter_path / "adapters.safetensors"
        required_config = adapter_path / "adapter_config.json"
        if not required_adapter.is_file() or not required_config.is_file():
            raise TrainingRunError("completed run did not produce a loadable adapter")
    except (OSError, ValueError, KeyError, TrainingRunError) as exc:
        print(f"Training run failed: {exc}", file=sys.stderr)
        return 1
    print(f"Run manifest: {run_dir / 'manifest.json'}")
    print(f"Adapter: {adapter_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
