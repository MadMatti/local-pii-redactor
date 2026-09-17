#!/usr/bin/env python3
"""Fuse the approved local adapter without requantization or overwriting evidence."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.model_artifacts import (
    ArtifactError, check_tokenizer_parity, inspect_floating_weights, inventory,
    preserve_approved_tokenizer, sha256_file, validate_approved_inputs,
)


def write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "models/packaging-v1.json")
    args = parser.parse_args(argv)
    manifest_path = None
    manifest = {}
    try:
        plan = json.loads(args.plan.read_text())
        print("Checking approved model and adapter hashes...", flush=True)
        base, adapter = validate_approved_inputs(PROJECT_ROOT, plan)
        dataset = PROJECT_ROOT / plan["validation"]["dataset"]
        if sha256_file(dataset) != plan["validation"]["sha256"]:
            raise ArtifactError("validation data differs from frozen hash")
        output = (PROJECT_ROOT / plan["fusion"]["output"]).resolve()
        run = (PROJECT_ROOT / plan["fusion"]["run_dir"]).resolve()
        allowed = (PROJECT_ROOT / "models/fused").resolve()
        if any(p == allowed or not p.is_relative_to(allowed) for p in (output, run)):
            raise ArtifactError("fusion outputs must be children of local models/fused")
        if output == run or output.is_relative_to(run) or run.is_relative_to(output):
            raise ArtifactError("artifact and evidence directories must be separate")
        if output.exists() or run.exists():
            raise ArtifactError("refusing to overwrite a prior fusion; choose new output paths")
        if shutil.disk_usage(PROJECT_ROOT).free < plan["fusion"]["minimum_free_bytes"]:
            raise ArtifactError("insufficient free disk for fusion plus safety reserve")
        before = {"base": inventory(base), "adapter": inventory(adapter)}
        run.mkdir(parents=True)
        command = [sys.executable, "-X", "pycache_prefix=/private/tmp/local-pii-redactor-pycache",
                   "-m", "mlx_lm", "fuse", "--model", str(base),
                   "--adapter-path", str(adapter), "--save-path", str(output), "--dequantize"]
        manifest = {
            "schema_version": 1, "status": "running", "plan_sha256": sha256_file(args.plan),
            "started_at_utc": datetime.now(UTC).isoformat(), "command": command,
            "inputs": before, "input_paths": {"base": str(base), "adapter": str(adapter)},
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                                   text=True).strip(),
            "versions": {name: version(name) for name in ("mlx", "mlx-lm", "transformers")},
            "python": platform.python_version(), "output": str(output),
            "implementation_sha256": {
                "scripts/model/fuse_adapter.py": sha256_file(Path(__file__)),
                "src/pii_redactor/model_artifacts.py": sha256_file(PROJECT_ROOT / "src/pii_redactor/model_artifacts.py"),
            },
            "note": "Dequantizes the trained 4-bit base; does not restore original prequantization weights.",
        }
        manifest_path = run / "manifest.json"
        write_manifest(manifest_path, manifest)
        started = time.monotonic()
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "TOKENIZERS_PARALLELISM": "false"}
        with (run / "fusion.log").open("w") as log:
            print("Fusing and dequantizing the approved local model...", flush=True)
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        manifest["return_code"] = result.returncode
        manifest["duration_seconds"] = round(time.monotonic() - started, 3)
        manifest["log_sha256"] = sha256_file(run / "fusion.log")
        if result.returncode:
            raise ArtifactError("MLX fusion failed; inspect the local fusion log")
        manifest["weights"] = inspect_floating_weights(output)
        manifest["tokenizer_preservation"] = preserve_approved_tokenizer(base, output, run / "generated-tokenizer")
        print("Checking floating-point tensors and frozen tokenizer parity...", flush=True)
        manifest["tokenizer_parity"] = check_tokenizer_parity(base, output, dataset)
        if before != {"base": inventory(base), "adapter": inventory(adapter)}:
            raise ArtifactError("source model or adapter changed during fusion")
        manifest["artifacts"] = inventory(output)
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
        write_manifest(manifest_path, manifest)
    except (ArtifactError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        if manifest_path:
            manifest.update(status="failed", failure_type=type(exc).__name__,
                            failure_reason=str(exc) if isinstance(exc, ArtifactError) else "See local log")
            write_manifest(manifest_path, manifest)
        print(f"Fusion failed ({type(exc).__name__}); check inputs and local evidence.", file=sys.stderr)
        return 1
    print(f"Fused model: {output}")
    print(f"Fusion manifest: {manifest_path}")
    print("Structural and tokenizer checks passed; generation parity is still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
