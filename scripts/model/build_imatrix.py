#!/usr/bin/env python3
"""Build and verify a local importance matrix from the approved calibration only."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.evaluation.metrics import EvaluationError
from pii_redactor.model_artifacts import ArtifactError, sha256_file
from scripts.model.audit_calibration import audit
from scripts.model.quantize_gguf import RESERVE_BYTES, check_parent, validate_toolchain


def native_environment():
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("LLAMA_", "GGML_", "HF_", "HUGGING_FACE_"))}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def completed_chunks(log):
    matches = re.findall(r"compute_imatrix: computing over (\d+) chunks, n_ctx=1024, batch_size=1024, n_seq=1", log)
    if len(matches) != 1 or int(matches[0]) < 1:
        raise ArtifactError("native calibration did not report the expected full-chunk workload")
    return int(matches[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "models/packaging-v1.json")
    parser.add_argument("--parent-dir", type=Path, default=PROJECT_ROOT / "models/gguf/v1-ckpt19000-bf16")
    parser.add_argument("--parent-evaluation", type=Path, default=PROJECT_ROOT / "evaluation/results/v1-gguf-bf16-smoke-valid")
    parser.add_argument("--toolchain", type=Path, default=PROJECT_ROOT / "models/gguf/toolchain-v1/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models/gguf/v1-ckpt19000-imatrix")
    args = parser.parse_args(argv)
    manifest_path, evidence = None, {}
    try:
        plan = json.loads(args.plan.read_text())
        if (plan["approval"]["status"] != "approved" or plan["approval"]["gate"] != "H-045"
                or "Q4_K_M-imatrix" not in plan["quantization"]["formats"]):
            raise ArtifactError("importance-matrix trial is not approved")
        output = args.output_dir.resolve()
        allowed = (PROJECT_ROOT / "models/gguf").resolve()
        if output == allowed or not output.is_relative_to(allowed) or args.output_dir.is_symlink() or output.exists():
            raise ArtifactError("matrix output must be a new local models/gguf directory")
        print("Verifying BF16 parity and train-only calibration provenance...", flush=True)
        checkout = validate_toolchain(PROJECT_ROOT, plan, args.toolchain)
        toolchain = json.loads(args.toolchain.read_text())
        if "build/bin/llama-imatrix" not in toolchain["binaries"]:
            raise ArtifactError("importance-matrix binary is not recorded")
        parent, parent_evidence = check_parent(PROJECT_ROOT, plan, args.parent_dir, args.parent_evaluation,
                                               args.toolchain, sha256_file(args.plan))
        calibration_audit = audit(PROJECT_ROOT, plan)
        if calibration_audit["status"] != "passed":
            raise ArtifactError("calibration overlaps frozen held-out data")
        available = shutil.disk_usage(PROJECT_ROOT).free
        if available < RESERVE_BYTES + 256 * 1024**2:
            raise ArtifactError("importance-matrix generation requires 4.25 GiB free including the disk reserve")
        calibration = (PROJECT_ROOT / plan["quantization"]["calibration_path"]).resolve()
        target = output / "importance.gguf"
        command = [str(checkout / "build/bin/llama-imatrix"), "--model", str(parent),
                   "--file", str(calibration), "--output-file", str(target), "--output-format", "gguf",
                   "--ctx-size", "1024", "--batch-size", "1024", "--ubatch-size", "512",
                   "--parallel", "1", "--threads", "4", "--gpu-layers", "all", "--fit", "off",
                   "--flash-attn", "on", "--chunks", "-1", "--no-ppl", "--output-frequency", "10",
                   "--save-frequency", "0", "--seed", "42", "--offline"]
        output.mkdir(parents=True)
        manifest_path = output / "manifest.json"
        evidence = {"schema_version": 1, "status": "running", "command": command, "model": str(target),
                    "plan_sha256": sha256_file(args.plan), "parent_sha256": parent_evidence["model_sha256"],
                    "parent": parent_evidence, "toolchain_sha256": sha256_file(args.toolchain),
                    "calibration_sha256": plan["quantization"]["calibration_sha256"],
                    "calibration_audit": calibration_audit, "free_bytes_before": available,
                    "started_at_utc": datetime.now(UTC).isoformat(),
                    "implementation_sha256": sha256_file(Path(__file__)),
                    "inspector_sha256_at_start": sha256_file(PROJECT_ROOT / "scripts/model/inspect_imatrix.py")}
        manifest_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        started = time.monotonic()
        print("Collecting importance statistics locally; source text stays in local files.", flush=True)
        log_path = output / "imatrix.log"
        with log_path.open("x") as log:
            result = subprocess.run(command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, env=native_environment())
        evidence.update(return_code=result.returncode, generation_seconds=round(time.monotonic()-started, 3))
        if result.returncode:
            raise ArtifactError("native importance-matrix generation failed; inspect the local log")
        chunks = completed_chunks(log_path.read_text())
        inspector_sha256 = sha256_file(PROJECT_ROOT / "scripts/model/inspect_imatrix.py")
        inspect_command = [str(PROJECT_ROOT / ".venv-conversion/bin/python"), "-X",
            "pycache_prefix=/private/tmp/local-pii-redactor-conversion-pycache",
            str(PROJECT_ROOT / "scripts/model/inspect_imatrix.py"), "--model", str(target),
            "--config", str(PROJECT_ROOT / plan["fusion"]["output"] / "config.json"),
            "--calibration", str(calibration), "--chunks", str(chunks), "--output", str(output / "metadata.json")]
        with (output / "verification.log").open("x") as log:
            result = subprocess.run(inspect_command, cwd=PROJECT_ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, env=native_environment())
        if result.returncode:
            raise ArtifactError("importance-matrix coverage verification failed; inspect the local log")
        metadata = json.loads((output / "metadata.json").read_text())
        if (metadata.get("status") != "passed" or metadata.get("implementation_sha256") != inspector_sha256
                or sha256_file(PROJECT_ROOT / "scripts/model/inspect_imatrix.py") != inspector_sha256):
            raise ArtifactError("importance-matrix inspector changed during verification")
        if (sha256_file(parent) != evidence["parent_sha256"]
                or sha256_file(calibration) != evidence["calibration_sha256"]):
            raise ArtifactError("parent or calibration changed during importance collection")
        evidence.update(status="completed", model_sha256=sha256_file(target), model_bytes=target.stat().st_size,
                        completed_at_utc=datetime.now(UTC).isoformat(), chunk_count=chunks,
                        inspector_sha256=inspector_sha256,
                        metadata_sha256=sha256_file(output / "metadata.json"),
                        log_sha256=sha256_file(log_path), free_bytes_after=shutil.disk_usage(PROJECT_ROOT).free)
        manifest_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    except (ArtifactError, EvaluationError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        if manifest_path:
            evidence.update(status="failed", failure_type=type(exc).__name__)
            if isinstance(exc, ArtifactError):
                evidence["failure_reason"] = str(exc)
            manifest_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(f"Importance-matrix generation stopped ({type(exc).__name__}).", file=sys.stderr)
        if isinstance(exc, ArtifactError):
            print(str(exc), file=sys.stderr)
        return 1
    print(f"Verified importance matrix: {target}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
