#!/usr/bin/env python3
"""Produce one approved quantization from the verified, parity-gated BF16 parent."""

from __future__ import annotations

import argparse
import json
import os
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
from pii_redactor.model_artifacts import ArtifactError, inspect_floating_weights, sha256_file
from scripts.evaluation.check_parity import compare

FORMATS = ("Q8_0", "Q5_K_M", "Q4_K_M", "Q4_K_M-imatrix")
RESERVE_BYTES = 4 * 1024**3


def check_parent(root, plan, parent_dir, evaluation_dir, toolchain_path, expected_plan_sha256):
    manifest_path = parent_dir / "verified-conversion.json"
    if not manifest_path.exists():
        manifest_path = parent_dir / "manifest.json"
    parent = json.loads(manifest_path.read_text())
    model = parent_dir / "model-bf16.gguf"
    if (parent.get("status") != "completed" or parent.get("plan_sha256") != expected_plan_sha256
            or model.is_symlink() or sha256_file(model) != parent["model_sha256"]):
        raise ArtifactError("BF16 parent is not verified or its checksum changed")
    if parent["toolchain_sha256"] != sha256_file(toolchain_path):
        raise ArtifactError("quantization must use the recorded conversion toolchain")
    metadata = json.loads((parent_dir / "metadata.json").read_text())
    if (metadata.get("status") != "passed" or set(metadata["tensor_dtypes"]) - {"BF16", "F32"}
            or metadata.get("model_sha256") != parent["model_sha256"]
            or sha256_file(parent_dir / "metadata.json") != parent["metadata_sha256"]):
        raise ArtifactError("quantization requires a floating-point common parent")
    reference = root / "evaluation/results/v1-fused-bf16-smoke-valid/predictions.jsonl"
    candidate = evaluation_dir / "predictions.jsonl"
    run = json.loads((evaluation_dir / "manifest.json").read_text())
    expected = {"model_sha256": parent["model_sha256"], "dataset_sha256": plan["validation"]["sha256"],
                "toolchain_sha256": sha256_file(toolchain_path), **plan["generation"]}
    if run.get("status") != "completed" or any(run.get(k) != v for k, v in expected.items()):
        raise ArtifactError("BF16 generation evidence differs from the approved configuration")
    if sha256_file(candidate) != run["predictions_sha256"] or sha256_file(reference) != parent["generation_gate"]["fused_predictions_sha256"]:
        raise ArtifactError("conversion-parity predictions changed")
    gate = compare(root / plan["validation"]["dataset"], reference, candidate, plan["validation"]["sha256"])
    if gate["status"] != "passed":
        raise ArtifactError("BF16 conversion parity must pass before quantization")
    return model, {"manifest_sha256": sha256_file(manifest_path), "model_sha256": parent["model_sha256"],
                   "validation_predictions_sha256": run["predictions_sha256"], "gates": gate["gates"]}


def validate_toolchain(root, plan, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    checkout = root / plan["llama_cpp"]["path"]
    if manifest.get("status") != "recorded" or manifest.get("revision") != plan["llama_cpp"]["revision"]:
        raise ArtifactError("approved native toolchain is required")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=checkout, text=True)
    if revision != manifest["revision"] or dirty:
        raise ArtifactError("native toolchain source changed after recording")
    for name, details in manifest["binaries"].items():
        file = checkout / name
        if not file.resolve().is_relative_to((checkout / "build").resolve()) or sha256_file(file) != details["sha256"]:
            raise ArtifactError("native binary or shared library changed")
    if "build/bin/llama-quantize" not in manifest["binaries"]:
        raise ArtifactError("quantizer binary is not recorded")
    return checkout


def minimum_free_bytes(parent_bytes, format):
    ratio = {"Q8_0": 0.60, "Q5_K_M": 0.45, "Q4_K_M": 0.38, "Q4_K_M-imatrix": 0.38}[format]
    return RESERVE_BYTES + int(parent_bytes * ratio) + 64 * 1024**2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=FORMATS, required=True)
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "models/packaging-v1.json")
    parser.add_argument("--parent-dir", type=Path, default=PROJECT_ROOT / "models/gguf/v1-ckpt19000-bf16")
    parser.add_argument("--parent-evaluation", type=Path, default=PROJECT_ROOT / "evaluation/results/v1-gguf-bf16-smoke-valid")
    parser.add_argument("--toolchain", type=Path, default=PROJECT_ROOT / "models/gguf/toolchain-v1/manifest.json")
    parser.add_argument("--imatrix-manifest", type=Path, default=PROJECT_ROOT / "models/gguf/v1-ckpt19000-imatrix/manifest.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    manifest_path, evidence = None, {}
    try:
        plan = json.loads(args.plan.read_text())
        if (plan["approval"]["status"] != "approved" or plan["approval"].get("gate") != "H-045"
                or args.format not in plan["quantization"]["formats"]):
            raise ArtifactError("quantization format is not approved")
        raw_output = args.output_dir or PROJECT_ROOT / f"models/gguf/v1-ckpt19000-{args.format.lower()}"
        output = raw_output.resolve()
        allowed = (PROJECT_ROOT / "models/gguf").resolve()
        if output == allowed or not output.is_relative_to(allowed) or raw_output.is_symlink() or output.exists():
            raise ArtifactError("quantization output must be a new local models/gguf directory")
        print("Checking BF16 parent, native parity, and disk reserve...", flush=True)
        checkout = validate_toolchain(PROJECT_ROOT, plan, args.toolchain)
        parent, parent_evidence = check_parent(PROJECT_ROOT, plan, args.parent_dir, args.parent_evaluation,
                                               args.toolchain, sha256_file(args.plan))
        source = PROJECT_ROOT / plan["fusion"]["output"]
        inspect_floating_weights(source)
        needed = minimum_free_bytes(parent.stat().st_size, args.format)
        available = shutil.disk_usage(PROJECT_ROOT).free
        if available < needed:
            raise ArtifactError(f"insufficient disk: need {needed / 1024**3:.2f} GiB including a 4 GiB reserve; available {available / 1024**3:.2f} GiB")
        target = output / f"model-{args.format.lower()}.gguf"
        command = [str(checkout / "build/bin/llama-quantize")]
        imatrix_evidence = None
        if args.format.endswith("-imatrix"):
            imatrix_evidence = json.loads(args.imatrix_manifest.read_text())
            matrix = Path(imatrix_evidence["model"])
            if (imatrix_evidence.get("status") != "completed"
                    or imatrix_evidence["plan_sha256"] != sha256_file(args.plan)
                    or imatrix_evidence["parent_sha256"] != parent_evidence["model_sha256"]
                    or imatrix_evidence["calibration_sha256"] != plan["quantization"]["calibration_sha256"]
                    or imatrix_evidence["toolchain_sha256"] != sha256_file(args.toolchain)
                    or sha256_file(matrix) != imatrix_evidence["model_sha256"]):
                raise ArtifactError("importance matrix does not match the approved parent and calibration")
            matrix_metadata_path = args.imatrix_manifest.parent / "metadata.json"
            matrix_metadata = json.loads(matrix_metadata_path.read_text())
            if (sha256_file(matrix_metadata_path) != imatrix_evidence["metadata_sha256"]
                    or matrix_metadata.get("status") != "passed"
                    or matrix_metadata.get("model_sha256") != imatrix_evidence["model_sha256"]
                    or matrix_metadata.get("complete_counts") is not True
                    or matrix_metadata.get("covered_layer_weights") != 196
                    or matrix_metadata.get("implementation_sha256") != sha256_file(PROJECT_ROOT / "scripts/model/inspect_imatrix.py")):
                raise ArtifactError("importance-matrix coverage evidence is missing or changed")
            command += ["--imatrix", str(matrix)]
        command += [str(parent), str(target), args.format.removesuffix("-imatrix"), "4"]
        output.mkdir(parents=True)
        manifest_path = output / "manifest.json"
        evidence = {"schema_version": 1, "status": "running", "format": args.format,
                    "started_at_utc": datetime.now(UTC).isoformat(), "plan_sha256": sha256_file(args.plan),
                    "parent": parent_evidence, "toolchain_sha256": sha256_file(args.toolchain), "command": command,
                    "imatrix_manifest_sha256": sha256_file(args.imatrix_manifest) if imatrix_evidence else None,
                    "model": str(target), "minimum_free_bytes": needed, "free_bytes_before": available,
                    "implementation_sha256": sha256_file(Path(__file__)),
                    "inspector_sha256": sha256_file(PROJECT_ROOT / "scripts/model/inspect_gguf.py")}
        manifest_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        started = time.monotonic()
        native_env = {key: value for key, value in os.environ.items()
                      if not key.startswith(("LLAMA_", "GGML_", "HF_", "HUGGING_FACE_"))}
        with (output / "quantization.log").open("x") as log:
            result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=native_env)
            evidence.update(return_code=result.returncode, quantization_seconds=round(time.monotonic()-started, 3))
            if result.returncode:
                raise ArtifactError("native quantizer failed; inspect the local quantization log")
            result = subprocess.run([str(PROJECT_ROOT / ".venv-conversion/bin/python"), "-X",
                "pycache_prefix=/private/tmp/local-pii-redactor-conversion-pycache",
                str(PROJECT_ROOT / "scripts/model/inspect_gguf.py"), "--model", str(target), "--source", str(source),
                "--output", str(output / "metadata.json")], cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
            if result.returncode:
                raise ArtifactError("quantized metadata verification failed; inspect the local log")
        if sha256_file(parent) != parent_evidence["model_sha256"]:
            raise ArtifactError("BF16 parent changed during quantization")
        evidence.update(status="completed", model_sha256=sha256_file(target), model_bytes=target.stat().st_size,
                        completed_at_utc=datetime.now(UTC).isoformat(), log_sha256=sha256_file(output / "quantization.log"),
                        metadata_sha256=sha256_file(output / "metadata.json"), free_bytes_after=shutil.disk_usage(PROJECT_ROOT).free)
        manifest_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    except (ArtifactError, EvaluationError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        if manifest_path:
            evidence.update(status="failed", failure_type=type(exc).__name__)
            if isinstance(exc, ArtifactError): evidence["failure_reason"] = str(exc)
            manifest_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(f"Quantization stopped ({type(exc).__name__}).", file=sys.stderr)
        if isinstance(exc, ArtifactError): print(str(exc), file=sys.stderr)
        return 1
    print(f"Quantized model: {target}")
    print(f"Manifest: {manifest_path}")
    print("Generation evaluation against the BF16 parent remains required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
