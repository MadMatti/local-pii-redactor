#!/usr/bin/env python3
"""Convert a verified, parity-gated fusion with the pinned official BF16 converter."""

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

from pii_redactor.model_artifacts import ArtifactError, evaluation_fingerprint, inventory, sha256_file, validate_approved_inputs
from pii_redactor.evaluation.metrics import EvaluationError
from scripts.evaluation.check_parity import compare


def verify_generation(root, plan, source, candidate_dir):
    reference = root / plan["validation"]["reference_predictions"]
    candidate = candidate_dir / "predictions.jsonl"
    for predictions, expected_model, expected_adapter in (
        (reference, plan["base"]["evaluation_fingerprint"], evaluation_fingerprint(root / plan["adapter"]["path"])),
        (candidate, evaluation_fingerprint(source), None),
    ):
        manifest = json.loads((predictions.parent / "manifest.json").read_text())
        expected = {"model_fingerprint": expected_model, "adapter_fingerprint": expected_adapter,
                    "dataset_sha256": plan["validation"]["sha256"], **plan["generation"]}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ArtifactError("generation evidence differs from approved inputs or decoding")
        if manifest["predictions_sha256"] != sha256_file(predictions):
            raise ArtifactError("generation predictions changed after evaluation")
    result = compare(root / plan["validation"]["dataset"], reference, candidate, plan["validation"]["sha256"])
    if result["status"] != "passed":
        raise ArtifactError("fusion generation parity must pass before conversion")
    return {"reference_predictions_sha256": sha256_file(reference),
            "fused_predictions_sha256": sha256_file(candidate), "gates": result["gates"],
            "output_changes": result["output_changes"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "models/packaging-v1.json")
    parser.add_argument("--fused-evaluation", type=Path, default=PROJECT_ROOT / "evaluation/results/v1-fused-bf16-smoke-valid")
    parser.add_argument("--toolchain", type=Path, default=PROJECT_ROOT / "models/gguf/toolchain-v1/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models/gguf/v1-ckpt19000-bf16")
    parser.add_argument("--verify-existing", action="store_true",
                        help="Verify a completed native conversion without rerunning or overwriting its evidence")
    args = parser.parse_args(argv)
    manifest_path, evidence = None, {}
    try:
        plan = json.loads(args.plan.read_text())
        output = args.output_dir.resolve()
        allowed = (PROJECT_ROOT / "models/gguf").resolve()
        if (output == allowed or not output.is_relative_to(allowed) or args.output_dir.is_symlink()
                or (output.exists() and not args.verify_existing)):
            raise ArtifactError("conversion output must be a new local models/gguf directory")
        print("Verifying approved fusion, generation parity, and pinned converter...", flush=True)
        validate_approved_inputs(PROJECT_ROOT, plan)
        source = PROJECT_ROOT / plan["fusion"]["output"]
        run = PROJECT_ROOT / plan["fusion"]["run_dir"]
        verification = run / "verified-fusion.json"
        if not verification.exists():
            verification = run / "manifest.json"
        verified = json.loads(verification.read_text())
        before = inventory(source)
        if verified.get("status") != "completed" or verified.get("plan_sha256") != sha256_file(args.plan) or verified.get("artifacts") != before:
            raise ArtifactError("fusion does not match successful immutable verification evidence")
        gate = verify_generation(PROJECT_ROOT, plan, source, args.fused_evaluation)
        toolchain = json.loads(args.toolchain.read_text())
        checkout = PROJECT_ROOT / plan["llama_cpp"]["path"]
        if toolchain.get("status") != "recorded" or toolchain.get("revision") != plan["llama_cpp"]["revision"]:
            raise ArtifactError("pinned toolchain evidence is required")
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=checkout, text=True)
        if revision != toolchain["revision"] or dirty:
            raise ArtifactError("converter checkout changed after recording")
        converter = checkout / "convert_hf_to_gguf.py"
        if sha256_file(converter) != toolchain["converter_script_sha256"]:
            raise ArtifactError("converter checksum changed")
        source_bytes = sum(p.stat().st_size for p in source.glob("*.safetensors"))
        if not args.verify_existing and shutil.disk_usage(PROJECT_ROOT).free < source_bytes + 4 * 1024**3:
            raise ArtifactError("conversion requires estimated output space plus a 4 GiB free reserve")
        python = PROJECT_ROOT / ".venv-conversion/bin/python"
        python_command = [str(python), "-X", "pycache_prefix=/private/tmp/local-pii-redactor-conversion-pycache"]
        packages = json.loads(subprocess.check_output([*python_command, "-m", "pip", "list", "--format=json"], text=True))
        if sorted(packages, key=lambda item: item["name"].lower()) != toolchain["converter_packages"]:
            raise ArtifactError("converter environment differs from recorded dependencies")
        model = output / "model-bf16.gguf"
        command = [*python_command, str(converter), str(source), "--outtype", "bf16", "--outfile", str(model)]
        original, model_before = None, None
        destination = output / ("verified-conversion.json" if args.verify_existing else "manifest.json")
        if args.verify_existing:
            original = json.loads((output / "manifest.json").read_text())
            if (original.get("return_code") != 0 or original.get("command") != command
                    or original.get("source_artifacts") != before
                    or original.get("plan_sha256") != sha256_file(args.plan)
                    or original.get("toolchain_sha256") != sha256_file(args.toolchain)):
                raise ArtifactError("existing native conversion does not match current verified inputs")
            if not model.is_file() or model.is_symlink() or destination.exists() or (output / "metadata.json").exists():
                raise ArtifactError("existing model is missing or verification evidence already exists")
            model_before = sha256_file(model)
        else:
            output.mkdir(parents=True)
        manifest_path = destination
        evidence = {"schema_version": 1, "status": "running", "started_at_utc": datetime.now(UTC).isoformat(),
                    "plan_sha256": sha256_file(args.plan), "fusion_verification_sha256": sha256_file(verification),
                    "toolchain_sha256": sha256_file(args.toolchain), "source_artifacts": before,
                    "generation_gate": gate, "command": command, "model": str(model),
                    "implementation_sha256": sha256_file(Path(__file__)),
                    "inspector_sha256": sha256_file(PROJECT_ROOT / "scripts/model/inspect_gguf.py")}
        if original is not None:
            evidence.update(original_manifest_sha256=sha256_file(output / "manifest.json"),
                            return_code=original["return_code"], model_sha256_before_verification=model_before)
        manifest_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
        started = time.monotonic()
        log_path = output / ("verification.log" if args.verify_existing else "conversion.log")
        print("Verifying existing BF16 GGUF..." if args.verify_existing else "Converting verified fused weights to BF16 GGUF...", flush=True)
        with log_path.open("x") as log:
            if not args.verify_existing:
                result = subprocess.run(command, env=env, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
                evidence.update(return_code=result.returncode, conversion_seconds=round(time.monotonic()-started, 3))
                if result.returncode:
                    raise ArtifactError("official converter failed; inspect the local conversion log")
            result = subprocess.run([*python_command, str(PROJECT_ROOT / "scripts/model/inspect_gguf.py"),
                                     "--model", str(model), "--source", str(source), "--floating",
                                     "--output", str(output / "metadata.json")], env=env, cwd=PROJECT_ROOT,
                                    stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise ArtifactError("GGUF metadata verification failed; inspect the local conversion log")
        if inventory(source) != before:
            raise ArtifactError("fused source changed during conversion")
        model_after = sha256_file(model)
        if model_before is not None and model_after != model_before:
            raise ArtifactError("existing GGUF changed during verification")
        evidence.update(status="completed", model_sha256=model_after, model_bytes=model.stat().st_size,
                        metadata_sha256=sha256_file(output / "metadata.json"), log_sha256=sha256_file(log_path),
                        completed_at_utc=datetime.now(UTC).isoformat())
        manifest_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    except (ArtifactError, EvaluationError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        if manifest_path:
            evidence.update(status="failed", failure_type=type(exc).__name__)
            if isinstance(exc, ArtifactError):
                evidence["failure_reason"] = str(exc)
            manifest_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(f"GGUF conversion stopped ({type(exc).__name__}).", file=sys.stderr)
        if isinstance(exc, ArtifactError):
            print(str(exc), file=sys.stderr)
        return 1
    print(f"BF16 GGUF: {model}")
    print(f"Manifest: {manifest_path}")
    print("Metadata checks passed; GGUF generation parity remains required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
