#!/usr/bin/env python3
"""Verify a completed fusion; explicitly preserve/recover its approved tokenizer."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.model_artifacts import (
    ArtifactError, check_tokenizer_parity, inspect_floating_weights, inventory,
    preserve_approved_tokenizer, sha256_file, validate_approved_inputs,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "models/packaging-v1.json")
    parser.add_argument("--restore-approved-tokenizer", action="store_true")
    args = parser.parse_args(argv)
    evidence_path = None
    evidence = {}
    try:
        plan = json.loads(args.plan.read_text())
        print("Verifying approved inputs and saved fusion...", flush=True)
        base, adapter = validate_approved_inputs(PROJECT_ROOT, plan)
        output = (PROJECT_ROOT / plan["fusion"]["output"]).resolve()
        run = (PROJECT_ROOT / plan["fusion"]["run_dir"]).resolve()
        allowed = (PROJECT_ROOT / "models/fused").resolve()
        if any(p == allowed or not p.is_relative_to(allowed) for p in (output, run)):
            raise ArtifactError("fusion paths must remain under local models/fused")
        original_path = run / "manifest.json"
        original = json.loads(original_path.read_text())
        if original.get("return_code") != 0 or original.get("plan_sha256") != sha256_file(args.plan):
            raise ArtifactError("fusion did not complete for this exact plan")
        if Path(original["output"]).resolve() != output:
            raise ArtifactError("fusion artifact path does not match the manifest")
        if original["inputs"] != {"base": inventory(base), "adapter": inventory(adapter)}:
            raise ArtifactError("fusion inputs differ from approved runtime files")
        dataset = PROJECT_ROOT / plan["validation"]["dataset"]
        if sha256_file(dataset) != plan["validation"]["sha256"]:
            raise ArtifactError("validation hash changed")
        weights = inspect_floating_weights(output)
        weight_hashes = {p.name: sha256_file(p) for p in output.glob("*.safetensors")}
        destination = run / "verified-fusion.json"
        if destination.exists():
            raise ArtifactError("verification evidence already exists")
        evidence = {"schema_version": 1, "status": "running", "weights": weights,
                    "original_manifest_sha256": sha256_file(original_path),
                    "plan_sha256": sha256_file(args.plan), "output": str(output),
                    "started_at_utc": datetime.now(UTC).isoformat(),
                    "implementation_sha256": {
                        "scripts/model/verify_fusion.py": sha256_file(Path(__file__)),
                        "src/pii_redactor/model_artifacts.py": sha256_file(PROJECT_ROOT / "src/pii_redactor/model_artifacts.py"),
                    }}
        evidence_path = destination
        if args.restore_approved_tokenizer:
            print("Preserving generated tokenizer and restoring approved tokenizer bytes...", flush=True)
            evidence["tokenizer_preservation"] = preserve_approved_tokenizer(base, output, run / "generated-tokenizer")
        evidence["tokenizer_parity"] = check_tokenizer_parity(base, output, dataset)
        if weight_hashes != {p.name: sha256_file(p) for p in output.glob("*.safetensors")}:
            raise ArtifactError("fusion weights changed during tokenizer verification")
        if original["inputs"] != {"base": inventory(base), "adapter": inventory(adapter)}:
            raise ArtifactError("approved inputs changed during tokenizer verification")
        evidence.update(status="completed", artifacts=inventory(output),
                        completed_at_utc=datetime.now(UTC).isoformat())
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    except (ArtifactError, OSError, ValueError, KeyError, TypeError) as exc:
        if evidence_path:
            evidence.update(status="failed", failure_type=type(exc).__name__)
            if isinstance(exc, ArtifactError):
                evidence["failure_reason"] = str(exc)
            evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(f"Fusion verification failed ({type(exc).__name__}). Original evidence is preserved.", file=sys.stderr)
        return 1
    print(f"Verified fusion: {evidence_path}")
    print("Generation parity is still required before conversion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
