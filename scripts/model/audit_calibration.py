#!/usr/bin/env python3
"""Audit existing calibration provenance and held-out overlap without changing data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.data.synthetic import generate_calibration_texts
from pii_redactor.model_artifacts import ArtifactError, sha256_file

SEPARATOR = "\n\n<|calibration_document|>\n\n"


def digest_text(text, *, normalized=False):
    if normalized:
        text = " ".join(text.lower().split())
    return hashlib.sha256(text.encode()).hexdigest()


def audit(root, plan):
    path = root / plan["quantization"]["calibration_path"]
    manifest = json.loads((path.parent / "manifest.json").read_text())
    expected_hash = plan["quantization"]["calibration_sha256"]
    if sha256_file(path) != expected_hash or manifest["artifact"]["sha256"] != expected_hash:
        raise ArtifactError("calibration checksum differs from approval")
    if manifest["seed"] != 42 or manifest["records"] != 500 or manifest["uses_frozen_test_data"] is not False:
        raise ArtifactError("calibration provenance differs from the approved corpus")
    regenerated = list(generate_calibration_texts(count=500, seed=42))
    if digest_text(SEPARATOR.join(regenerated) + "\n") != expected_hash:
        raise ArtifactError("current train-only generator cannot reproduce approved calibration")
    hashes = {digest_text(text) for text in regenerated}
    normalized_hashes = {digest_text(text, normalized=True) for text in regenerated}
    build = json.loads((root / "data/processed/manifest.json").read_text())
    overlaps, datasets = {}, {}
    for split in ("train", "valid", "test"):
        relative = f"data/processed/{split}.jsonl"
        dataset = root / relative
        expected = build["artifacts"][relative]["sha256"]
        if sha256_file(dataset) != expected:
            raise ArtifactError("prepared split changed after the frozen dataset build")
        if split == "test":
            frozen = json.loads((root / "evaluation/results/v1-selected-ckpt19000-full-test/manifest.json").read_text())
            if frozen["dataset_sha256"] != expected:
                raise ArtifactError("test split differs from the selected adapter's frozen evaluation")
        exact = normalized = records = 0
        for line in dataset.open(encoding="utf-8"):
            row = json.loads(line)
            text = row["messages"][1]["content"]
            records += 1
            exact += digest_text(text) in hashes
            normalized += digest_text(text, normalized=True) in normalized_hashes
        datasets[split] = {"records": records, "sha256": expected}
        overlaps[split] = {"exact_records": exact, "normalized_records": normalized}
    held_out_overlap = any(overlaps[s][key] for s in ("valid", "test") for key in ("exact_records", "normalized_records"))
    return {"schema_version": 1, "status": "review_required" if held_out_overlap else "passed",
            "calibration_sha256": expected_hash, "records": 500, "positive_generator_records": 250,
            "negative_generator_records": 250, "seed": 42, "source_split": "train",
            "unique_exact_documents": len(hashes), "unique_normalized_documents": len(normalized_hashes),
            "regenerated_bytes_equal": True, "split_overlap": overlaps, "datasets": datasets,
            "generator_sha256": sha256_file(root / "src/pii_redactor/data/synthetic.py"),
            "limitations": ["Synthetic train-style calibration; training overlap is allowed and reported.",
                            "Exact/normalized checks do not rule out shared phrases or entity values.",
                            "This audits the approved corpus; it does not establish optimal calibration representativeness."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "models/packaging-v1.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models/gguf/calibration-audit-v1/manifest.json")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink() or not args.output.resolve().is_relative_to((PROJECT_ROOT / "models/gguf").resolve()):
            raise ArtifactError("calibration audit requires a new local models/gguf report")
        result = audit(PROJECT_ROOT, json.loads(args.plan.read_text()))
        result.update(plan_sha256=sha256_file(args.plan), implementation_sha256=sha256_file(Path(__file__)))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (ArtifactError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Calibration audit failed ({type(exc).__name__}).", file=sys.stderr)
        return 1
    print(f"Calibration audit: {result['status']}")
    print(f"Report: {args.output}")
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
