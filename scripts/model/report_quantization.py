#!/usr/bin/env python3
"""Recompute all approved validation gates and export a text-free packaging report."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.evaluation.metrics import EvaluationError
from pii_redactor.model_artifacts import ArtifactError, sha256_file
from scripts.evaluation.check_parity import compare
from scripts.model.quantize_gguf import check_parent, validate_toolchain


def aggregate(pair, manifest, artifact, predictions):
    metrics = pair["candidate_metrics"]
    seconds = prompt_ms = predicted_ms = prompt_n = predicted_n = 0.0
    timing_anomalies = 0
    for line in predictions.open(encoding="utf-8"):
        row = json.loads(line)
        seconds += row["elapsed_seconds"]
        timings = row["timings"]
        observed = [row["elapsed_seconds"], timings["prompt_ms"], timings["predicted_ms"],
                    timings["prompt_n"], timings["predicted_n"]]
        valid_numbers = all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in observed)
        if not valid_numbers:
            raise ArtifactError("native evaluation timing contains invalid numeric values")
        # Both clocks should cover the same request. Suspend/clock discontinuity
        # can invalidate rates without changing the saved extraction outputs.
        timing_anomalies += (timings["prompt_ms"] + timings["predicted_ms"]) / 1000 > row["elapsed_seconds"] + 1
        prompt_ms += timings["prompt_ms"]
        prompt_n += timings["prompt_n"]
        predicted_ms += timings["predicted_ms"]
        # Native timing excludes the first sampled token from the timed decode.
        predicted_n += max(0, timings["predicted_n"] - 1)
    return {"status": pair["status"], "model_sha256": manifest["model_sha256"],
            "model_bytes": artifact["model_bytes"], "predictions_sha256": manifest["predictions_sha256"],
            "micro": metrics["micro"], "documents": metrics["documents"], "validity": metrics["validity"],
            "per_class": metrics["per_class"], "per_class_deltas": pair["per_class_deltas"],
            "gates": pair["gates"],
            "changed_outputs": pair["output_changes"]["raw_changed_records"],
            "changed_entity_sequences": pair["output_changes"]["entity_sequence_changed_records"],
            "local_mac_diagnostics": {"request_seconds": round(seconds, 6),
                "timing_status": "inconsistent_native_and_client_clocks" if timing_anomalies else "diagnostic_only",
                "timing_anomaly_records": timing_anomalies,
                "prompt_tokens_per_second": round(prompt_n / (prompt_ms / 1000), 6) if prompt_ms and not timing_anomalies else None,
                "decode_tokens_per_second": round(predicted_n / (predicted_ms / 1000), 6) if predicted_ms and not timing_anomalies else None,
                "peak_ram_bytes": None}}


def build_report(root, plan_path):
    plan = json.loads(plan_path.read_text())
    if plan["approval"]["status"] != "approved" or plan["approval"]["gate"] != "H-045":
        raise ArtifactError("packaging trial is not approved")
    toolchain = root / "models/gguf/toolchain-v1/manifest.json"
    validate_toolchain(root, plan, toolchain)
    parent_dir = root / "models/gguf/v1-ckpt19000-bf16"
    parent_eval = root / "evaluation/results/v1-gguf-bf16-smoke-valid"
    parent, parent_evidence = check_parent(root, plan, parent_dir, parent_eval, toolchain, sha256_file(plan_path))
    reference = parent_eval / "predictions.jsonl"
    parent_run = json.loads((parent_eval / "manifest.json").read_text())
    tokenizer = json.loads((parent_eval / "tokenizer-parity.json").read_text())
    dataset = root / plan["validation"]["dataset"]
    dataset_hash = plan["validation"]["sha256"]
    results = {}
    for format in ["BF16", *plan["quantization"]["formats"]]:
        slug = format.lower()
        model_dir = root / f"models/gguf/v1-ckpt19000-{slug}"
        evaluation = root / f"evaluation/results/v1-gguf-{slug}-smoke-valid"
        manifest = json.loads((evaluation / "manifest.json").read_text())
        predictions = evaluation / "predictions.jsonl"
        artifact_path = model_dir / ("verified-conversion.json" if format == "BF16" else "manifest.json")
        artifact = json.loads(artifact_path.read_text())
        model = parent if format == "BF16" else model_dir / f"model-{slug}.gguf"
        compare_keys = ["dataset_sha256", "toolchain_sha256", "revision", "seed", "temperature",
                        "enable_thinking", "max_tokens", "context", "threads", "batch_size", "ubatch_size",
                        "gpu_layers", "flash_attention", "kv_dtype", "cache_prompt", "tokenizer_files_sha256",
                        "tokenizers_version", "transformers_version", "implementation_sha256"]
        if (manifest["status"] != "completed" or artifact["status"] != "completed"
                or artifact["plan_sha256"] != sha256_file(plan_path)
                or any(manifest.get(key) != parent_run.get(key) for key in compare_keys)
                or sha256_file(predictions) != manifest["predictions_sha256"]
                or sha256_file(model) != manifest["model_sha256"]
                or artifact["model_sha256"] != manifest["model_sha256"]
                or model.stat().st_size != artifact["model_bytes"]
                or json.loads((evaluation / "tokenizer-parity.json").read_text()) != tokenizer):
            raise ArtifactError("candidate provenance differs from the completed frozen experiment")
        if format != "BF16" and artifact["parent"]["model_sha256"] != parent_evidence["model_sha256"]:
            raise ArtifactError("candidate was not quantized from the approved BF16 parent")
        pair = compare(dataset, reference, predictions, dataset_hash,
            recall_tolerance=0 if format == "BF16" else plan["quantization"]["maximum_absolute_recall_drop"],
            document_tolerance=0 if format == "BF16" else plan["quantization"]["maximum_absolute_complete_document_recall_drop"])
        results[format] = aggregate(pair, manifest, artifact, predictions)
        results[format]["artifact_manifest_sha256"] = sha256_file(artifact_path)
    matrix_dir = root / "models/gguf/v1-ckpt19000-imatrix"
    matrix_manifest_path = matrix_dir / "manifest.json"
    matrix_manifest = json.loads(matrix_manifest_path.read_text())
    matrix_metadata = json.loads((matrix_dir / "metadata.json").read_text())
    imatrix_quant = json.loads((root / "models/gguf/v1-ckpt19000-q4_k_m-imatrix/manifest.json").read_text())
    if (matrix_manifest["status"] != "completed" or matrix_metadata["status"] != "passed"
            or matrix_manifest["metadata_sha256"] != sha256_file(matrix_dir / "metadata.json")
            or matrix_manifest["model_sha256"] != sha256_file(matrix_dir / "importance.gguf")
            or matrix_metadata["model_sha256"] != matrix_manifest["model_sha256"]
            or matrix_manifest["calibration_sha256"] != plan["quantization"]["calibration_sha256"]
            or matrix_manifest["parent_sha256"] != parent_evidence["model_sha256"]
            or imatrix_quant["imatrix_manifest_sha256"] != sha256_file(matrix_manifest_path)):
        raise ArtifactError("importance-matrix provenance changed")
    calibration = {key: matrix_metadata[key] for key in ("model_sha256", "model_bytes", "chunk_count",
        "chunk_size", "processed_tokens", "tensor_count", "covered_layer_weights", "complete_counts",
        "finite_nonnegative", "output_weight_calibrated", "limitations")}
    calibration.update(calibration_sha256=matrix_manifest["calibration_sha256"],
        split_overlap=matrix_manifest["calibration_audit"]["split_overlap"],
        generation_seconds=matrix_manifest["generation_seconds"],
        manifest_sha256=sha256_file(matrix_manifest_path),
        actual_inspector_sha256=matrix_metadata["implementation_sha256"])
    passed = [fmt for fmt in plan["quantization"]["formats"] if results[fmt]["status"] == "passed"]
    return {"schema_version": 1, "approval_gate": "H-045", "checkpoint": 19000,
            "status": "validation_candidates_available" if passed else "review_required_no_quantization_passed",
            "plan_sha256": sha256_file(plan_path), "dataset_sha256": dataset_hash,
            "records": parent_run["records"], "prompt_tokens_sha256": tokenizer["prompt_tokens_sha256"],
            "toolchain_sha256": sha256_file(toolchain), "llama_cpp_revision": plan["llama_cpp"]["revision"],
            "generation": plan["generation"], "passing_formats": passed, "selected_format": None,
            "results": results, "importance_matrix": calibration,
            "limitations": ["Frozen 100-record validation comparison, not the full validation or test set.",
                "No checkpoint, dataset, decoding, evaluator, or tolerance changes were made.",
                "Local Mac timing is diagnostic, not a controlled speed comparison or Pi benchmark; peak native RAM was not measured.",
                "Native rates are suppressed when native phase durations exceed client request time by more than one second; request_seconds is only the client-clock sum.",
                "No deployment, Pi transfer, or failed-candidate promotion is authorized by this report.",
                "Per-class estimates are noisy on this small set; no inference of universal equivalence is justified."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "models/packaging-v1.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evaluation/baselines/quantization-v1.json")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink() or args.output.parent.resolve() != (PROJECT_ROOT / "evaluation/baselines").resolve():
            raise ArtifactError("aggregate report requires a new file in evaluation/baselines")
        report = build_report(PROJECT_ROOT, args.plan)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (ArtifactError, EvaluationError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f"Quantization report stopped ({type(exc).__name__}).", file=sys.stderr)
        if isinstance(exc, ArtifactError): print(str(exc), file=sys.stderr)
        return 1
    print(f"Quantization report: {args.output}")
    print(f"Passing formats: {', '.join(report['passing_formats']) or 'none'}")
    return 0 if report["passing_formats"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
