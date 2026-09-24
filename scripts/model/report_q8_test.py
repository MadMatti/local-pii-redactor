#!/usr/bin/env python3
"""Verify and summarize the approved Q8 full test without selecting on test data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.evaluation.metrics import EvaluationError
from pii_redactor.evaluation.parsing import parse_prediction
from pii_redactor.model_artifacts import ArtifactError, sha256_file
from scripts.evaluation.check_parity import compare
from scripts.model.quantize_gguf import validate_toolchain
from scripts.model.report_quantization import aggregate


def generation_diagnostics(predictions, *, context, max_tokens, eos_token_id):
    stops, invalid_reasons, invalid_stops = Counter(), Counter(), Counter()
    records = maximum_prompt = maximum_sampled = 0
    for line in predictions.open(encoding="utf-8"):
        row = json.loads(line)
        tokens = row["generated_token_ids"]
        stop = row["stop_type"]
        if (stop not in {"eos", "limit"} or not tokens or any(type(t) is not int for t in tokens)
                or len(tokens) != row["sampled_tokens"] or len(tokens) > max_tokens
                or row["prompt_tokens"] + max_tokens > context
                or (stop == "eos" and tokens[-1] != eos_token_id)
                or (stop == "limit" and len(tokens) != max_tokens)):
            raise ArtifactError("saved generation violates frozen context, cap, or EOS behavior")
        records += 1
        maximum_prompt = max(maximum_prompt, row["prompt_tokens"])
        maximum_sampled = max(maximum_sampled, row["sampled_tokens"])
        stops[stop] += 1
        parsed = parse_prediction(row["raw_output"])
        if not parsed.schema_valid:
            invalid_reasons[parsed.error] += 1
            invalid_stops[stop] += 1
    return {"records": records, "stop_types": dict(sorted(stops.items())),
            "schema_invalid_records": sum(invalid_stops.values()),
            "schema_invalid_reasons": dict(sorted(invalid_reasons.items())),
            "schema_invalid_stop_types": dict(sorted(invalid_stops.items())),
            "maximum_prompt_tokens": maximum_prompt, "maximum_sampled_tokens": maximum_sampled,
            "saved_eos_and_context_checks_passed": True}


def check_evidence(root, protocol_path):
    protocol = json.loads(protocol_path.read_text())
    if (protocol["approval"]["status"] != "approved"
            or protocol["selection_basis"]["dataset_partition"] != "validation"
            or protocol["selection_basis"]["passing_format"] != "Q8_0"):
        raise ArtifactError("Q8 full-test approval and validation-only selection are required")
    selection_path = root / protocol["selection_basis"]["report"]
    selection = json.loads(selection_path.read_text())
    if (sha256_file(selection_path) != protocol["selection_basis"]["report_sha256"]
            or selection["passing_formats"] != ["Q8_0"]
            or selection["results"]["Q8_0"]["model_sha256"] != protocol["model"]["sha256"]):
        raise ArtifactError("validation-selected artifact changed")
    packaging = json.loads((root / "models/packaging-v1.json").read_text())
    toolchain = root / "models/gguf/toolchain-v1/manifest.json"
    validate_toolchain(root, packaging, toolchain)
    model = root / protocol["model"]["path"]
    if (model.is_symlink() or sha256_file(model) != protocol["model"]["sha256"]
            or model.stat().st_size != protocol["model"]["bytes"]):
        raise ArtifactError("approved Q8 model changed")
    output = root / protocol["output_dir"]
    run = json.loads((output / "manifest.json").read_text())
    validation = json.loads((root / "evaluation/results/v1-gguf-q8_0-smoke-valid/manifest.json").read_text())
    expected = {"model_sha256": protocol["model"]["sha256"], "dataset_sha256": protocol["test"]["sha256"],
                "records": protocol["test"]["records"], "revision": protocol["implementation"]["llama_cpp_revision"],
                "toolchain_sha256": sha256_file(toolchain), **protocol["generation"]}
    implementation = {"runner": protocol["implementation"]["runner_sha256"],
                      "transport": protocol["implementation"]["transport_sha256"]}
    if (run["status"] != "completed" or any(run.get(key) != value for key, value in expected.items())
            or run["implementation_sha256"] != implementation
            or sha256_file(root / "scripts/evaluation/run_gguf_baseline.py") != implementation["runner"]
            or sha256_file(root / "src/pii_redactor/evaluation/llama_local.py") != implementation["transport"]
            or any(run[key] != validation[key] for key in ("tokenizer_files_sha256", "tokenizers_version",
                                                          "transformers_version", "implementation_sha256"))):
        raise ArtifactError("full-test generation differs from approved frozen configuration")
    predictions = output / "predictions.jsonl"
    reference = root / protocol["comparison"]["reference"]
    ref_run = json.loads((reference.parent / "manifest.json").read_text())
    if (sha256_file(predictions) != run["predictions_sha256"]
            or sha256_file(reference) != protocol["comparison"]["reference_sha256"]
            or ref_run["predictions_sha256"] != protocol["comparison"]["reference_sha256"]
            or ref_run["dataset_sha256"] != protocol["test"]["sha256"]
            or any(ref_run[key] != protocol["generation"][key] for key in ("max_tokens", "seed", "temperature", "enable_thinking"))):
        raise ArtifactError("completed full-test prediction or reference evidence changed")
    prompt_check = json.loads((output / "tokenizer-parity.json").read_text())
    runtime = json.loads((output / "last-runtime-check.json").read_text())
    if (prompt_check["records"] != protocol["test"]["records"] or prompt_check["template_equal"] is not True
            or prompt_check["prompt_ids_equal"] is not True or runtime["context"] != protocol["generation"]["context"]
            or Path(runtime["model_path"]).resolve() != model.resolve()
            or runtime["eos_token_id"] != 151645 or runtime["slots"] != 1
            or runtime["ui_enabled"] is not False or runtime["cors_proxy_enabled"] is not False):
        raise ArtifactError("native prompt, EOS, or private-runtime evidence differs")
    return protocol, run, predictions, reference, prompt_check


def build_report(root, protocol_path):
    protocol, run, predictions, reference, prompt_check = check_evidence(root, protocol_path)
    pair = compare(root / protocol["test"]["path"], reference, predictions, protocol["test"]["sha256"])
    metrics = pair["candidate_metrics"]
    if (metrics["dataset"]["records"] != protocol["test"]["records"]
            or metrics["documents"]["pii_records"] != protocol["test"]["positive_records"]
            or metrics["documents"]["pii_free_records"] != protocol["test"]["negative_records"]
            or metrics["micro"]["tp"] + metrics["micro"]["fn"] != protocol["test"]["gold_entities"]):
        raise ArtifactError("scored test composition differs from the frozen protocol")
    # Reuse timing checks, but remove threshold/selection semantics: this reference
    # is the original adapter, not a full-test BF16 quantization parent.
    candidate = aggregate(pair, run, {"model_bytes": protocol["model"]["bytes"]}, predictions)
    candidate.pop("gates")
    candidate.pop("status")
    reference_metrics = {key: pair["reference_metrics"][key] for key in ("micro", "documents", "validity", "per_class")}
    deltas = {section: {key: round(value - reference_metrics[section][key], 8)
                       for key, value in metrics[section].items()}
              for section in ("micro", "documents", "validity")}
    return {"schema_version": 1, "status": "completed_descriptive_full_test",
            "protocol_sha256": sha256_file(protocol_path), "test": protocol["test"],
            "selection_basis": "validation_only_and_explicit_user_approval_before_test",
            "model_format": "Q8_0", "checkpoint": protocol["model"]["checkpoint"],
            "deployment_approved": False, "generation": protocol["generation"],
            "prompt_tokens_sha256": prompt_check["prompt_tokens_sha256"],
            "run_manifest_sha256": sha256_file(predictions.parent / "manifest.json"),
            "reference_predictions_sha256": protocol["comparison"]["reference_sha256"],
            "candidate": candidate, "selected_adapter_reference": reference_metrics,
            "deltas_from_selected_adapter": deltas,
            "generation_diagnostics": generation_diagnostics(predictions, context=protocol["generation"]["context"],
                max_tokens=protocol["generation"]["max_tokens"], eos_token_id=151645),
            "implementation_sha256": {"reporter": sha256_file(Path(__file__)),
                "comparison": sha256_file(root / "scripts/evaluation/check_parity.py"),
                "timing_summary": sha256_file(root / "scripts/model/report_quantization.py")},
            "limitations": ["Test results describe the already validation-selected Q8 artifact; no test-based reselection or tuning.",
                "The reference is the selected MLX adapter. No full-test BF16 output exists, so this is not a BF16/Q8 full-test quantization-parity claim.",
                "No deployment-safety, Pi performance, or numerical-equivalence claim follows from these scores.",
                "Local timing is diagnostic only; peak native RAM is not measured, and clock anomalies suppress throughput."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "models/q8-full-test-v1.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evaluation/baselines/q8-full-test-v1.json")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink() or args.output.parent.resolve() != (PROJECT_ROOT / "evaluation/baselines").resolve():
            raise ArtifactError("full-test summary requires new aggregate-only baseline evidence")
        result = build_report(PROJECT_ROOT, args.protocol)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (ArtifactError, EvaluationError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f"Full-test summary stopped ({type(exc).__name__}).", file=sys.stderr)
        if isinstance(exc, ArtifactError): print(str(exc), file=sys.stderr)
        return 1
    print(f"Q8 full-test report: {args.output}")
    print("Descriptive test results only; no deployment approval or test-based model selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
