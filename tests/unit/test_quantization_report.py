import json

import pytest

from pii_redactor.model_artifacts import ArtifactError, sha256_file

from scripts.model import report_quantization as report


def test_aggregate_contains_no_source_text_or_sample_ids(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"sample_id": "PRIVATE_ID", "raw_output": "PRIVATE_ENTITY",
        "elapsed_seconds": 2, "timings": {"prompt_ms": 500, "prompt_n": 100,
                                           "predicted_ms": 1000, "predicted_n": 11}}) + "\n")
    pair = {"status": "review_required", "candidate_metrics": {
        "micro": {"recall": 0.5}, "documents": {}, "validity": {}, "per_class": {}},
        "per_class_deltas": {}, "gates": [], "output_changes": {
            "raw_changed_records": 1, "entity_sequence_changed_records": 1,
            "records": [{"sample_id": "PRIVATE_ID"}]}}
    result = report.aggregate(pair, {"model_sha256": "model-hash", "predictions_sha256": "prediction-hash"},
                              {"model_bytes": 123}, predictions)
    serialized = json.dumps(result)
    assert "PRIVATE" not in serialized
    assert result["status"] == "review_required"
    assert result["local_mac_diagnostics"] == {"request_seconds": 2, "prompt_tokens_per_second": 200,
                                              "decode_tokens_per_second": 10, "peak_ram_bytes": None,
                                              "timing_status": "diagnostic_only", "timing_anomaly_records": 0}
    row = json.loads(predictions.read_text())
    row["timings"]["predicted_ms"] = 900_000
    predictions.write_text(json.dumps(row) + "\n")
    result = report.aggregate(pair, {"model_sha256": "m", "predictions_sha256": "p"}, {"model_bytes": 123}, predictions)
    assert result["local_mac_diagnostics"]["timing_anomaly_records"] == 1
    assert result["local_mac_diagnostics"]["decode_tokens_per_second"] is None
    assert result["local_mac_diagnostics"]["prompt_tokens_per_second"] is None


def test_failed_gates_still_write_report_and_never_overwrite(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(report, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "evaluation/baselines/comparison.json"
    output.parent.mkdir(parents=True)
    monkeypatch.setattr(report, "build_report", lambda *args: {"passing_formats": [], "status": "review_required"})
    assert report.main(["--output", str(output)]) == 2
    before = output.read_bytes()
    assert "Passing formats: none" in capsys.readouterr().out
    assert report.main(["--output", str(output)]) == 1
    assert output.read_bytes() == before


def test_report_path_must_stay_in_commit_safe_baselines(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "PROJECT_ROOT", tmp_path)
    assert report.main(["--output", str(tmp_path / "wrong.json")]) == 1
    assert not (tmp_path / "wrong.json").exists()


def complete_trial_fixture(root, monkeypatch):
    formats = ["Q8_0", "Q5_K_M", "Q4_K_M", "Q4_K_M-imatrix"]
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps({"approval": {"status": "approved", "gate": "H-045"},
        "validation": {"dataset": "valid.jsonl", "sha256": "frozen"}, "generation": {"seed": 42},
        "llama_cpp": {"revision": "pinned"}, "quantization": {"formats": formats,
            "maximum_absolute_recall_drop": .01, "maximum_absolute_complete_document_recall_drop": .01,
            "calibration_sha256": "calibration"}}))
    chain = root / "models/gguf/toolchain-v1/manifest.json"
    chain.parent.mkdir(parents=True)
    chain.write_text("{}")
    parent_dir = root / "models/gguf/v1-ckpt19000-bf16"
    parent_dir.mkdir()
    parent = parent_dir / "model-bf16.gguf"
    parent.write_bytes(b"BF16")
    monkeypatch.setattr(report, "validate_toolchain", lambda *a: None)
    monkeypatch.setattr(report, "check_parent", lambda *a: (parent, {"model_sha256": sha256_file(parent)}))
    def compare(*args, **kwargs):
        passing = "q8_0" in str(args[2]) or "-bf16-" in str(args[2])
        return {"status": "passed" if passing else "review_required", "candidate_metrics": {
            "micro": {}, "documents": {}, "validity": {}, "per_class": {}}, "per_class_deltas": {},
            "gates": [], "output_changes": {"raw_changed_records": 0, "entity_sequence_changed_records": 0}}
    monkeypatch.setattr(report, "compare", compare)
    for format in ["BF16", *formats]:
        slug = format.lower()
        model_dir = root / f"models/gguf/v1-ckpt19000-{slug}"
        model_dir.mkdir(exist_ok=True)
        model = model_dir / f"model-{slug}.gguf"
        model.write_bytes(format.encode())
        evaluation = root / f"evaluation/results/v1-gguf-{slug}-smoke-valid"
        evaluation.mkdir(parents=True)
        predictions = evaluation / "predictions.jsonl"
        predictions.write_text(json.dumps({"elapsed_seconds": 1, "timings": {"prompt_ms": 1, "prompt_n": 1,
            "predicted_ms": 1, "predicted_n": 2}}) + "\n")
        run = {"status": "completed", "records": 100, "model_sha256": sha256_file(model),
               "predictions_sha256": sha256_file(predictions), "temperature": 0}
        (evaluation / "manifest.json").write_text(json.dumps(run))
        (evaluation / "tokenizer-parity.json").write_text(json.dumps({"prompt_tokens_sha256": "prompts"}))
        artifact = {"status": "completed", "model_sha256": sha256_file(model), "model_bytes": model.stat().st_size,
                    "plan_sha256": sha256_file(plan_path), "parent": {"model_sha256": sha256_file(parent)}}
        artifact_path = model_dir / ("verified-conversion.json" if format == "BF16" else "manifest.json")
        artifact_path.write_text(json.dumps(artifact))
    matrix_dir = root / "models/gguf/v1-ckpt19000-imatrix"
    matrix_dir.mkdir()
    matrix = matrix_dir / "importance.gguf"
    matrix.write_bytes(b"matrix")
    metadata = {"status": "passed", "model_sha256": sha256_file(matrix), "model_bytes": 6,
        "chunk_count": 1, "chunk_size": 1024, "processed_tokens": 1024, "tensor_count": 392,
        "covered_layer_weights": 196, "complete_counts": True, "finite_nonnegative": True,
        "output_weight_calibrated": False, "limitations": [], "implementation_sha256": "inspector"}
    metadata_path = matrix_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    matrix_manifest = {"status": "completed", "model_sha256": sha256_file(matrix),
        "metadata_sha256": sha256_file(metadata_path), "calibration_sha256": "calibration",
        "parent_sha256": sha256_file(parent), "calibration_audit": {"split_overlap": {}}, "generation_seconds": 1}
    matrix_manifest_path = matrix_dir / "manifest.json"
    matrix_manifest_path.write_text(json.dumps(matrix_manifest))
    artifact_path = root / "models/gguf/v1-ckpt19000-q4_k_m-imatrix/manifest.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["imatrix_manifest_sha256"] = sha256_file(matrix_manifest_path)
    artifact_path.write_text(json.dumps(artifact))
    return plan_path


def test_complete_report_keeps_failed_candidates_and_does_not_select_on_test(tmp_path, monkeypatch):
    plan = complete_trial_fixture(tmp_path, monkeypatch)
    result = report.build_report(tmp_path, plan)
    assert result["passing_formats"] == ["Q8_0"]
    assert result["selected_format"] is None
    assert result["results"]["Q4_K_M"]["status"] == "review_required"
    assert result["importance_matrix"]["covered_layer_weights"] == 196
    assert report.build_report(tmp_path, plan) == result


@pytest.mark.parametrize("defect", ["temperature", "model_bytes", "parent", "prediction", "matrix"])
def test_complete_report_rejects_provenance_drift(tmp_path, monkeypatch, defect):
    plan = complete_trial_fixture(tmp_path, monkeypatch)
    model_dir = tmp_path / "models/gguf/v1-ckpt19000-q5_k_m"
    evaluation = tmp_path / "evaluation/results/v1-gguf-q5_k_m-smoke-valid"
    if defect in {"temperature", "model_bytes", "parent"}:
        path = (evaluation if defect == "temperature" else model_dir) / "manifest.json"
        data = json.loads(path.read_text())
        data[defect] = {"model_sha256": "different"} if defect == "parent" else 123
        path.write_text(json.dumps(data))
    elif defect == "prediction":
        (evaluation / "predictions.jsonl").write_text("changed")
    else:
        (tmp_path / "models/gguf/v1-ckpt19000-imatrix/importance.gguf").write_bytes(b"changed")
    with pytest.raises(ArtifactError): report.build_report(tmp_path, plan)
