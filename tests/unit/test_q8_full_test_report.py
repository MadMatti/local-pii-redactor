import json

import pytest

from pii_redactor.model_artifacts import ArtifactError, sha256_file
from scripts.model import report_q8_test as report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def prediction(*, stop="eos", raw='{"entities": []}'):
    tokens = [123, 151645] if stop == "eos" else [123] * 4
    return {"sample_id": "PRIVATE_SAMPLE", "raw_output": raw, "stop_type": stop,
            "generated_token_ids": tokens, "sampled_tokens": len(tokens), "prompt_tokens": 3,
            "elapsed_seconds": 1, "timings": {"prompt_ms": 100, "prompt_n": 3,
                                                "predicted_ms": 500, "predicted_n": len(tokens)}}


def test_diagnostics_count_invalid_and_limited_outputs_without_text(tmp_path):
    rows = [prediction(), prediction(stop="limit", raw="PRIVATE_INVALID_OUTPUT")]
    path = tmp_path / "predictions.jsonl"
    path.write_text("\n".join(map(json.dumps, rows)) + "\n")
    result = report.generation_diagnostics(path, context=10, max_tokens=4, eos_token_id=151645)
    assert result["records"] == 2
    assert result["stop_types"] == {"eos": 1, "limit": 1}
    assert result["schema_invalid_records"] == 1
    assert result["schema_invalid_stop_types"] == {"limit": 1}
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("change", [{"generated_token_ids": [123, 999]}, {"sampled_tokens": 1},
    {"prompt_tokens": 8}, {"stop_type": "word"}, {"stop_type": "limit"}, {"generated_token_ids": []}])
def test_diagnostics_reject_runtime_evidence_drift(tmp_path, change):
    row = prediction()
    row.update(change)
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ArtifactError):
        report.generation_diagnostics(path, context=10, max_tokens=4, eos_token_id=151645)


def evidence_fixture(root, monkeypatch):
    model = root / "models/gguf/q8/model.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"Q8 model")
    selection_path = root / "evaluation/baselines/quantization.json"
    write_json(selection_path, {"passing_formats": ["Q8_0"], "results": {
        "Q8_0": {"model_sha256": sha256_file(model)}}})
    write_json(root / "models/packaging-v1.json", {})
    toolchain = root / "models/gguf/toolchain-v1/manifest.json"
    write_json(toolchain, {})
    monkeypatch.setattr(report, "validate_toolchain", lambda *a: None)
    runner = root / "scripts/evaluation/run_gguf_baseline.py"
    transport = root / "src/pii_redactor/evaluation/llama_local.py"
    for path in (runner, transport):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen implementation")
    generation = {"context": 10, "max_tokens": 4, "seed": 42, "temperature": 0, "enable_thinking": False}
    implementation = {"runner": sha256_file(runner), "transport": sha256_file(transport)}
    reference = root / "evaluation/results/adapter/predictions.jsonl"
    reference.parent.mkdir(parents=True)
    reference.write_text(json.dumps(prediction()) + "\n")
    write_json(reference.parent / "manifest.json", {"predictions_sha256": sha256_file(reference),
        "dataset_sha256": "frozen", **generation})
    output = root / "evaluation/results/q8-test"
    output.mkdir()
    predictions = output / "predictions.jsonl"
    predictions.write_bytes(reference.read_bytes())
    run = {"status": "completed", "model_sha256": sha256_file(model), "dataset_sha256": "frozen",
        "records": 1, "revision": "pinned", "toolchain_sha256": sha256_file(toolchain), **generation,
        "implementation_sha256": implementation, "tokenizer_files_sha256": {}, "tokenizers_version": "1",
        "transformers_version": "1", "predictions_sha256": sha256_file(predictions)}
    write_json(output / "manifest.json", run)
    write_json(root / "evaluation/results/v1-gguf-q8_0-smoke-valid/manifest.json", run)
    write_json(output / "tokenizer-parity.json", {"records": 1, "template_equal": True,
        "prompt_ids_equal": True, "prompt_tokens_sha256": "prompts"})
    write_json(output / "last-runtime-check.json", {"context": 10, "model_path": str(model),
        "eos_token_id": 151645, "slots": 1, "ui_enabled": False, "cors_proxy_enabled": False})
    protocol = {"approval": {"status": "approved"}, "selection_basis": {"dataset_partition": "validation",
        "passing_format": "Q8_0", "report": str(selection_path.relative_to(root)),
        "report_sha256": sha256_file(selection_path)}, "model": {"path": str(model.relative_to(root)),
        "sha256": sha256_file(model), "bytes": model.stat().st_size, "checkpoint": 19000},
        "test": {"path": "test.jsonl", "sha256": "frozen", "records": 1, "positive_records": 0,
                 "negative_records": 1, "gold_entities": 0}, "generation": generation,
        "implementation": {"llama_cpp_revision": "pinned", "runner_sha256": sha256_file(runner),
                           "transport_sha256": sha256_file(transport)},
        "comparison": {"reference": str(reference.relative_to(root)), "reference_sha256": sha256_file(reference)},
        "output_dir": str(output.relative_to(root))}
    protocol_path = root / "protocol.json"
    write_json(protocol_path, protocol)
    return protocol_path, output


def test_complete_evidence_passes_without_touching_inputs(tmp_path, monkeypatch):
    protocol_path, output = evidence_fixture(tmp_path, monkeypatch)
    before = (output / "predictions.jsonl").read_bytes()
    protocol, run, predictions, reference, prompt = report.check_evidence(tmp_path, protocol_path)
    assert run["records"] == 1 and prompt["prompt_ids_equal"]
    assert predictions.read_bytes() == before
    assert reference.read_bytes() == before


@pytest.mark.parametrize("defect", ["approval", "test_selection", "model", "run_config", "prediction",
    "tokenizer", "reference", "runtime", "incomplete"])
def test_full_test_report_rejects_unapproved_or_changed_evidence(tmp_path, monkeypatch, defect):
    protocol_path, output = evidence_fixture(tmp_path, monkeypatch)
    if defect in {"approval", "test_selection"}:
        data = json.loads(protocol_path.read_text())
        if defect == "approval": data["approval"]["status"] = "pending"
        else: data["selection_basis"]["dataset_partition"] = "test"
        write_json(protocol_path, data)
    elif defect in {"run_config", "incomplete"}:
        path = output / "manifest.json"
        data = json.loads(path.read_text())
        data["temperature" if defect == "run_config" else "status"] = 1 if defect == "run_config" else "running"
        write_json(path, data)
    elif defect == "tokenizer":
        write_json(output / "tokenizer-parity.json", {"records": 1, "template_equal": False})
    elif defect == "runtime":
        data = json.loads((output / "last-runtime-check.json").read_text())
        data["ui_enabled"] = True
        write_json(output / "last-runtime-check.json", data)
    else:
        path = {"model": tmp_path / "models/gguf/q8/model.gguf", "prediction": output / "predictions.jsonl",
                "reference": tmp_path / "evaluation/results/adapter/predictions.jsonl"}[defect]
        path.write_bytes(b"changed")
    with pytest.raises(ArtifactError): report.check_evidence(tmp_path, protocol_path)


def test_cli_preserves_existing_summary_and_rejects_wrong_output_path(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "evaluation/baselines/report.json"
    write_json(output, {"existing": True})
    monkeypatch.setattr(report, "build_report", lambda *a: pytest.fail("must not recompute"))
    assert report.main(["--output", str(output)]) == 1
    assert json.loads(output.read_text()) == {"existing": True}
    assert report.main(["--output", str(tmp_path / "wrong.json")]) == 1


def mocked_comparison():
    metrics = {"dataset": {"records": 1}, "micro": {"tp": 0, "fp": 0, "fn": 0},
        "documents": {"pii_records": 0, "pii_free_records": 1}, "validity": {"schema_valid_rate": 1.0},
        "per_class": {}}
    return {"candidate_metrics": metrics, "reference_metrics": metrics, "status": "review_required",
            "gates": [{"passed": False}], "per_class_deltas": {},
            "output_changes": {"raw_changed_records": 0, "entity_sequence_changed_records": 0}}


def test_full_summary_is_descriptive_deterministic_and_text_free(tmp_path, monkeypatch):
    protocol, _ = evidence_fixture(tmp_path, monkeypatch)
    for name in ("scripts/evaluation/check_parity.py", "scripts/model/report_quantization.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen reporting helper")
    monkeypatch.setattr(report, "compare", lambda *a: mocked_comparison())
    result = report.build_report(tmp_path, protocol)
    assert result["status"] == "completed_descriptive_full_test"
    assert result["deployment_approved"] is False
    assert "gates" not in result["candidate"] and "status" not in result["candidate"]
    assert "PRIVATE" not in json.dumps(result)
    assert "BF16" in " ".join(result["limitations"])
    assert report.build_report(tmp_path, protocol) == result


def test_full_summary_rejects_wrong_test_composition(tmp_path, monkeypatch):
    protocol, _ = evidence_fixture(tmp_path, monkeypatch)
    pair = mocked_comparison()
    pair["candidate_metrics"]["dataset"]["records"] = 2
    monkeypatch.setattr(report, "compare", lambda *a: pair)
    with pytest.raises(ArtifactError, match="composition"):
        report.build_report(tmp_path, protocol)
