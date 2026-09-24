import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pii_redactor.model_artifacts import ArtifactError, sha256_file
from scripts.model import quantize_gguf as quant


def test_disk_reserve_is_preserved_for_every_format():
    sizes = {fmt: quant.minimum_free_bytes(3_447_348_928, fmt) for fmt in quant.FORMATS}
    assert all(size > quant.RESERVE_BYTES for size in sizes.values())
    assert sizes["Q8_0"] > sizes["Q5_K_M"] > sizes["Q4_K_M"]
    assert sizes["Q4_K_M"] == sizes["Q4_K_M-imatrix"]


def fixture(root, monkeypatch, *, free_bytes=20 * 1024**3):
    plan = root / "plan.json"
    plan.write_text(json.dumps({"approval": {"gate": "H-045", "status": "approved"},
                               "quantization": {"formats": list(quant.FORMATS)},
                               "fusion": {"output": "fused"}}))
    parent = root / "parent.gguf"
    parent.write_bytes(b"floating common parent")
    toolchain = root / "toolchain.json"
    toolchain.write_text("{}")
    inspector = root / "scripts/model/inspect_gguf.py"
    inspector.parent.mkdir(parents=True)
    inspector.write_text("# test inspector")
    monkeypatch.setattr(quant, "PROJECT_ROOT", root)
    monkeypatch.setattr(quant, "validate_toolchain", lambda *args: root / "vendor")
    monkeypatch.setattr(quant, "check_parent", lambda *args: (parent, {"model_sha256": sha256_file(parent)}))
    monkeypatch.setattr(quant, "inspect_floating_weights", lambda *args: {})
    monkeypatch.setattr(quant.shutil, "disk_usage", lambda *args: SimpleNamespace(free=free_bytes))
    return ["--plan", str(plan), "--toolchain", str(toolchain)], parent


def test_native_quantization_uses_common_parent_and_preserves_evidence(tmp_path, monkeypatch):
    args, parent = fixture(tmp_path, monkeypatch)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if "--output" in command:
            Path(command[command.index("--output") + 1]).write_text("{}")
        else:
            assert command[-4] == str(parent)
            assert command[-2:] == ["Q4_K_M", "4"]
            assert "--allow-requantize" not in command
            Path(command[-3]).write_bytes(b"quantized result")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(quant.subprocess, "run", run)
    assert quant.main(args + ["--format", "Q4_K_M"]) == 0
    report = tmp_path / "models/gguf/v1-ckpt19000-q4_k_m/manifest.json"
    before = report.read_bytes()
    assert json.loads(before)["status"] == "completed"
    assert parent.read_bytes() == b"floating common parent"
    assert len(calls) == 2
    assert quant.main(args + ["--format", "Q4_K_M"]) == 1
    assert report.read_bytes() == before
    assert len(calls) == 2


def test_low_disk_blocks_before_creating_output_or_running_quantizer(tmp_path, monkeypatch):
    args, parent = fixture(tmp_path, monkeypatch, free_bytes=quant.RESERVE_BYTES)
    monkeypatch.setattr(quant.subprocess, "run", lambda *a, **k: pytest.fail("must not quantize"))
    assert quant.main(args + ["--format", "Q4_K_M"]) == 1
    assert not (tmp_path / "models/gguf/v1-ckpt19000-q4_k_m").exists()
    assert parent.read_bytes() == b"floating common parent"


def test_unverified_imatrix_and_unsafe_output_are_rejected(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(quant.subprocess, "run", lambda *a, **k: pytest.fail("must not quantize"))
    assert quant.main(args + ["--format", "Q4_K_M-imatrix"]) == 1
    assert not (tmp_path / "models/gguf/v1-ckpt19000-q4_k_m-imatrix").exists()
    assert quant.main(args + ["--format", "Q4_K_M", "--output-dir", str(tmp_path / "outside")]) == 1


def test_parent_quality_gate_cannot_be_skipped(tmp_path, monkeypatch):
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    model = parent_dir / "model-bf16.gguf"
    model.write_bytes(b"parent")
    model_hash = sha256_file(model)
    metadata = parent_dir / "metadata.json"
    metadata.write_text(json.dumps({"status": "passed", "tensor_dtypes": {"BF16": 1}, "model_sha256": model_hash}))
    toolchain = tmp_path / "toolchain.json"
    toolchain.write_text("{}")
    reference = tmp_path / "evaluation/results/v1-fused-bf16-smoke-valid/predictions.jsonl"
    reference.parent.mkdir(parents=True)
    reference.write_text("{}\n")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "predictions.jsonl").write_text("{}\n")
    generation = {"max_tokens": 384, "seed": 42, "temperature": 0, "enable_thinking": False}
    plan = {"validation": {"sha256": "frozen", "dataset": "valid.jsonl"}, "generation": generation}
    manifest = {"status": "completed", "model_sha256": model_hash, "plan_sha256": "approved-plan",
                "toolchain_sha256": sha256_file(toolchain), "metadata_sha256": sha256_file(metadata),
                "generation_gate": {"fused_predictions_sha256": sha256_file(reference)}}
    (parent_dir / "manifest.json").write_text(json.dumps(manifest))
    (candidate / "manifest.json").write_text(json.dumps({"status": "completed", "model_sha256": model_hash,
        "dataset_sha256": "frozen", "toolchain_sha256": sha256_file(toolchain),
        "predictions_sha256": sha256_file(candidate / "predictions.jsonl"), **generation}))
    monkeypatch.setattr(quant, "compare", lambda *args: {"status": "review_required"})
    with pytest.raises(ArtifactError, match="parity"):
        quant.check_parent(tmp_path, plan, parent_dir, candidate, toolchain, "approved-plan")
    monkeypatch.setattr(quant, "compare", lambda *args: {"status": "passed", "gates": []})
    assert quant.check_parent(tmp_path, plan, parent_dir, candidate, toolchain, "approved-plan")[0] == model
    with pytest.raises(ArtifactError, match="not verified"):
        quant.check_parent(tmp_path, plan, parent_dir, candidate, toolchain, "wrong-plan")


@pytest.mark.parametrize("coverage_passes", [False, True])
def test_imatrix_quantization_requires_verified_coverage(tmp_path, monkeypatch, coverage_passes):
    args, parent = fixture(tmp_path, monkeypatch)
    plan_path = tmp_path / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["quantization"]["calibration_sha256"] = "approved-calibration"
    plan_path.write_text(json.dumps(plan))
    inspector = tmp_path / "scripts/model/inspect_imatrix.py"
    inspector.write_text("# matrix inspector")
    directory = tmp_path / "models/gguf/v1-ckpt19000-imatrix"
    directory.mkdir(parents=True)
    matrix = directory / "importance.gguf"
    matrix.write_bytes(b"matrix")
    metadata = directory / "metadata.json"
    metadata.write_text(json.dumps({"status": "passed", "model_sha256": sha256_file(matrix),
        "complete_counts": coverage_passes, "covered_layer_weights": 196,
        "implementation_sha256": sha256_file(inspector)}))
    (directory / "manifest.json").write_text(json.dumps({"status": "completed", "model": str(matrix),
        "model_sha256": sha256_file(matrix), "parent_sha256": sha256_file(parent),
        "calibration_sha256": "approved-calibration", "plan_sha256": sha256_file(plan_path),
        "toolchain_sha256": sha256_file(tmp_path / "toolchain.json"), "metadata_sha256": sha256_file(metadata)}))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if "--output" in command:
            Path(command[command.index("--output") + 1]).write_text("{}")
        else:
            assert command[1:3] == ["--imatrix", str(matrix)]
            assert command[-4] == str(parent) and command[-2] == "Q4_K_M"
            Path(command[-3]).write_bytes(b"calibrated Q4")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(quant.subprocess, "run", run)
    assert quant.main(args + ["--format", "Q4_K_M-imatrix"]) == (0 if coverage_passes else 1)
    assert len(calls) == (2 if coverage_passes else 0)
