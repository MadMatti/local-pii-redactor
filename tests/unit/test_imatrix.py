import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pii_redactor.model_artifacts import ArtifactError, sha256_file
from scripts.model import build_imatrix as build
from scripts.model.inspect_imatrix import inspect


def reader_fixture():
    config = {"hidden_size": 2, "num_attention_heads": 1, "head_dim": 2,
              "intermediate_size": 4, "num_hidden_layers": 1}
    values = {"general.type": "imatrix", "imatrix.chunk_size": 1024,
              "imatrix.chunk_count": 2, "imatrix.datasets": ["/approved.txt"]}
    tensors = []
    for name in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
        for suffix in ("in_sum2", "counts"):
            width = 1 if suffix == "counts" else 4 if name == "ffn_down" else 2
            tensors.append(SimpleNamespace(name=f"blk.0.{name}.weight.{suffix}", shape=[width, 1],
                tensor_type=SimpleNamespace(name="F32"), data=np.full((1, width), 2048.0)))
    reader = SimpleNamespace(fields={k: SimpleNamespace(contents=lambda v=v: v) for k, v in values.items()},
                             tensors=tensors)
    return reader, config


def test_complete_dense_matrix_passes():
    reader, config = reader_fixture()
    result = inspect(reader, config, Path("/approved.txt"), 2)
    assert result["covered_layer_weights"] == 7
    assert result["processed_tokens"] == 2048
    for tensor in reader.tensors:
        tensor.shape = tensor.shape[:1]
    assert inspect(reader, config, Path("/approved.txt"), 2)["status"] == "passed"


@pytest.mark.parametrize("defect", ["missing", "nan", "negative", "zero", "partial", "shape", "dtype"])
def test_matrix_defects_fail_closed(defect):
    reader, config = reader_fixture()
    if defect == "missing": reader.tensors.pop()
    elif defect == "nan": reader.tensors[0].data[0, 0] = np.nan
    elif defect == "negative": reader.tensors[0].data[0, 0] = -1
    elif defect == "zero": reader.tensors[0].data[:] = 0
    elif defect == "partial": reader.tensors[1].data[:] = 1024
    elif defect == "shape": reader.tensors[0].shape = [1, 2]
    elif defect == "dtype": reader.tensors[0].tensor_type.name = "F16"
    with pytest.raises(ArtifactError):
        inspect(reader, config, Path("/approved.txt"), 2)


def test_metadata_wrong_dataset_and_chunk_count_fail():
    reader, config = reader_fixture()
    for dataset, chunks in [("/test.txt", 2), ("/approved.txt", 1), ("/approved.txt", 0)]:
        with pytest.raises(ArtifactError): inspect(reader, config, Path(dataset), chunks)


def test_chunk_log_is_strict_and_environment_discards_overrides(monkeypatch):
    line = "compute_imatrix: computing over 71 chunks, n_ctx=1024, batch_size=1024, n_seq=1\n"
    assert build.completed_chunks(line) == 71
    for bad in ("", line + line, line.replace("1024", "512")):
        with pytest.raises(ArtifactError): build.completed_chunks(bad)
    monkeypatch.setenv("LLAMA_ARG_MODEL", "unapproved")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("GGML_BACKEND_PATH", "unapproved")
    env = build.native_environment()
    assert "LLAMA_ARG_MODEL" not in env and "HF_TOKEN" not in env and "GGML_BACKEND_PATH" not in env
    assert env["HF_HUB_OFFLINE"] == "1"


def build_fixture(root, monkeypatch, *, audit_status="passed", free=10 * 1024**3):
    calibration = root / "calibration.txt"
    calibration.write_text("PRIVATE CALIBRATION CONTENT")
    parent = root / "parent.gguf"
    parent.write_bytes(b"floating parent")
    plan = root / "plan.json"
    plan.write_text(json.dumps({"approval": {"gate": "H-045", "status": "approved"},
        "quantization": {"formats": ["Q4_K_M-imatrix"], "calibration_path": "calibration.txt",
                         "calibration_sha256": sha256_file(calibration)}, "fusion": {"output": "fused"}}))
    toolchain = root / "toolchain.json"
    toolchain.write_text(json.dumps({"binaries": {"build/bin/llama-imatrix": {}}}))
    inspector = root / "scripts/model/inspect_imatrix.py"
    inspector.parent.mkdir(parents=True)
    inspector.write_text("# inspector")
    monkeypatch.setattr(build, "PROJECT_ROOT", root)
    monkeypatch.setattr(build, "validate_toolchain", lambda *args: root / "vendor")
    monkeypatch.setattr(build, "check_parent", lambda *args: (parent, {"model_sha256": sha256_file(parent)}))
    monkeypatch.setattr(build, "audit", lambda *args: {"status": audit_status})
    monkeypatch.setattr(build.shutil, "disk_usage", lambda *args: SimpleNamespace(free=free))
    return ["--plan", str(plan), "--toolchain", str(toolchain),
            "--output-dir", str(root / "models/gguf/matrix")]


@pytest.mark.parametrize("kwargs", [{"audit_status": "review_required"}, {"free": build.RESERVE_BYTES}])
def test_unsafe_calibration_or_disk_blocks_before_native(tmp_path, monkeypatch, kwargs):
    args = build_fixture(tmp_path, monkeypatch, **kwargs)
    monkeypatch.setattr(build.subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
    assert build.main(args) == 1
    assert not (tmp_path / "models/gguf/matrix").exists()


def test_build_keeps_inputs_and_refuses_overwrite(tmp_path, monkeypatch, capsys):
    args = build_fixture(tmp_path, monkeypatch)
    def run(command, **kwargs):
        if "--output-file" in command:
            assert command[command.index("--chunks") + 1] == "-1"
            assert "--no-ppl" in command and "--parse-special" not in command
            Path(command[command.index("--output-file") + 1]).write_bytes(b"matrix")
            kwargs["stdout"].write("compute_imatrix: computing over 2 chunks, n_ctx=1024, batch_size=1024, n_seq=1\n")
        else:
            Path(command[command.index("--output") + 1]).write_text(json.dumps({"status": "passed",
                "implementation_sha256": sha256_file(tmp_path / "scripts/model/inspect_imatrix.py")}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(build.subprocess, "run", run)
    assert build.main(args) == 0
    manifest = tmp_path / "models/gguf/matrix/manifest.json"
    before = manifest.read_bytes()
    assert json.loads(before)["status"] == "completed"
    assert build.main(args) == 1
    assert manifest.read_bytes() == before
    assert "PRIVATE CALIBRATION CONTENT" not in capsys.readouterr().out
