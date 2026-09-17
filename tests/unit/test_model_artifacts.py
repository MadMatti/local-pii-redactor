from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from pii_redactor.model_artifacts import (
    ArtifactError, evaluation_fingerprint, inspect_floating_weights, inventory,
    preserve_approved_tokenizer, sha256_file, validate_approved_inputs,
)
from scripts.model import fuse_adapter, record_toolchain, verify_fusion


def weights(path: Path, *, dtype="BF16", name="model.weight", offsets=None):
    width = 4 if dtype in {"F32", "U32"} else 2
    header = json.dumps({name: {"dtype": dtype, "shape": [2, 2],
                               "data_offsets": offsets or [0, 4 * width]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(4 * width))


def model(tmp_path):
    path = tmp_path / "base"
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"qwen3"}')
    weights(path / "model.safetensors")
    return path


def test_floating_header_inspection_without_ml_frameworks(tmp_path):
    path = model(tmp_path)
    assert inspect_floating_weights(path) == {
        "tensor_count": 1, "parameter_count": 4, "tensor_dtypes": {"BF16": 1}
    }


@pytest.mark.parametrize("dtype,name,offsets", [
    ("U32", "model.weight", None), ("BF16", "model.lora_a", None),
    ("BF16", "model.scales", None), ("BF16", "model.weight", [0, 100]),
    ("BF16", "model.weight", [1, 8]),
])
def test_reject_unfused_or_malformed_weights(tmp_path, dtype, name, offsets):
    path = model(tmp_path)
    weights(path / "model.safetensors", dtype=dtype, name=name, offsets=offsets)
    with pytest.raises(ArtifactError):
        inspect_floating_weights(path)


def test_quantized_config_and_truncated_headers_fail(tmp_path):
    path = model(tmp_path)
    (path / "config.json").write_text('{"model_type":"qwen3","quantization":{}}')
    with pytest.raises(ArtifactError, match="quantized"):
        inspect_floating_weights(path)
    (path / "config.json").write_text('{"model_type":"qwen3"}')
    for content in (b"short", struct.pack("<Q", 10**10) + b"{}"):
        (path / "model.safetensors").write_bytes(content)
        with pytest.raises(ArtifactError):
            inspect_floating_weights(path)


def test_inventory_includes_jinja_and_refuses_symlinks(tmp_path):
    path = model(tmp_path)
    template = path / "chat_template.jinja"
    template.write_text("{{ messages }}")
    cache = path / ".cache"
    cache.mkdir()
    (cache / "volatile.json").write_text("{}")
    entries = inventory(path)
    assert ".cache/volatile.json" not in entries
    assert entries[template.name]["sha256"] == sha256_file(template)
    (path / "linked.json").symlink_to(path / "config.json")
    with pytest.raises(ArtifactError, match="symlinks"):
        inventory(path)


def approved_plan(tmp_path):
    base = model(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights(adapter / "adapters.safetensors")
    (adapter / "adapter_config.json").write_text("{}")
    return {"approval": {"gate": "H-045", "status": "approved"},
            "base": {"path": "base", "evaluation_fingerprint": evaluation_fingerprint(base)},
            "adapter": {"path": "adapter",
                        "weights_sha256": sha256_file(adapter / "adapters.safetensors"),
                        "config_sha256": sha256_file(adapter / "adapter_config.json")}}


def test_approval_and_hashes_are_required(tmp_path):
    plan = approved_plan(tmp_path)
    assert validate_approved_inputs(tmp_path, plan) == (tmp_path / "base", tmp_path / "adapter")
    plan["approval"]["status"] = "pending"
    with pytest.raises(ArtifactError, match="approval"):
        validate_approved_inputs(tmp_path, plan)
    plan["approval"]["status"] = "approved"
    (tmp_path / "adapter/adapter_config.json").write_text('{"changed":true}')
    with pytest.raises(ArtifactError, match="checksum"):
        validate_approved_inputs(tmp_path, plan)


@pytest.mark.parametrize("destination", ["models/fused", "elsewhere/model", "models/fused/existing"])
def test_fusion_cli_rejects_unsafe_or_existing_outputs_before_mlx(tmp_path, monkeypatch, destination):
    plan = approved_plan(tmp_path)
    data = tmp_path / "valid.jsonl"
    data.write_text("{}\n")
    plan["validation"] = {"dataset": data.name, "sha256": sha256_file(data)}
    plan["fusion"] = {"output": destination, "run_dir": "models/fused/run", "minimum_free_bytes": 1}
    (tmp_path / "models/fused/existing").mkdir(parents=True)
    config = tmp_path / "plan.json"
    config.write_text(json.dumps(plan))
    monkeypatch.setattr(fuse_adapter, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(fuse_adapter.subprocess, "run", lambda *a, **k: pytest.fail("MLX must not run"))
    assert fuse_adapter.main(["--plan", str(config)]) == 1
    assert not (tmp_path / "models/fused/run").exists()


def test_legacy_fingerprint_matches_frozen_evaluator(tmp_path):
    from scripts.evaluation.run_model_baseline import _directory_fingerprint
    path = model(tmp_path)
    assert evaluation_fingerprint(path) == _directory_fingerprint(path)


def tokenizers(tmp_path):
    base = model(tmp_path)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (base / "tokenizer.json").write_text('{"approved": true}')
    (base / "tokenizer_config.json").write_text('{"chat_template":"inline"}')
    (base / "added_tokens.json").write_text('{"special":1}')
    (candidate / "tokenizer.json").write_text('{"reserialized":true}')
    (candidate / "tokenizer_config.json").write_text('{}')
    (candidate / "chat_template.jinja").write_text("generated")
    (candidate / "config.json").write_text('{"model_type":"qwen3"}')
    weights(candidate / "model.safetensors")
    return base, candidate, tmp_path / "backup"


def test_restore_exact_tokenizer_preserves_originals_and_weights(tmp_path):
    base, candidate, backup = tokenizers(tmp_path)
    base_before, generated_before = inventory(base), inventory(candidate)
    result = preserve_approved_tokenizer(base, candidate, backup)
    assert inventory(base) == base_before
    for name in ("tokenizer.json", "tokenizer_config.json", "added_tokens.json"):
        assert (candidate / name).read_bytes() == (base / name).read_bytes()
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        assert sha256_file(backup / name) == generated_before[name]["sha256"]
        assert result["generated_tokenizer_sha256"][name] == generated_before[name]["sha256"]
    assert not (candidate / "chat_template.jinja").exists()
    for name in ("model.safetensors", "config.json"):
        assert sha256_file(candidate / name) == generated_before[name]["sha256"]
    with pytest.raises(ArtifactError, match="backup exists"):
        preserve_approved_tokenizer(base, candidate, backup)


@pytest.mark.parametrize("case", ["same", "nested", "backup_inside", "missing", "symlink", "broken_link"])
def test_tokenizer_restore_rejects_unsafe_paths_without_mutation(tmp_path, case):
    base, candidate, backup = tokenizers(tmp_path)
    original = (candidate / "tokenizer.json").read_bytes()
    if case == "same":
        base = candidate
    elif case == "nested":
        candidate = base / "nested"
        candidate.mkdir()
    elif case == "backup_inside":
        backup = candidate / "backup"
    elif case == "missing":
        candidate = tmp_path / "missing"
    elif case == "symlink":
        link = tmp_path / "link"
        link.symlink_to(candidate, target_is_directory=True)
        candidate = link
    else:
        (candidate / "added_tokens.json").symlink_to(tmp_path / "missing")
    with pytest.raises(ArtifactError):
        preserve_approved_tokenizer(base, candidate, backup)
    assert not backup.exists()
    assert (tmp_path / "candidate/tokenizer.json").read_bytes() == original


def recovery_fixture(tmp_path, monkeypatch):
    base, candidate, _ = tokenizers(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights(adapter / "adapters.safetensors")
    (adapter / "adapter_config.json").write_text("{}")
    plan = {"approval": {"gate": "H-045", "status": "approved"},
            "base": {"path": "base", "evaluation_fingerprint": evaluation_fingerprint(base)},
            "adapter": {"path": "adapter", "weights_sha256": sha256_file(adapter / "adapters.safetensors"),
                        "config_sha256": sha256_file(adapter / "adapter_config.json")},
            "fusion": {"output": "models/fused/model", "run_dir": "models/fused/run"}}
    output, run = [tmp_path / plan["fusion"][key] for key in ("output", "run_dir")]
    run.mkdir(parents=True)
    candidate.rename(output)
    dataset = tmp_path / "valid.jsonl"
    dataset.write_text("{}\n")
    plan["validation"] = {"dataset": dataset.name, "sha256": sha256_file(dataset)}
    config = tmp_path / "plan.json"
    config.write_text(json.dumps(plan))
    original = {"status": "failed", "return_code": 0, "plan_sha256": sha256_file(config),
                "output": str(output), "inputs": {"base": inventory(base), "adapter": inventory(adapter)}}
    (run / "manifest.json").write_text(json.dumps(original))
    implementation = tmp_path / "src/pii_redactor/model_artifacts.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("# test implementation\n")
    monkeypatch.setattr(verify_fusion, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(verify_fusion, "check_tokenizer_parity", lambda *args: {"records": 100})
    return config, output, run


def test_recovery_cli_preserves_failed_manifest_and_refuses_overwrite(tmp_path, monkeypatch):
    config, output, run = recovery_fixture(tmp_path, monkeypatch)
    original = (run / "manifest.json").read_bytes()
    weight_hash = sha256_file(output / "model.safetensors")
    args = ["--plan", str(config), "--restore-approved-tokenizer"]
    assert verify_fusion.main(args) == 0
    evidence = (run / "verified-fusion.json").read_bytes()
    assert json.loads(evidence)["status"] == "completed"
    assert (run / "manifest.json").read_bytes() == original
    assert sha256_file(output / "model.safetensors") == weight_hash
    assert verify_fusion.main(args) == 1
    assert (run / "verified-fusion.json").read_bytes() == evidence


def test_recovery_cli_checks_validation_before_restoring(tmp_path, monkeypatch):
    config, output, run = recovery_fixture(tmp_path, monkeypatch)
    (tmp_path / "valid.jsonl").write_text("changed\n")
    before = inventory(output)
    assert verify_fusion.main(["--plan", str(config), "--restore-approved-tokenizer"]) == 1
    assert inventory(output) == before
    assert not (run / "generated-tokenizer").exists()


def test_recovery_cli_records_parity_failure_without_rewriting_history(tmp_path, monkeypatch, capsys):
    config, _, run = recovery_fixture(tmp_path, monkeypatch)
    original = (run / "manifest.json").read_bytes()
    def fail(*args):
        raise ArtifactError("rendered prompt token IDs changed")
    monkeypatch.setattr(verify_fusion, "check_tokenizer_parity", fail)
    assert verify_fusion.main(["--plan", str(config), "--restore-approved-tokenizer"]) == 1
    evidence = json.loads((run / "verified-fusion.json").read_text())
    assert evidence["status"] == "failed"
    assert evidence["failure_reason"] == "rendered prompt token IDs changed"
    assert (run / "manifest.json").read_bytes() == original
    assert "generated" not in capsys.readouterr().err


def test_toolchain_requires_pinned_clean_source(tmp_path, monkeypatch):
    plan = {"llama_cpp": {"path": "vendor/llama.cpp", "revision": "approved"}}
    monkeypatch.setattr(record_toolchain, "output", lambda *args: "different")
    with pytest.raises(ArtifactError, match="revision"):
        record_toolchain.collect(tmp_path, plan)
    monkeypatch.setattr(record_toolchain, "output",
                        lambda command, cwd: "approved" if command[1] == "rev-parse" else " M source.cpp")
    with pytest.raises(ArtifactError, match="local changes"):
        record_toolchain.collect(tmp_path, plan)


def test_toolchain_records_artifacts_without_loading_models(tmp_path, monkeypatch):
    checkout = tmp_path / "vendor/llama.cpp"
    build = checkout / "build"
    (build / "bin").mkdir(parents=True)
    options = {"CMAKE_BUILD_TYPE": "Release", "GGML_METAL": "ON", "GGML_METAL_EMBED_LIBRARY": "ON",
               "LLAMA_BUILD_TESTS": "OFF", "LLAMA_BUILD_EXAMPLES": "OFF", "LLAMA_OPENSSL": "OFF",
               "CMAKE_CXX_COMPILER": "clang++", "CMAKE_COMMAND": "cmake"}
    (build / "CMakeCache.txt").write_text("\n".join(f"{k}:STRING={v}" for k, v in options.items()))
    for name in ("llama-server", "llama-quantize", "llama-imatrix", "libllama.dylib"):
        (build / "bin" / name).write_bytes(b"fake binary")
    (checkout / "convert_hf_to_gguf.py").write_text("# fake converter")
    (tmp_path / "requirements-conversion.txt").write_text("fake==1\n")
    def fake_output(command, cwd):
        if command[:2] == ["git", "rev-parse"]:
            return "approved"
        if command[:2] == ["git", "status"]:
            return ""
        if "--format=json" in command:
            return '[{"name":"fake","version":"1"}]'
        return "tool version"
    monkeypatch.setattr(record_toolchain, "output", fake_output)
    plan = {"llama_cpp": {"path": "vendor/llama.cpp", "revision": "approved", "repository": "official"}}
    result = record_toolchain.collect(tmp_path, plan)
    assert result["source_clean"]
    assert len(result["binaries"]) == 4
    assert result["converter_packages"] == [{"name": "fake", "version": "1"}]
    (build / "CMakeCache.txt").write_text("GGML_METAL:BOOL=OFF")
    with pytest.raises(ArtifactError, match="build options"):
        record_toolchain.collect(tmp_path, plan)


def test_toolchain_cli_preserves_existing_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(record_toolchain, "PROJECT_ROOT", tmp_path)
    report = tmp_path / "models/gguf/toolchain-v1/manifest.json"
    report.parent.mkdir(parents=True)
    report.write_text("previous evidence")
    monkeypatch.setattr(record_toolchain, "collect", lambda *args: pytest.fail("must refuse before collection"))
    assert record_toolchain.main(["--output", str(report)]) == 1
    assert report.read_text() == "previous evidence"
    assert record_toolchain.main(["--output", str(tmp_path / "outside.json")]) == 1
