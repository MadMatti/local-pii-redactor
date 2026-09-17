import json
import struct
from types import SimpleNamespace

import pytest

from pii_redactor.model_artifacts import ArtifactError, evaluation_fingerprint, inventory, sha256_file
from scripts.model import convert_gguf, inspect_gguf


def source_and_reader(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {"model_type": "qwen3", "num_hidden_layers": 1, "hidden_size": 2,
              "intermediate_size": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
              "max_position_embeddings": 2048, "rope_theta": 1000000,
              "eos_token_id": 1, "bos_token_id": 0, "vocab_size": 2}
    (source / "config.json").write_text(json.dumps(config))
    (source / "tokenizer_config.json").write_text('{"chat_template":"test template"}')
    (source / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"start": 0}},
                                                     "added_tokens": [{"content": "end", "id": 1}]}))
    header = json.dumps({"weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode()
    (source / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(8))
    fields = {"general.architecture": "qwen3", "qwen3.block_count": 1, "qwen3.embedding_length": 2,
              "qwen3.feed_forward_length": 4, "qwen3.attention.head_count": 2, "qwen3.attention.head_count_kv": 1,
              "qwen3.context_length": 2048, "qwen3.rope.freq_base": 1000000,
              "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.eos_token_id": 1,
              "tokenizer.ggml.bos_token_id": 0, "tokenizer.ggml.add_bos_token": False,
              "tokenizer.chat_template": "test template", "tokenizer.ggml.tokens": ["start", "end"]}
    reader = SimpleNamespace(fields={k: SimpleNamespace(contents=lambda v=v: v) for k,v in fields.items()},
                             tensors=[SimpleNamespace(tensor_type=SimpleNamespace(name="BF16"), n_elements=4)])
    return source, reader


def test_complete_gguf_metadata_vocabulary_and_floating_checks(tmp_path):
    source, reader = source_and_reader(tmp_path)
    result = inspect_gguf.inspect(reader, source, floating=True)
    assert result["status"] == "passed"
    assert result["approved_vocabulary_entries_verified"] == 2
    assert result["parameter_count"] == 4
    reader.tensors[0].tensor_type.name = "Q4_K"
    with pytest.raises(ArtifactError, match="non-floating"):
        inspect_gguf.inspect(reader, source, floating=True)
    assert inspect_gguf.inspect(reader, source)["tensor_dtypes"] == {"Q4_K": 1}


@pytest.mark.parametrize("field,value", [("qwen3.block_count", 2), ("tokenizer.ggml.eos_token_id", 0),
                                        ("tokenizer.ggml.add_bos_token", True),
                                        ("tokenizer.chat_template", "changed"),
                                        ("tokenizer.ggml.tokens", ["wrong", "end"])])
def test_gguf_rejects_metadata_template_or_vocabulary_drift(tmp_path, field, value):
    source, reader = source_and_reader(tmp_path)
    reader.fields[field] = SimpleNamespace(contents=lambda: value)
    with pytest.raises(ArtifactError):
        inspect_gguf.inspect(reader, source, floating=True)


def test_conversion_refuses_unsafe_or_existing_output_before_models(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    monkeypatch.setattr(convert_gguf, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(convert_gguf, "validate_approved_inputs", lambda *args: pytest.fail("must refuse before input reads"))
    existing = tmp_path / "models/gguf/existing"
    existing.mkdir(parents=True)
    for output in (existing, tmp_path / "elsewhere", tmp_path / "models/gguf"):
        assert convert_gguf.main(["--plan", str(plan), "--output-dir", str(output)]) == 1


def test_conversion_checks_generation_hashes_and_gate_before_running_converter(tmp_path, monkeypatch):
    source, _ = source_and_reader(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    generation = {"max_tokens": 384, "seed": 42, "temperature": 0.0, "enable_thinking": False}
    for directory, model_hash, adapter_hash in ((reference, "base", evaluation_fingerprint(adapter)),
                                               (candidate, evaluation_fingerprint(source), None)):
        directory.mkdir()
        predictions = directory / "predictions.jsonl"
        predictions.write_text("{}\n")
        (directory / "manifest.json").write_text(json.dumps({"model_fingerprint": model_hash,
            "adapter_fingerprint": adapter_hash, "dataset_sha256": "frozen",
            "predictions_sha256": sha256_file(predictions), **generation}))
    plan = {"validation": {"reference_predictions": "reference/predictions.jsonl", "dataset": "valid.jsonl", "sha256": "frozen"},
            "base": {"evaluation_fingerprint": "base"}, "adapter": {"path": "adapter"}, "generation": generation}
    monkeypatch.setattr(convert_gguf, "compare", lambda *args: {"status": "review_required"})
    with pytest.raises(ArtifactError, match="parity"):
        convert_gguf.verify_generation(tmp_path, plan, source, candidate)
    monkeypatch.setattr(convert_gguf, "compare", lambda *args: {"status": "passed", "gates": [], "output_changes": {}})
    assert convert_gguf.verify_generation(tmp_path, plan, source, candidate)["gates"] == []
    (candidate / "predictions.jsonl").write_text("changed")
    with pytest.raises(ArtifactError, match="changed after evaluation"):
        convert_gguf.verify_generation(tmp_path, plan, source, candidate)


def test_verify_existing_conversion_never_reruns_weights_or_rewrites_original_evidence(tmp_path, monkeypatch):
    source, _ = source_and_reader(tmp_path)
    run = tmp_path / "models/fused/run"
    run.mkdir(parents=True)
    checkout = tmp_path / "vendor/llama.cpp"
    checkout.mkdir(parents=True)
    converter = checkout / "convert_hf_to_gguf.py"
    converter.write_text("# pinned converter")
    inspector = tmp_path / "scripts/model/inspect_gguf.py"
    inspector.parent.mkdir(parents=True)
    inspector.write_text("# pinned inspector")
    plan = {"fusion": {"output": "source", "run_dir": "models/fused/run"},
            "llama_cpp": {"path": "vendor/llama.cpp", "revision": "pinned"}}
    config = tmp_path / "plan.json"
    config.write_text(json.dumps(plan))
    before = inventory(source)
    (run / "verified-fusion.json").write_text(json.dumps({"status": "completed", "artifacts": before,
                                                        "plan_sha256": sha256_file(config)}))
    toolchain = tmp_path / "toolchain.json"
    toolchain.write_text(json.dumps({"status": "recorded", "revision": "pinned", "converter_packages": [],
                                    "converter_script_sha256": sha256_file(converter)}))
    output = tmp_path / "models/gguf/existing"
    output.mkdir(parents=True)
    model = output / "model-bf16.gguf"
    model.write_bytes(b"existing model")
    command = [str(tmp_path / ".venv-conversion/bin/python"), "-X",
               "pycache_prefix=/private/tmp/local-pii-redactor-conversion-pycache",
               str(converter), str(source), "--outtype", "bf16", "--outfile", str(model)]
    original = {"status": "failed", "return_code": 0, "command": command, "source_artifacts": before,
                "plan_sha256": sha256_file(config), "toolchain_sha256": sha256_file(toolchain)}
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(original))
    original_bytes = manifest.read_bytes()
    monkeypatch.setattr(convert_gguf, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(convert_gguf, "validate_approved_inputs", lambda *args: None)
    monkeypatch.setattr(convert_gguf, "verify_generation", lambda *args: {"gates": []})
    monkeypatch.setattr(convert_gguf.subprocess, "check_output",
                        lambda cmd, **kwargs: "pinned" if "rev-parse" in cmd else "" if "status" in cmd else "[]")
    calls = []
    def inspect_command(cmd, **kwargs):
        assert str(converter) not in cmd
        assert str(inspector) in cmd
        calls.append(cmd)
        (output / "metadata.json").write_text("{}")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(convert_gguf.subprocess, "run", inspect_command)
    args = ["--plan", str(config), "--toolchain", str(toolchain), "--output-dir", str(output), "--verify-existing"]
    assert convert_gguf.main(args) == 0
    assert manifest.read_bytes() == original_bytes
    assert model.read_bytes() == b"existing model"
    assert json.loads((output / "verified-conversion.json").read_text())["status"] == "completed"
    assert len(calls) == 1
    assert convert_gguf.main(args) == 1
    assert manifest.read_bytes() == original_bytes
    assert len(calls) == 1
