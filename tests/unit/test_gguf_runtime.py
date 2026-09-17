import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from pii_redactor.data.prepared import sample_from_text_and_values
from pii_redactor.evaluation import llama_local
from pii_redactor.evaluation.llama_local import LocalClient, LocalRuntimeError, generation_request
from pii_redactor.model_artifacts import sha256_file
from pii_redactor.schema import EntityType
from scripts.evaluation import run_gguf_baseline as runner


class Tokenizer:
    eos_token = "end"
    eos_token_id = 22

    def encode(self, text, **kwargs):
        return [11]

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        return [10, 20, 30] if tokenize else "private rendered prompt"


class Client:
    def __init__(self):
        self.calls = []

    def request(self, endpoint, body=None):
        self.calls.append((endpoint, body))
        if endpoint == "/props":
            return {"model_path": self.model, "total_slots": 1,
                    "default_generation_settings": {"n_ctx": self.context},
                    "ui": False, "cors_proxy_enabled": False, "eos_token": "end"}
        if endpoint == "/apply-template":
            return {"prompt": "private rendered prompt"}
        if endpoint == "/tokenize":
            return {"tokens": [22] if body["content"] == "end" else [10, 20, 30]}
        return {"content": '{"entities":[{"type":"PERSON_NAME","text":"Ada"}]}',
                "tokens": [11, 22], "truncated": False, "stop_type": "eos"}


def test_native_greedy_request_disables_penalties_and_cache():
    result = generation_request([12, 34], 384, 42)
    assert result["prompt"] == [12, 34]
    assert result["samplers"] == ["temperature"]
    assert result["temperature"] == 0
    assert result["repeat_penalty"] == 1
    assert result["repeat_last_n"] == 0
    assert result["cache_prompt"] is False
    assert result["n_predict"] == 384
    assert "grammar" not in result
    with pytest.raises(LocalRuntimeError):
        generation_request(["private text"], 384, 42)


def test_transport_never_redirects_or_logs_private_error_bodies(monkeypatch):
    client = LocalClient(12345, "test-key")
    assert client.origin == "http://127.0.0.1:12345"
    def fail(*args, **kwargs):
        raise URLError("private rendered prompt and secret response")
    monkeypatch.setattr(client.opener, "open", fail)
    with pytest.raises(LocalRuntimeError) as exc:
        client.request("/completion", {"prompt": "private text"})
    assert "private" not in str(exc.value)
    with pytest.raises(LocalRuntimeError, match="endpoint"):
        client.request("//remote.example/upload")
    with pytest.raises(LocalRuntimeError, match="redirect"):
        llama_local.NoRedirects().redirect_request(None, None, 302, "", {}, "https://remote.example")


def test_prompt_parity_checks_templates_ids_and_context_without_generation():
    rows = [{"sample_id": "one", "messages": []}]
    client = Client()
    prompts, digest = runner.verify_prompts(client, Tokenizer(), rows, 1024, 384)
    assert prompts == {"one": [10, 20, 30]}
    assert len(digest) == 64
    assert [e for e, _ in client.calls] == ["/apply-template", "/tokenize"]
    with pytest.raises(LocalRuntimeError, match="truncation"):
        runner.verify_prompts(client, Tokenizer(), rows, 385, 384)
    client.request = lambda *args: {"prompt": "changed private content"}
    with pytest.raises(LocalRuntimeError, match="chat template"):
        runner.verify_prompts(client, Tokenizer(), rows, 1024, 384)
    client.request = lambda endpoint, body: {"prompt": "private rendered prompt", "tokens": [99]}
    with pytest.raises(LocalRuntimeError, match="tokenizer"):
        runner.verify_prompts(client, Tokenizer(), rows, 1024, 384)


def test_owned_server_is_loopback_offline_and_always_stopped(monkeypatch, tmp_path):
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, address): assert address == ("127.0.0.1", 0)
        def getsockname(self): return ("127.0.0.1", 12345)
    class Process:
        stopped = False
        def poll(self): return 0 if self.stopped else None
        def terminate(self): self.stopped = True
        def wait(self, timeout): return 0
    process, captured = Process(), {}
    def spawn(command, **kwargs):
        captured.update(command=command, **kwargs)
        return process
    monkeypatch.setattr(llama_local.socket, "socket", Socket)
    monkeypatch.setattr(llama_local.subprocess, "Popen", spawn)
    monkeypatch.setattr(LocalClient, "request", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setenv("LLAMA_ARG_RPC", "remote.example")
    monkeypatch.setenv("HF_TOKEN", "private-test-credential")
    with pytest.raises(RuntimeError, match="test interruption"):
        with llama_local.local_server(tmp_path / "server", tmp_path / "model", context=1024, threads=4):
            raise RuntimeError("test interruption")
    assert process.stopped
    assert "--offline" in captured["command"]
    assert "--no-webui" in captured["command"]
    assert "127.0.0.1" in captured["command"]
    assert "LLAMA_ARG_RPC" not in captured["env"]
    assert "HF_TOKEN" not in captured["env"]
    assert captured["env"]["LLAMA_API_KEY"] not in str(captured["command"])


def test_mocked_gguf_evaluation_and_completed_resume_do_not_need_frameworks(tmp_path, monkeypatch, capsys):
    row = sample_from_text_and_values(split="validation", template_family="test:gguf", variant=1,
                                     synthetic_kind="unit", text="Ada writes.",
                                     values=((EntityType.PERSON_NAME, "Ada"),)).to_json_record()
    dataset = tmp_path / "valid.jsonl"
    dataset.write_text(json.dumps(row) + "\n")
    model = tmp_path / "models/gguf/model.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fake gguf")
    checkout = tmp_path / "vendor/llama.cpp"
    binary = checkout / "build/bin/llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"fake binary")
    toolchain = tmp_path / "toolchain.json"
    toolchain.write_text(json.dumps({"status": "recorded", "source_clean": True, "checkout": str(checkout),
                                    "revision": "pinned", "binaries": {
                                        "build/bin/llama-server": {"sha256": sha256_file(binary)}}}))
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    transport = tmp_path / "src/pii_redactor/evaluation/llama_local.py"
    transport.parent.mkdir(parents=True)
    transport.write_text("# test implementation")
    client, starts = Client(), []
    @contextmanager
    def server(*args, **kwargs):
        starts.append(True)
        client.model = str(args[1])
        client.context = kwargs["context"]
        yield client, ["server", "--port", "12345"]
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "local_server", server)
    monkeypatch.setattr(runner, "version", lambda name: "test")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer())))
    output = tmp_path / "evaluation/results/test"
    args = ["--model", str(model), "--expected-model-sha256", sha256_file(model),
            "--dataset", str(dataset), "--expected-dataset-sha256", sha256_file(dataset),
            "--output-dir", str(output), "--tokenizer", str(tokenizer), "--toolchain-manifest", str(toolchain)]
    assert runner.main(args) == 0
    before = (output / "predictions.jsonl").read_bytes()
    assert len(starts) == 1
    assert runner.main(args + ["--resume"]) == 0
    assert len(starts) == 1
    assert (output / "predictions.jsonl").read_bytes() == before
    assert runner.main(args) == 1
    assert "Ada" not in capsys.readouterr().out
    binary.write_bytes(b"changed binary")
    assert runner.main(args + ["--resume"]) == 1
    assert (output / "predictions.jsonl").read_bytes() == before


def test_native_eos_and_context_shift_cannot_silently_change_decoding():
    response = {"content": "{}", "tokens": [10, 22], "stop_type": "eos", "truncated": False}
    runner.verify_completion(response, 22)
    with pytest.raises(LocalRuntimeError, match="different end-of-generation"):
        runner.verify_completion(response, 99)
    response["truncated"] = True
    with pytest.raises(LocalRuntimeError, match="truncated"):
        runner.verify_completion(response, 22)
