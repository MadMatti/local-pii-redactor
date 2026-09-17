#!/usr/bin/env python3
"""Evaluate a checksum-pinned GGUF with an owned, private local llama.cpp server."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.evaluation.llama_local import LocalRuntimeError, generation_request, local_server
from pii_redactor.evaluation.metrics import EvaluationError, evaluate_predictions, load_predictions, write_evaluation
from pii_redactor.model_artifacts import ArtifactError, TOKENIZER_FILES, sha256_file


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_runtime(client, tokenizer, model, context):
    props = client.request("/props")
    if (Path(props.get("model_path", "")).resolve() != model.resolve()
            or props.get("total_slots") != 1
            or props.get("default_generation_settings", {}).get("n_ctx") != context):
        raise LocalRuntimeError("native model or context differs from requested runtime")
    if props.get("ui") is not False or props.get("cors_proxy_enabled") is not False:
        raise LocalRuntimeError("native UI or proxy is unexpectedly enabled")
    if props.get("eos_token") != tokenizer.eos_token:
        raise LocalRuntimeError("native EOS token differs from approved tokenizer")
    eos = client.request("/tokenize", {"content": tokenizer.eos_token, "add_special": False, "parse_special": True})
    if eos.get("tokens") != [tokenizer.eos_token_id]:
        raise LocalRuntimeError("native EOS token ID differs from approved tokenizer")
    return {"model_path": props["model_path"], "context": context, "slots": 1,
            "eos_token_id": tokenizer.eos_token_id, "build_info": props.get("build_info"),
            "ui_enabled": False, "cors_proxy_enabled": False}


def verify_completion(result, eos_token_id):
    if result.get("truncated") or result.get("stop_type") not in {"eos", "limit"}:
        raise LocalRuntimeError("generation truncated or stopped unexpectedly")
    if (not isinstance(result.get("content"), str) or not isinstance(result.get("tokens"), list)
            or not result["tokens"] or any(type(t) is not int for t in result["tokens"])):
        raise LocalRuntimeError("generation response lacks text or token evidence")
    if result["stop_type"] == "eos" and result["tokens"][-1] != eos_token_id:
        raise LocalRuntimeError("native generation stopped on a different end-of-generation token")


def verify_prompts(client, tokenizer, rows, context, max_tokens):
    prompts = {}
    digest = hashlib.sha256()
    for row in rows:
        messages = row["messages"][:2]
        kwargs = {"add_generation_prompt": True, "enable_thinking": False}
        text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)
        if len(tokens) + max_tokens > context:
            raise LocalRuntimeError("context is insufficient; refusing prompt truncation")
        native_text = client.request("/apply-template", {
            "messages": messages, "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none"})
        if native_text.get("prompt") != text:
            raise LocalRuntimeError("GGUF chat template differs from approved non-thinking prompt")
        native_tokens = client.request("/tokenize", {"content": text, "add_special": False,
                                                     "parse_special": True})
        if native_tokens.get("tokens") != tokens:
            raise LocalRuntimeError("GGUF tokenizer differs from approved prompt token IDs")
        prompts[row["sample_id"]] = tokens
        digest.update(json.dumps(tokens, separators=(",", ":")).encode())
        digest.update(b"\n")
    return prompts, digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_ROOT / "models/base-qwen3-1.7b-4bit")
    parser.add_argument("--toolchain-manifest", type=Path, default=PROJECT_ROOT / "models/gguf/toolchain-v1/manifest.json")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if min(args.max_tokens, args.context, args.threads) < 1:
            raise LocalRuntimeError("token and thread limits must be positive")
        output = args.output_dir.resolve()
        allowed = (PROJECT_ROOT / "evaluation/results").resolve()
        if output == allowed or not output.is_relative_to(allowed) or args.output_dir.is_symlink():
            raise LocalRuntimeError("GGUF reports must stay under local evaluation/results")
        if args.model.is_symlink() or not args.model.resolve().is_relative_to((PROJECT_ROOT / "models/gguf").resolve()):
            raise LocalRuntimeError("GGUF model must remain under local models/gguf")
        if sha256_file(args.model) != args.expected_model_sha256 or sha256_file(args.dataset) != args.expected_dataset_sha256:
            raise LocalRuntimeError("frozen model or dataset checksum changed")
        toolchain = json.loads(args.toolchain_manifest.read_text())
        if toolchain.get("status") != "recorded" or not toolchain.get("source_clean"):
            raise LocalRuntimeError("verified toolchain evidence is required")
        checkout = Path(toolchain["checkout"])
        for name, artifact in toolchain["binaries"].items():
            path = checkout / name
            if not path.resolve().is_relative_to((checkout / "build").resolve()) or sha256_file(path) != artifact["sha256"]:
                raise LocalRuntimeError("native runtime differs from recorded toolchain")
        binary = checkout / "build/bin/llama-server"
        if "build/bin/llama-server" not in toolchain["binaries"]:
            raise LocalRuntimeError("server binary is not recorded")
        tokenizer_hashes = {name: sha256_file(args.tokenizer / name) for name in TOKENIZER_FILES
                            if (args.tokenizer / name).is_file()}
        rows = [json.loads(line) for line in args.dataset.open(encoding="utf-8")]
        ids = [row["sample_id"] for row in rows]
        if not ids or len(set(ids)) != len(ids):
            raise LocalRuntimeError("frozen dataset must contain unique record IDs")
        run_config = {"schema_version": 1, "runtime": "llama.cpp", "revision": toolchain["revision"],
                      "toolchain_sha256": sha256_file(args.toolchain_manifest),
                      "model": str(args.model.resolve()), "model_sha256": args.expected_model_sha256,
                      "dataset": str(args.dataset.resolve()), "dataset_sha256": args.expected_dataset_sha256,
                      "tokenizer_files_sha256": tokenizer_hashes,
                      "transformers_version": version("transformers"), "tokenizers_version": version("tokenizers"),
                      "seed": args.seed, "temperature": 0.0, "enable_thinking": False,
                      "max_tokens": args.max_tokens, "context": args.context, "threads": args.threads,
                      "cache_prompt": False, "flash_attention": True, "kv_dtype": "f16",
                      "gpu_layers": "all", "batch_size": 512, "ubatch_size": 512,
                      "implementation_sha256": {"runner": sha256_file(Path(__file__)),
                          "transport": sha256_file(PROJECT_ROOT / "src/pii_redactor/evaluation/llama_local.py")}}
        config_path, predictions_path = output / "run_config.json", output / "predictions.jsonl"
        if output.exists():
            if not args.resume or not config_path.is_file() or json.loads(config_path.read_text()) != run_config:
                raise LocalRuntimeError("existing GGUF run requires --resume and identical configuration")
            if any(path.is_symlink() for path in output.iterdir()):
                raise LocalRuntimeError("evaluation evidence must not contain symlinks")
        completed = load_predictions(predictions_path) if predictions_path.exists() else {}
        if not completed.keys() <= set(ids):
            raise LocalRuntimeError("resumed predictions contain unknown records")
        output.mkdir(parents=True, exist_ok=True)
        if not config_path.exists():
            save_json(config_path, run_config)
        if len(completed) != len(rows):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
            print("Starting private local GGUF runtime and verifying every frozen prompt...", flush=True)
            with local_server(binary, args.model.resolve(), context=args.context, threads=args.threads) as (client, command):
                runtime = verify_runtime(client, tokenizer, args.model, args.context)
                save_json(output / "last-runtime-check.json", runtime)
                prompts, prompt_hash = verify_prompts(client, tokenizer, rows, args.context, args.max_tokens)
                prompt_evidence = {"records": len(rows), "prompt_tokens_sha256": prompt_hash,
                                   "template_equal": True, "prompt_ids_equal": True}
                parity_path = output / "tokenizer-parity.json"
                if parity_path.exists() and json.loads(parity_path.read_text()) != prompt_evidence:
                    raise LocalRuntimeError("resumed prompt parity differs")
                save_json(parity_path, prompt_evidence)
                # Only the ephemeral port changes on resume; the API key is never persisted.
                save_json(output / "last-server-command.json", command)
                started = time.monotonic()
                with predictions_path.open("a", encoding="utf-8") as destination:
                    for index, row in enumerate(rows, 1):
                        if row["sample_id"] in completed:
                            continue
                        tokens = prompts[row["sample_id"]]
                        before = time.monotonic()
                        result = client.request("/completion", generation_request(tokens, args.max_tokens, args.seed))
                        verify_completion(result, tokenizer.eos_token_id)
                        prediction = {"sample_id": row["sample_id"], "raw_output": result["content"],
                                      "prompt_tokens": len(tokens),
                                      "generated_tokens": len(tokenizer.encode(result["content"], add_special_tokens=False)),
                                      "sampled_tokens": len(result["tokens"]), "generated_token_ids": result["tokens"],
                                      "elapsed_seconds": round(time.monotonic()-before, 6),
                                      "stop_type": result["stop_type"], "timings": result.get("timings", {})}
                        destination.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
                        destination.flush()
                        if index % 25 == 0 or index == len(rows):
                            print(f"Generated {index}/{len(rows)} records; elapsed={time.monotonic()-started:.1f}s", flush=True)
        result = evaluate_predictions(args.dataset, predictions_path, expected_dataset_sha256=args.expected_dataset_sha256)
        write_evaluation(result, output, title="Frozen GGUF model evaluation")
        save_json(output / "manifest.json", {**run_config, "status": "completed", "records": len(rows),
                                            "predictions_sha256": sha256_file(predictions_path),
                                            "completed_at_utc": datetime.now(UTC).isoformat()})
    except (LocalRuntimeError, ArtifactError, EvaluationError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"GGUF evaluation failed ({type(exc).__name__}); completed local predictions are preserved.", file=sys.stderr)
        if isinstance(exc, LocalRuntimeError):
            print(str(exc), file=sys.stderr)
        return 1
    print(f"Report: {output / 'report.md'}")
    print(f"Micro F1: {result.metrics['micro']['f1']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
