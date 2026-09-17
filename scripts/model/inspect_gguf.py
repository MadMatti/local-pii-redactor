#!/usr/bin/env python3
"""Verify GGUF architecture and complete vocabulary against the approved local source."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.model_artifacts import ArtifactError, inspect_floating_weights, sha256_file


def inspect(reader, source, *, floating=False):
    config = json.loads((source / "config.json").read_text())
    tokenizer = json.loads((source / "tokenizer.json").read_text())
    tokenizer_config = json.loads((source / "tokenizer_config.json").read_text())
    def value(name):
        field = reader.fields.get(name)
        return field.contents() if field is not None else None
    expected = {"general.architecture": "qwen3", "qwen3.block_count": config["num_hidden_layers"],
                "qwen3.embedding_length": config["hidden_size"],
                "qwen3.feed_forward_length": config["intermediate_size"],
                "qwen3.attention.head_count": config["num_attention_heads"],
                "qwen3.attention.head_count_kv": config["num_key_value_heads"],
                "qwen3.context_length": config["max_position_embeddings"],
                "qwen3.rope.freq_base": config["rope_theta"],
                "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.eos_token_id": config["eos_token_id"],
                "tokenizer.ggml.bos_token_id": config["bos_token_id"],
                "tokenizer.ggml.add_bos_token": False}
    for name, expected_value in expected.items():
        if value(name) != expected_value:
            raise ArtifactError(f"GGUF metadata differs from approved source: {name}")
    template = tokenizer_config["chat_template"]
    if (source / "chat_template.jinja").exists():
        template = (source / "chat_template.jinja").read_text()
    if value("tokenizer.chat_template") != template:
        raise ArtifactError("GGUF embedded chat template changed")
    tokens = value("tokenizer.ggml.tokens")
    if not isinstance(tokens, list) or len(tokens) != config["vocab_size"]:
        raise ArtifactError("GGUF vocabulary size changed")
    vocabulary = dict(tokenizer["model"]["vocab"])
    vocabulary.update({token["content"]: token["id"] for token in tokenizer["added_tokens"]})
    for text, index in vocabulary.items():
        if tokens[index] != text:
            raise ArtifactError("GGUF vocabulary token text or ID changed")
    dtypes = Counter(tensor.tensor_type.name for tensor in reader.tensors)
    parameters = sum(int(tensor.n_elements) for tensor in reader.tensors)
    source_weights = inspect_floating_weights(source)
    if parameters != source_weights["parameter_count"] or len(reader.tensors) != source_weights["tensor_count"]:
        raise ArtifactError("GGUF tensor or parameter count differs from fused weights")
    if floating and (not dtypes.get("BF16") or set(dtypes) - {"BF16", "F32"}):
        raise ArtifactError("BF16 conversion unexpectedly contains non-floating weights")
    return {"schema_version": 1, "status": "passed", "metadata": expected,
            "tokenizer_pre": value("tokenizer.ggml.pre"), "vocabulary_size": len(tokens),
            "approved_vocabulary_entries_verified": len(vocabulary),
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "tensor_count": len(reader.tensors), "parameter_count": parameters,
            "tensor_dtypes": dict(sorted(dtypes.items())),
            "interpretation": "Static metadata and vocabulary checks; runtime prompt/EOS behavior and generation parity remain required."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--floating", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink() or not args.output.resolve().is_relative_to((PROJECT_ROOT / "models/gguf").resolve()):
            raise ArtifactError("GGUF metadata evidence must be new and under local models/gguf")
        # Match the official converter's explicit pinned-package import. Editable
        # .pth discovery may be disabled for hidden virtual-environment files.
        local_gguf = PROJECT_ROOT / "vendor/llama.cpp/gguf-py"
        sys.path.insert(0, str(local_gguf))
        import gguf
        if not Path(gguf.__file__).resolve().is_relative_to(local_gguf.resolve()):
            raise ArtifactError("GGUF reader did not load from the pinned checkout")
        GGUFReader = gguf.GGUFReader
        result = inspect(GGUFReader(args.model, mode="r"), args.source, floating=args.floating)
        result.update(model_sha256=sha256_file(args.model), model_bytes=args.model.stat().st_size,
                      implementation_sha256=sha256_file(Path(__file__)))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (ArtifactError, OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"GGUF inspection failed ({type(exc).__name__}).", file=sys.stderr)
        if isinstance(exc, ArtifactError):
            print(str(exc), file=sys.stderr)
        return 1
    print(f"GGUF metadata and vocabulary checks passed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
