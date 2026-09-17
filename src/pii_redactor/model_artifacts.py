"""CPU-only provenance and structural checks for local model packaging."""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
from collections import Counter
from pathlib import Path
from typing import Any


class ArtifactError(ValueError):
    """An immutable input or generated model failed a packaging check."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(path: Path) -> dict[str, dict[str, Any]]:
    """Hash runtime files, including templates, but not download caches/cards."""
    result = {}
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if item.suffix not in {".json", ".safetensors", ".txt", ".model", ".jinja", ".py", ".tiktoken"}:
            continue
        if item.is_symlink():
            raise ArtifactError("model artifacts must not contain symlinks")
        if item.is_file():
            result[relative.as_posix()] = {
                "bytes": item.stat().st_size, "sha256": sha256_file(item)
            }
    if not result:
        raise ArtifactError("model directory has no files")
    return result


def evaluation_fingerprint(path: Path) -> str:
    """Match the existing frozen evaluator's version-1 fingerprint exactly."""
    files = sorted({p for pattern in ("*.json", "*.safetensors", "*.txt", "*.model")
                    for p in path.glob(pattern)})
    if not files:
        raise ArtifactError("no fingerprint inputs")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.name.encode())
        digest.update(sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def validate_approved_inputs(root: Path, plan: dict[str, Any]) -> tuple[Path, Path]:
    if plan["approval"]["status"] != "approved" or plan["approval"]["gate"] != "H-045":
        raise ArtifactError("H-045 adapter approval is required")
    base, adapter = root / plan["base"]["path"], root / plan["adapter"]["path"]
    if evaluation_fingerprint(base) != plan["base"]["evaluation_fingerprint"]:
        raise ArtifactError("base fingerprint differs from the approved evaluation")
    for name, key in [("adapters.safetensors", "weights_sha256"),
                      ("adapter_config.json", "config_sha256")]:
        if sha256_file(adapter / name) != plan["adapter"][key]:
            raise ArtifactError("adapter checksum differs from approval")
    for directory in (base, adapter):
        if not directory.is_dir() or directory.is_symlink():
            raise ArtifactError("model inputs must be local directories")
    return base, adapter


def inspect_floating_weights(path: Path) -> dict[str, Any]:
    """Inspect safetensors headers without loading tensors, MLX, or PyTorch."""
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "qwen3":
        raise ArtifactError("expected the approved Qwen3 architecture")
    if "quantization" in config or "quantization_config" in config:
        raise ArtifactError("fused config still declares quantized weights")
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ArtifactError("no fused weights")
    dtypes: Counter[str] = Counter()
    names: set[str] = set()
    elements = 0
    for file in files:
        with file.open("rb") as handle:
            size_bytes = handle.read(8)
            if len(size_bytes) != 8:
                raise ArtifactError("truncated safetensors file")
            size = struct.unpack("<Q", size_bytes)[0]
            if not 2 <= size <= min(100_000_000, file.stat().st_size - 8):
                raise ArtifactError("invalid safetensors header size")
            header = json.loads(handle.read(size))
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if name in names or any(part in name for part in ("lora_a", "lora_b", ".scales")):
                raise ArtifactError("unfused or duplicate tensor")
            names.add(name)
            dtype = tensor["dtype"]
            if dtype not in {"BF16", "F16", "F32"}:
                raise ArtifactError("fused tensor is not floating point")
            count = 1
            for dimension in tensor["shape"]:
                if not isinstance(dimension, int) or dimension < 0:
                    raise ArtifactError("invalid tensor shape")
                count *= dimension
            start, end = tensor["data_offsets"]
            width = 4 if dtype == "F32" else 2
            if not 0 <= start <= end <= file.stat().st_size - 8 - size or end-start != count*width:
                raise ArtifactError("tensor data offsets do not match its shape")
            dtypes[dtype] += 1
            elements += count
    return {"tensor_count": len(names), "parameter_count": elements,
            "tensor_dtypes": dict(sorted(dtypes.items()))}


def check_tokenizer_parity(base: Path, candidate: Path, dataset: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    left, right = [AutoTokenizer.from_pretrained(str(p), local_files_only=True,
                                               trust_remote_code=False)
                   for p in (base, candidate)]
    if left.get_vocab() != right.get_vocab() or left.special_tokens_map != right.special_tokens_map:
        raise ArtifactError("tokenizer vocabulary or special tokens changed")
    count = 0
    digest = hashlib.sha256()
    for line in dataset.open(encoding="utf-8"):
        row = json.loads(line)
        kwargs = dict(tokenize=True, add_generation_prompt=True, enable_thinking=False,
                      return_dict=False)
        expected = left.apply_chat_template(row["messages"][:2], **kwargs)
        actual = right.apply_chat_template(row["messages"][:2], **kwargs)
        if expected != actual:
            raise ArtifactError("rendered prompt token IDs changed")
        digest.update(json.dumps(expected, separators=(",", ":")).encode())
        digest.update(b"\n")
        count += 1
    if not count:
        raise ArtifactError("empty tokenizer parity dataset")
    return {"records": count, "prompt_tokens_sha256": digest.hexdigest(),
            "vocabulary_equal": True, "special_tokens_equal": True,
            "thinking_enabled": False}


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
                   "chat_template.jinja")


def preserve_approved_tokenizer(base: Path, candidate: Path, backup: Path) -> dict[str, Any]:
    """Preserve generated serialization, then copy the exact approved tokenizer.

    Weights and model configuration are never modified. A pre-existing backup
    is a hard stop, including after an interrupted attempt.
    """
    if any(p.is_symlink() for p in (base, candidate, backup)):
        raise ArtifactError("tokenizer directories must not be symlinks")
    base, candidate, backup = base.resolve(), candidate.resolve(), backup.resolve()
    if not base.is_dir() or not candidate.is_dir():
        raise ArtifactError("tokenizer source and destination must exist")
    if base == candidate or base.is_relative_to(candidate) or candidate.is_relative_to(base):
        raise ArtifactError("tokenizer source and destination must be separate")
    if backup == base or backup.is_relative_to(base) or backup == candidate or backup.is_relative_to(candidate):
        raise ArtifactError("tokenizer backup must be outside model directories")
    if backup.exists() or not (base / "tokenizer_config.json").is_file() or not (base / "tokenizer.json").is_file():
        raise ArtifactError("tokenizer backup exists or approved tokenizer files are missing")
    source_files = {name: base / name for name in TOKENIZER_FILES
                    if (base / name).exists() or (base / name).is_symlink()}
    existing = {name: candidate / name for name in TOKENIZER_FILES
                if (candidate / name).exists() or (candidate / name).is_symlink()}
    if any(p.is_symlink() or not p.is_file() for p in [*source_files.values(), *existing.values()]):
        raise ArtifactError("tokenizer files must be regular files")
    before = {name: sha256_file(path) for name, path in existing.items()}
    expected = {name: sha256_file(path) for name, path in source_files.items()}
    backup.mkdir(parents=True)
    for name, path in existing.items():
        shutil.move(str(path), backup / name)
    for name, path in source_files.items():
        shutil.copy2(path, candidate / name)
    actual = {name: sha256_file(candidate / name) for name in TOKENIZER_FILES
              if (candidate / name).exists()}
    if actual != expected:
        raise ArtifactError("copied tokenizer differs from approved files")
    return {"policy": "exact approved tokenizer bytes; preserve generated serialization separately",
            "backup": str(backup), "generated_tokenizer_sha256": before,
            "approved_tokenizer_sha256": actual}
