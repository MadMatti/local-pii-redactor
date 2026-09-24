#!/usr/bin/env python3
"""Check dense Qwen3 importance-matrix coverage and finite calibration statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.model_artifacts import ArtifactError, sha256_file


def inspect(reader, config, calibration, expected_chunks):
    import numpy as np

    def value(key):
        field = reader.fields.get(key)
        return field.contents() if field is not None else None

    if (value("general.type") != "imatrix" or value("imatrix.chunk_size") != 1024
            or expected_chunks < 1 or value("imatrix.chunk_count") != expected_chunks
            or value("imatrix.datasets") != [str(calibration)]):
        raise ArtifactError("importance matrix metadata or completed chunk coverage differs")
    widths = {"attn_q": config["hidden_size"], "attn_k": config["hidden_size"],
              "attn_v": config["hidden_size"],
              "attn_output": config["num_attention_heads"] * config["head_dim"],
              "ffn_gate": config["hidden_size"], "ffn_up": config["hidden_size"],
              "ffn_down": config["intermediate_size"]}
    expected = {}
    for layer in range(config["num_hidden_layers"]):
        for name, width in widths.items():
            expected[f"blk.{layer}.{name}.weight.in_sum2"] = width
            expected[f"blk.{layer}.{name}.weight.counts"] = 1
    if len(reader.tensors) != len(expected) or {t.name for t in reader.tensors} != set(expected):
        raise ArtifactError("importance matrix does not cover every expected dense layer weight")
    for tensor in reader.tensors:
        data = tensor.data
        # GGUF serialization may omit the trailing singleton matrix dimension.
        if (tensor.tensor_type.name != "F32" or list(tensor.shape) not in ([expected[tensor.name]], [expected[tensor.name], 1])
                or data.size != expected[tensor.name] or not np.isfinite(data).all()
                or (data < 0).any()):
            raise ArtifactError("importance matrix has invalid shapes, types, or activation statistics")
        if tensor.name.endswith(".counts"):
            if not (data == expected_chunks * 1024).all():
                raise ArtifactError("importance matrix contains incomplete activation counts")
        elif not (data > 0).any():
            raise ArtifactError("importance matrix contains an entirely unobserved layer weight")
    return {"schema_version": 1, "status": "passed", "chunk_count": expected_chunks,
            "chunk_size": 1024, "processed_tokens": expected_chunks * 1024,
            "tensor_count": len(expected), "covered_layer_weights": len(expected) // 2,
            "finite_nonnegative": True, "complete_counts": True,
            "output_weight_calibrated": False,
            "limitations": ["Native calibration processes all complete 1024-token chunks; a final partial chunk is unused.",
                            "Uses approved plain-text calibration, not inference chat prompts."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--chunks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if (args.output.exists() or args.output.is_symlink()
                or not args.output.resolve().is_relative_to((PROJECT_ROOT / "models/gguf").resolve())):
            raise ArtifactError("matrix inspection requires new local models/gguf evidence")
        local_gguf = PROJECT_ROOT / "vendor/llama.cpp/gguf-py"
        sys.path.insert(0, str(local_gguf))
        import gguf
        if not Path(gguf.__file__).resolve().is_relative_to(local_gguf.resolve()):
            raise ArtifactError("GGUF reader is not from the pinned checkout")
        result = inspect(gguf.GGUFReader(args.model, mode="r"), json.loads(args.config.read_text()),
                         args.calibration, args.chunks)
        result.update(model_sha256=sha256_file(args.model), model_bytes=args.model.stat().st_size,
                      implementation_sha256=sha256_file(Path(__file__)))
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (ArtifactError, OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"Matrix inspection stopped ({type(exc).__name__}).", file=sys.stderr)
        if isinstance(exc, ArtifactError):
            print(str(exc), file=sys.stderr)
        return 1
    print(f"Importance-matrix coverage passed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
