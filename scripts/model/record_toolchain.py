#!/usr/bin/env python3
"""Record the pinned local GGUF toolchain without loading a model."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.model_artifacts import ArtifactError, sha256_file


def output(command, cwd):
    return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.PIPE).strip()


def collect(root: Path, plan: dict):
    checkout = root / plan["llama_cpp"]["path"]
    revision = output(["git", "rev-parse", "HEAD"], checkout)
    if revision != plan["llama_cpp"]["revision"]:
        raise ArtifactError("llama.cpp checkout differs from the approved revision")
    if output(["git", "status", "--porcelain", "--untracked-files=all"], checkout):
        raise ArtifactError("llama.cpp source checkout has local changes")
    build = checkout / "build"
    cache = build / "CMakeCache.txt"
    entries = {}
    for line in cache.read_text().splitlines():
        if line and not line.startswith(("#", "//")) and ":" in line and "=" in line:
            name, value = line.split("=", 1)
            entries[name.split(":", 1)[0]] = value
    expected = {"CMAKE_BUILD_TYPE": "Release", "GGML_METAL": "ON",
                "GGML_METAL_EMBED_LIBRARY": "ON", "LLAMA_BUILD_TESTS": "OFF",
                "LLAMA_BUILD_EXAMPLES": "OFF", "LLAMA_OPENSSL": "OFF"}
    if any(entries.get(key) != value for key, value in expected.items()):
        raise ArtifactError("native build options differ from the packaging recipe")
    files = [build / "bin" / name for name in ("llama-server", "llama-quantize", "llama-imatrix")]
    files.extend(sorted((build / "bin").glob("*.dylib")))
    artifacts = {}
    for path in files:
        if not path.is_file() or not path.resolve().is_relative_to(build.resolve()):
            raise ArtifactError("required toolchain binary is missing or outside the build")
        artifacts[path.relative_to(checkout).as_posix()] = {
            "bytes": path.stat().st_size, "sha256": sha256_file(path),
            "resolved_path": str(path.resolve()),
        }
    if len(files) == 3:
        raise ArtifactError("expected the Mac shared-library build")
    converter_python = root / ".venv-conversion/bin/python"
    packages = json.loads(output([str(converter_python), "-X",
                                  "pycache_prefix=/private/tmp/local-pii-redactor-conversion-pycache",
                                  "-m", "pip", "list", "--format=json"], root))
    return {"schema_version": 1, "status": "recorded", "repository": plan["llama_cpp"]["repository"],
            "revision": revision, "checkout": str(checkout), "source_clean": True,
            "platform": platform.platform(), "architecture": platform.machine(),
            "cmake_cache_sha256": sha256_file(cache), "build_options": expected,
            "compiler": output([entries["CMAKE_CXX_COMPILER"], "--version"], root),
            "cmake_version": output([entries["CMAKE_COMMAND"], "--version"], root),
            "converter_python": output([str(converter_python), "--version"], root),
            "converter_requirements_sha256": sha256_file(root / "requirements-conversion.txt"),
            "converter_packages": sorted(packages, key=lambda item: item["name"].lower()),
            "converter_script_sha256": sha256_file(checkout / "convert_hf_to_gguf.py"),
            "binaries": artifacts,
            "interpretation": "Build provenance only; no model conversion or parity claim."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "models/packaging-v1.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models/gguf/toolchain-v1/manifest.json")
    args = parser.parse_args(argv)
    try:
        destination = args.output.resolve()
        allowed = (PROJECT_ROOT / "models/gguf").resolve()
        if not destination.is_relative_to(allowed) or destination == allowed or args.output.is_symlink():
            raise ArtifactError("toolchain evidence must remain under models/gguf")
        if destination.exists():
            raise ArtifactError("toolchain evidence already exists")
        print("Verifying pinned source and recording native toolchain...", flush=True)
        evidence = collect(PROJECT_ROOT, json.loads(args.plan.read_text()))
        evidence.update(recorded_at_utc=datetime.now(UTC).isoformat(),
                        plan_sha256=sha256_file(args.plan), implementation_sha256=sha256_file(Path(__file__)))
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as handle:
            json.dump(evidence, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (ArtifactError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f"Toolchain recording failed ({type(exc).__name__}); no model was changed.", file=sys.stderr)
        return 1
    print(f"Toolchain manifest: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
