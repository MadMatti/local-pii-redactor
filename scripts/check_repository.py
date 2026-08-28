#!/usr/bin/env python3
"""Reject accidentally tracked local artifacts, large files, and likely secrets."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 5 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".gguf",
    ".imatrix",
    ".npy",
    ".npz",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
FORBIDDEN_PREFIXES = (
    "data/raw/",
    "data/intermediate/",
    "data/processed/full_source/",
    "data/processed/length_views/",
    "data/processed/stages/",
    "data/processed/validation/",
    "models/adapters/",
    "models/base-qwen3-1.7b-4bit/",
    "models/fused/",
    "models/gguf/",
    "training/runs/",
)
SECRET_PATTERNS = (
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"ghp_[A-Za-z0-9]{30,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(rb"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
)


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        PROJECT_ROOT / item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    ]


def repository_violations(paths: list[Path]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_TRACKED_BYTES:
            violations.append(f"tracked file exceeds 5 MiB: {relative}")
        if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            violations.append(f"forbidden artifact extension: {relative}")
        if relative.startswith(FORBIDDEN_PREFIXES):
            violations.append(f"forbidden local-artifact path: {relative}")
        content = path.read_bytes()
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            violations.append(f"likely secret material: {relative}")
    return violations


def main() -> int:
    try:
        violations = repository_violations(tracked_files())
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        print(f"Repository check failed to run: {exc}", file=sys.stderr)
        return 1
    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        return 2
    print("Repository artifact and secret checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
