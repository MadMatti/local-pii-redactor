from pathlib import Path

from scripts.check_repository import repository_violations


def test_repository_check_rejects_model_artifact_suffix(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "weights.safetensors"
    artifact.write_bytes(b"safe-looking test bytes")
    monkeypatch.setattr("scripts.check_repository.PROJECT_ROOT", tmp_path)
    violations = repository_violations([artifact])
    assert violations == ["forbidden artifact extension: weights.safetensors"]


def test_repository_check_detects_private_key_marker(tmp_path, monkeypatch) -> None:
    file_path = tmp_path / "config.txt"
    marker = "-----BEGIN " + "PRIVATE KEY-----\n"
    file_path.write_text(marker, encoding="utf-8")
    monkeypatch.setattr("scripts.check_repository.PROJECT_ROOT", tmp_path)
    violations = repository_violations([file_path])
    assert violations == ["likely secret material: config.txt"]
