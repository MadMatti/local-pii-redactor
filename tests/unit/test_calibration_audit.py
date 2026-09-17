import json

import pytest

from pii_redactor.data.synthetic import generate_calibration_texts
from pii_redactor.model_artifacts import ArtifactError, sha256_file
from scripts.model import audit_calibration


def fixture(root, held_out=None):
    texts = list(generate_calibration_texts(count=500, seed=42))
    calibration = root / "data/calibration/pii_calibration.txt"
    calibration.parent.mkdir(parents=True)
    calibration.write_text(audit_calibration.SEPARATOR.join(texts) + "\n")
    digest = sha256_file(calibration)
    (calibration.parent / "manifest.json").write_text(json.dumps({"seed": 42, "records": 500,
        "uses_frozen_test_data": False, "artifact": {"sha256": digest}}))
    artifacts = {}
    for split in ("train", "valid", "test"):
        path = root / f"data/processed/{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        text = texts[0] if split == "train" or split == held_out else f"Separate {split} document"
        path.write_text(json.dumps({"messages": [{}, {"content": text}]}) + "\n")
        artifacts[path.relative_to(root).as_posix()] = {"sha256": sha256_file(path)}
    (root / "data/processed/manifest.json").write_text(json.dumps({"artifacts": artifacts}))
    frozen = root / "evaluation/results/v1-selected-ckpt19000-full-test/manifest.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_text(json.dumps({"dataset_sha256": artifacts["data/processed/test.jsonl"]["sha256"]}))
    generator = root / "src/pii_redactor/data/synthetic.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("# test generator provenance")
    return {"quantization": {"calibration_path": "data/calibration/pii_calibration.txt", "calibration_sha256": digest}}


@pytest.mark.parametrize("held_out", [None, "valid", "test"])
def test_calibration_allows_training_overlap_but_blocks_held_out_overlap(tmp_path, held_out):
    plan = fixture(tmp_path, held_out)
    result = audit_calibration.audit(tmp_path, plan)
    assert result["status"] == ("review_required" if held_out else "passed")
    assert result["split_overlap"]["train"]["exact_records"] == 1
    assert result["unique_exact_documents"] == 500
    assert result["regenerated_bytes_equal"]


def test_calibration_rejects_changed_bytes_and_generator(tmp_path, monkeypatch):
    plan = fixture(tmp_path)
    monkeypatch.setattr(audit_calibration, "generate_calibration_texts", lambda **kwargs: ["different corpus"])
    with pytest.raises(ArtifactError, match="reproduce"):
        audit_calibration.audit(tmp_path, plan)
    plan["quantization"]["calibration_sha256"] = "changed"
    with pytest.raises(ArtifactError, match="checksum"):
        audit_calibration.audit(tmp_path, plan)
