import json

from pii_redactor.data.synthetic import generate_positive_samples
from pii_redactor.data.validation import _load_and_validate_file


def test_file_validator_accepts_exact_prepared_record(tmp_path) -> None:
    record = generate_positive_samples(split="train", count=1, seed=42)[
        0
    ].to_json_record()
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    errors: list[str] = []

    records, statistics = _load_and_validate_file(
        path, expected_split="train", errors=errors
    )

    assert errors == []
    assert len(records) == 1
    assert statistics["negative_records"] == 0


def test_file_validator_rejects_assistant_span_disagreement(tmp_path) -> None:
    record = generate_positive_samples(split="train", count=1, seed=42)[
        0
    ].to_json_record()
    record["messages"][2]["content"] = '{"entities":[]}'
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    errors: list[str] = []

    _load_and_validate_file(path, expected_split="train", errors=errors)

    assert any("assistant entities and metadata spans differ" in error for error in errors)
