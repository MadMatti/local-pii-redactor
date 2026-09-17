import json

import pytest

from pii_redactor.data.prepared import sample_from_text_and_values
from pii_redactor.evaluation.metrics import EvaluationError, sha256_file
from pii_redactor.schema import EntityType
from scripts.evaluation.check_parity import compare, main


def fixture(tmp_path, output):
    row = sample_from_text_and_values(split="validation", template_family="test:parity", variant=1,
                                     synthetic_kind="unit", text="Ada writes.",
                                     values=((EntityType.PERSON_NAME, "Ada"),)).to_json_record()
    dataset, reference, candidate = [tmp_path / name for name in ("data.jsonl", "ref.jsonl", "cand.jsonl")]
    dataset.write_text(json.dumps(row) + "\n")
    reference.write_text(json.dumps({"sample_id": row["sample_id"],
                                     "raw_output": row["messages"][2]["content"]}) + "\n")
    candidate.write_text(json.dumps({"sample_id": row["sample_id"], "raw_output": output}) + "\n")
    return dataset, reference, candidate, sha256_file(dataset)


def test_formatting_difference_is_not_semantic_regression(tmp_path):
    result = compare(*fixture(tmp_path, '{ "entities": [ { "type": "PERSON_NAME", "text": "Ada" } ] }'))
    assert result["status"] == "passed"
    assert result["output_changes"]["entity_sequence_changed_records"] == 0


def test_recall_regression_writes_review_status_and_tolerances_are_explicit(tmp_path):
    args = fixture(tmp_path, '{"entities":[]}')
    result = compare(*args)
    assert result["status"] == "review_required"
    assert result["per_class_deltas"]["PERSON_NAME"]["fn"] == 1
    assert compare(*args, recall_tolerance=1, document_tolerance=1)["status"] == "passed"
    for tolerance in (float("nan"), float("inf"), -0.01, 1.1):
        with pytest.raises(EvaluationError):
            compare(*args, recall_tolerance=tolerance)


def test_invalid_schema_is_never_waived_by_recall_tolerance(tmp_path):
    result = compare(*fixture(tmp_path, "bad output"), recall_tolerance=1, document_tolerance=1)
    assert result["status"] == "review_required"
    assert result["gates"][-1]["passed"] is False


def test_frozen_hash_and_complete_coverage_required(tmp_path):
    dataset, reference, candidate, frozen = fixture(tmp_path, '{"entities":[]}')
    with pytest.raises(EvaluationError, match="hash"):
        compare(dataset, reference, candidate, "0" * 64)
    candidate.write_text("")
    with pytest.raises(EvaluationError, match="coverage"):
        compare(dataset, reference, candidate, frozen)


def test_cli_reports_regression_without_logging_source_and_is_idempotent(tmp_path, monkeypatch, capsys):
    from scripts.evaluation import check_parity
    dataset, reference, candidate, frozen = fixture(tmp_path, '{"entities":[]}')
    monkeypatch.setattr(check_parity, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "evaluation/results/parity"
    argv = ["--dataset", str(dataset), "--reference-predictions", str(reference),
            "--candidate-predictions", str(candidate), "--expected-dataset-sha256", frozen,
            "--output-dir", str(output)]
    assert main(argv) == 2
    before = (output / "parity.json").read_bytes()
    assert main(argv) == 2
    assert before == (output / "parity.json").read_bytes()
    assert "Ada" not in capsys.readouterr().out
    candidate.write_text(reference.read_text())
    assert main(argv) == 1
    assert before == (output / "parity.json").read_bytes()
