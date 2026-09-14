import hashlib
import json
from collections import Counter

import pytest

from pii_redactor.data.prepared import sample_from_text_and_values
from pii_redactor.evaluation.error_analysis import analyze_errors, write_error_analysis
from pii_redactor.evaluation.metrics import EvaluationError, evaluate_predictions, sha256_file
from pii_redactor.schema import EntityType
from scripts.evaluation import analyze_errors as command


def _sample(variant, text, values=()):
    return sample_from_text_and_values(
        split="test", template_family="unit:error-analysis", variant=variant,
        synthetic_kind="unit", text=text, values=values,
    ).to_json_record()


def _inputs(tmp_path, cases):
    dataset = tmp_path / "test.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row, _ in cases))
    predictions.write_text("".join(
        json.dumps({"sample_id": row["sample_id"], "raw_output": raw}) + "\n"
        for row, raw in cases
    ))
    return dataset, predictions


def _output(*entities):
    return json.dumps({"entities": [{"type": label, "text": text} for label, text in entities]})


def test_causes_reconcile_with_exact_evaluator_including_repeats_and_invalid_schema(tmp_path):
    cases = [
        (_sample(1, "Ada writes.", ((EntityType.PERSON_NAME, "Ada"),)), _output(("USERNAME", "Ada"))),
        (_sample(2, "Ada Lovelace writes.", ((EntityType.PERSON_NAME, "Ada Lovelace"),)), _output(("PERSON_NAME", "Ada"))),
        (_sample(3, "a@example.net then a@example.net", ((EntityType.EMAIL, "a@example.net"),)), _output(("EMAIL", "a@example.net"))),
        (_sample(4, "a@example.net", ((EntityType.EMAIL, "a@example.net"),)), _output(("EMAIL", "a@example.net"), ("EMAIL", "a@example.net"), ("EMAIL", "absent@example.net"))),
        (_sample(5, "A neutral word."), _output(("PERSON_NAME", "neutral"))),
        (_sample(6, "Ada writes.", ((EntityType.PERSON_NAME, "Ada"),)), "broken JSON"),
        (_sample(7, "Nothing private."), "broken JSON"),
        (_sample(8, "Ada writes.", ((EntityType.PERSON_NAME, "Ada"),)), _output()),
    ]
    dataset, predictions = _inputs(tmp_path, cases)
    result = analyze_errors(dataset, predictions, expected_dataset_sha256=sha256_file(dataset))
    counts = {(g["kind"], g["suspected_cause"]): g["errors"] for g in result.statistics["groups"]}
    assert counts == {
        ("FN", "wrong_label"): 1, ("FP", "wrong_label"): 1,
        ("FN", "boundary_mismatch"): 1, ("FP", "boundary_mismatch"): 1,
        ("FN", "repeated_occurrence_omission"): 1,
        ("FP", "overpredicted_occurrence"): 1, ("FP", "hallucinated_text"): 1,
        ("FP", "spurious_entity"): 1, ("FN", "invalid_schema"): 1,
        ("FN", "missed_entity"): 1, ("SCHEMA", "invalid_schema"): 2,
    }
    metrics = evaluate_predictions(dataset, predictions).metrics
    assert result.statistics["totals"]["fn"] == metrics["micro"]["fn"] == 5
    assert result.statistics["totals"]["fp"] == metrics["micro"]["fp"] == 5
    assert metrics["validity"]["hallucinated_substrings"] == 2
    assert len(result.review_samples) == 8


def test_all_gold_fn_and_no_fp_when_one_predicted_type_invalidates_whole_schema(tmp_path):
    sample = _sample(1, "Ada emails a@example.net", ((EntityType.PERSON_NAME, "Ada"), (EntityType.EMAIL, "a@example.net")))
    dataset, predictions = _inputs(tmp_path, [(sample, _output(("PERSON_NAME", "Ada"), ("UNKNOWN", "a@example.net")))])
    result = analyze_errors(dataset, predictions, expected_dataset_sha256=sha256_file(dataset))
    assert result.statistics["totals"]["fn"] == 2
    assert result.statistics["totals"]["fp"] == 0
    assert {error["suspected_cause"] for error in result.review_samples[0]["errors"]} == {"invalid_schema"}


def test_deterministic_bounded_sampling_and_artifact_integrity(tmp_path):
    cases = [
        (_sample(i, f"Ada writes item {i}.", ((EntityType.PERSON_NAME, "Ada"),)), _output(("USERNAME", "Ada")))
        for i in range(20)
    ]
    dataset, predictions = _inputs(tmp_path, cases)
    kwargs = {"expected_dataset_sha256": sha256_file(dataset), "samples_per_group": 4, "max_review_records": 5}
    first = analyze_errors(dataset, predictions, **kwargs)
    second = analyze_errors(dataset, predictions, **kwargs)
    assert first == second
    assert len(first.review_samples) == 5
    reasons = Counter(group for row in first.review_samples for group in row["selection_reasons"])
    assert all(count <= 4 for count in reasons.values())
    assert len({row["sample_id"] for row in first.review_samples}) == 5
    output = tmp_path / "analysis"
    manifest_path = write_error_analysis(first, output)
    snapshots = {path.name: path.read_bytes() for path in output.iterdir()}
    write_error_analysis(second, output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == snapshots
    manifest = json.loads(manifest_path.read_text())
    for name, metadata in manifest["artifacts"].items():
        assert metadata["sha256"] == hashlib.sha256(snapshots[name]).hexdigest()
        assert metadata["bytes"] == len(snapshots[name])
    assert b"Ada" not in snapshots["statistics.json"]
    assert b"Ada" not in snapshots["summary.md"]
    assert b"Ada" not in snapshots["inspection_manifest.json"]
    assert b"Ada" in snapshots["review_samples.jsonl"]
    third = analyze_errors(dataset, predictions, **{**kwargs, "seed": 17})
    with pytest.raises(EvaluationError, match="refusing to overwrite"):
        write_error_analysis(third, output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == snapshots


def test_oracle_has_no_error_samples_and_detects_coverage_and_hash_failures(tmp_path):
    sample = _sample(1, "A neutral sentence.")
    dataset, predictions = _inputs(tmp_path, [(sample, _output())])
    result = analyze_errors(dataset, predictions, expected_dataset_sha256=sha256_file(dataset))
    assert result.review_samples == ()
    assert result.statistics["groups"] == []
    with pytest.raises(EvaluationError, match="hash mismatch"):
        analyze_errors(dataset, predictions, expected_dataset_sha256="0" * 64)
    predictions.write_text("")
    with pytest.raises(EvaluationError, match="coverage mismatch"):
        analyze_errors(dataset, predictions, expected_dataset_sha256=sha256_file(dataset))
    predictions.write_text(json.dumps({"sample_id": sample["sample_id"], "raw_output": _output()}) + "\n")
    predictions.write_text(predictions.read_text() * 2)
    with pytest.raises(EvaluationError, match="duplicate prediction"):
        analyze_errors(dataset, predictions, expected_dataset_sha256=sha256_file(dataset))


def test_cli_keeps_text_and_entity_values_out_of_success_and_failure_logs(tmp_path, monkeypatch, capsys):
    secret = "PrivateValueNeverLog827"
    sample = _sample(1, secret, ((EntityType.PERSON_NAME, secret),))
    sample["sample_id"] = secret
    dataset, predictions = _inputs(tmp_path, [(sample, _output())])
    monkeypatch.setattr(command, "PROJECT_ROOT", tmp_path)
    argv = ["--dataset", str(dataset), "--predictions", str(predictions), "--expected-dataset-sha256", sha256_file(dataset)]
    assert command.main(argv) == 0
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert "False negatives: 1" in captured.out
    assert len(list((tmp_path / "evaluation" / "results").glob("*/inspection_manifest.json"))) == 1
    predictions.write_text(predictions.read_text() * 2)
    assert command.main(argv) == 1
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err


def test_writer_refuses_symlinks_and_unrelated_files_without_overwrite(tmp_path):
    dataset, predictions = _inputs(tmp_path, [(_sample(1, "Neutral."), _output())])
    result = analyze_errors(dataset, predictions, expected_dataset_sha256=sha256_file(dataset))
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(EvaluationError, match="symlink"):
        write_error_analysis(result, linked)
    (target / "unrelated.txt").write_text("preserve")
    with pytest.raises(EvaluationError, match="unexpected entries"):
        write_error_analysis(result, target)
    assert (target / "unrelated.txt").read_text() == "preserve"
