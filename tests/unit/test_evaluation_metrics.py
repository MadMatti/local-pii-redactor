import json
from pathlib import Path

import pytest

from pii_redactor.data.prepared import sample_from_text_and_values
from pii_redactor.evaluation.metrics import (
    EvaluationError,
    align_predictions,
    evaluate_predictions,
    sha256_file,
)
from pii_redactor.evaluation.parsing import PredictionEntity
from pii_redactor.schema import EntityType


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sample(sample_id_variant: int, text: str, values: tuple) -> dict:
    return sample_from_text_and_values(
        split="test",
        template_family="test:unit",
        variant=sample_id_variant,
        synthetic_kind="unit",
        text=text,
        values=values,
    ).to_json_record()


def test_align_predictions_distinguishes_repeats_and_hallucinations() -> None:
    entity = PredictionEntity(EntityType.EMAIL, "a@example.net")
    aligned = align_predictions(
        "a@example.net then a@example.net", (entity, entity, entity)
    )
    assert [(item.start, item.end) for item in aligned] == [
        (0, 13),
        (19, 32),
        (None, None),
    ]


def test_perfect_predictions_score_one_and_track_negative_documents(tmp_path) -> None:
    positive = _sample(
        1,
        "Contact a@example.net twice: a@example.net.",
        ((EntityType.EMAIL, "a@example.net"),),
    )
    negative = _sample(2, "No private value is present.", ())
    dataset = tmp_path / "test.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(dataset, [positive, negative])
    _write_jsonl(
        predictions,
        [
            {
                "sample_id": positive["sample_id"],
                "raw_output": positive["messages"][2]["content"],
            },
            {
                "sample_id": negative["sample_id"],
                "raw_output": negative["messages"][2]["content"],
            },
        ],
    )

    result = evaluate_predictions(
        dataset, predictions, expected_dataset_sha256=sha256_file(dataset)
    )

    assert result.metrics["micro"]["f1"] == 1.0
    assert result.metrics["documents"]["complete_document_recall"] == 1.0
    assert result.metrics["documents"]["pii_free_false_positive_rate"] == 0.0


def test_wrong_type_and_hallucinated_text_are_fp_and_fn(tmp_path) -> None:
    sample = _sample(
        3,
        "Contact a@example.net.",
        ((EntityType.EMAIL, "a@example.net"),),
    )
    dataset = tmp_path / "test.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(dataset, [sample])
    raw = {
        "entities": [
            {"type": "USERNAME", "text": "a@example.net"},
            {"type": "EMAIL", "text": "missing@example.net"},
        ]
    }
    _write_jsonl(
        predictions,
        [{"sample_id": sample["sample_id"], "raw_output": json.dumps(raw)}],
    )

    result = evaluate_predictions(dataset, predictions)

    assert result.metrics["micro"] == {
        "tp": 0,
        "fp": 2,
        "fn": 1,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    assert result.metrics["validity"]["hallucinated_substrings"] == 1


def test_relaxed_overlap_is_diagnostic_and_does_not_change_exact_score(tmp_path) -> None:
    sample = _sample(
        5,
        "Contact Ada Lovelace today.",
        ((EntityType.PERSON_NAME, "Ada Lovelace"),),
    )
    dataset = tmp_path / "test.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(dataset, [sample])
    _write_jsonl(
        predictions,
        [
            {
                "sample_id": sample["sample_id"],
                "raw_output": json.dumps(
                    {"entities": [{"type": "PERSON_NAME", "text": "Ada"}]}
                ),
            }
        ],
    )

    result = evaluate_predictions(dataset, predictions)

    assert result.metrics["micro"]["f1"] == 0.0
    assert result.metrics["relaxed_overlap_micro"]["f1"] == 1.0


def test_evaluation_rejects_incomplete_predictions_and_hash_mismatch(tmp_path) -> None:
    sample = _sample(4, "Nothing private.", ())
    dataset = tmp_path / "test.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(dataset, [sample])
    _write_jsonl(predictions, [])
    with pytest.raises(EvaluationError, match="coverage mismatch"):
        evaluate_predictions(dataset, predictions)
    with pytest.raises(EvaluationError, match="hash mismatch"):
        evaluate_predictions(dataset, predictions, expected_dataset_sha256="0" * 64)
