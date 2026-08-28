import json

import pytest

from pii_redactor.data.models import SourceEntity, SourceRecord
from pii_redactor.data.prepared import PreparationError, source_record_to_sample


def _record(text: str, entities: tuple[SourceEntity, ...]) -> SourceRecord:
    return SourceRecord(
        uid="source-one",
        domain="test",
        document_type="note",
        document_description="toy",
        text=text,
        entities=entities,
    )


def test_source_conversion_filters_and_orders_exact_spans() -> None:
    record = _record(
        "Email ada@example.net before Ada Lovelace in London.",
        (
            SourceEntity(entity="Ada Lovelace", types=("name",)),
            SourceEntity(entity="London", types=("city",)),
            SourceEntity(entity="ada@example.net", types=("email",)),
        ),
    )
    sample = source_record_to_sample(record, split="train")

    assert [entity.text for entity in sample.entities] == [
        "ada@example.net",
        "Ada Lovelace",
    ]
    assert [entity.start for entity in sample.entities] == [6, 29]
    assert json.loads(sample.assistant_content) == {
        "entities": [
            {"type": "EMAIL", "text": "ada@example.net"},
            {"type": "PERSON_NAME", "text": "Ada Lovelace"},
        ]
    }


def test_source_conversion_retains_every_repeated_occurrence() -> None:
    record = _record(
        "EMP-7 appeared, then EMP-7 appeared again.",
        (
            SourceEntity(entity="EMP-7", types=("employee_id",)),
            SourceEntity(entity="EMP-7", types=("employee_id",)),
        ),
    )
    sample = source_record_to_sample(record, split="valid")
    assert [entity.start for entity in sample.entities] == [0, 21]


def test_source_conversion_rejects_annotation_occurrence_mismatch() -> None:
    record = _record(
        "Ada appears once.",
        (
            SourceEntity(entity="Ada", types=("first_name",)),
            SourceEntity(entity="Ada", types=("first_name",)),
        ),
    )
    with pytest.raises(PreparationError, match="annotation/occurrence mismatch"):
        source_record_to_sample(record, split="train")


def test_source_conversion_rejects_overlapping_annotations() -> None:
    record = _record(
        "Ada Lovelace filed the report.",
        (
            SourceEntity(entity="Ada", types=("first_name",)),
            SourceEntity(entity="Ada Lovelace", types=("name",)),
        ),
    )
    with pytest.raises(PreparationError, match="overlap"):
        source_record_to_sample(record, split="train")
