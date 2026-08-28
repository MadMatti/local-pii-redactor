from __future__ import annotations

import pytest

from pii_redactor.data.constants import (
    EXCLUDED_SOURCE_LABELS,
    KNOWN_SOURCE_LABELS,
    SOURCE_LABEL_MAPPING,
)
from pii_redactor.data.models import SourceRecord
from pii_redactor.data.source import (
    EntityParseError,
    classify_source_types,
    count_exact_occurrences,
    normalize_for_duplicate,
    parse_entities,
)
from pii_redactor.schema import EntityType


def test_parse_json_entities() -> None:
    parsed = parse_entities(
        '[{"entity":"Renée O’Neil","types":["name","first_name"]}]'
    )

    assert parsed[0].entity == "Renée O’Neil"
    assert parsed[0].types == ("name", "first_name")


def test_parse_python_literal_with_apostrophe() -> None:
    parsed = parse_entities(
        '[{"entity": "O\'Neil", "types": ["last_name"]}]'
    )

    assert parsed[0].entity == "O'Neil"


def test_parse_materialized_list() -> None:
    parsed = parse_entities([{"entity": "alice@example.com", "types": ["email"]}])

    assert parsed[0].types == ("email",)


@pytest.mark.parametrize(
    "value",
    [
        "__import__('os').system('touch should-not-exist')",
        "{}",
        "",
        None,
        [{"entity": "x"}],
        [{"entity": "x", "types": "email"}],
        [{"entity": "", "types": ["email"]}],
        [{"entity": "x", "types": []}],
        [{"entity": "x", "types": [""]}],
        [{"entity": "x", "types": ["email"], "extra": True}],
    ],
)
def test_reject_invalid_entities(value: object) -> None:
    with pytest.raises(EntityParseError):
        parse_entities(value)


def test_source_record_ignores_extra_dataset_columns() -> None:
    record = SourceRecord.from_mapping(
        {
            "uid": "sample",
            "domain": "test",
            "document_type": "note",
            "document_description": "A note",
            "text": "Email alice@example.com",
            "entities": "[{'entity': 'alice@example.com', 'types': ['email']}]",
            "future_column": "allowed",
        }
    )

    assert record.uid == "sample"
    assert record.entities[0].entity == "alice@example.com"


def test_source_label_policy_classifies_all_42_labels() -> None:
    assert len(KNOWN_SOURCE_LABELS) == 42
    assert set(SOURCE_LABEL_MAPPING).isdisjoint(EXCLUDED_SOURCE_LABELS)
    assert set(SOURCE_LABEL_MAPPING) | EXCLUDED_SOURCE_LABELS == KNOWN_SOURCE_LABELS


def test_classification_deduplicates_same_canonical_label() -> None:
    classification = classify_source_types(("first_name", "name"))

    assert classification.canonical_types == (EntityType.PERSON_NAME,)
    assert not classification.is_ambiguous


def test_classification_marks_multiple_canonical_labels_ambiguous() -> None:
    classification = classify_source_types(("name", "email"))

    assert classification.is_ambiguous
    assert classification.canonical_types == (
        EntityType.PERSON_NAME,
        EntityType.EMAIL,
    )


def test_classification_reports_unknown_labels() -> None:
    classification = classify_source_types(("future_secret",))

    assert classification.unknown_labels == ("future_secret",)
    assert not classification.canonical_types


def test_count_exact_occurrences() -> None:
    assert count_exact_occurrences("John called John and John replied", "John") == 3
    assert count_exact_occurrences("Nothing here", "John") == 0
    assert count_exact_occurrences("aaaa", "aa") == 2


def test_duplicate_normalization_is_diagnostic_only() -> None:
    original = "  Straße\n\tNAME  "

    assert normalize_for_duplicate(original) == "strasse name"
    assert original == "  Straße\n\tNAME  "
