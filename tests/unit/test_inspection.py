from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from pii_redactor.data.inspection import (
    InspectionConfig,
    _length_summary,
    _prompt_overhead_tokens,
    inspect_source_dataset,
)


class FakeTokenizer:
    def __call__(self, texts: list[str], **_: Any) -> dict[str, list[list[int]]]:
        return {"input_ids": [self.encode(text) for text in texts]}

    def encode(self, text: str, **_: Any) -> list[int]:
        return list(range(len(text.split())))

    def apply_chat_template(self, messages: list[dict[str, str]], **_: Any) -> list[int]:
        return list(range(10 + sum(len(message["content"].split()) for message in messages)))


class MappingChatTemplateTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages: list[dict[str, str]], **_: Any) -> dict[str, list[int]]:
        return {"input_ids": list(range(17)), "attention_mask": [1] * 17}


def source_row(
    uid: str,
    text: str,
    entities: str,
    *,
    domain: str = "test",
    document_type: str = "Note",
) -> dict[str, list[str]]:
    return {
        "uid": [uid],
        "domain": [domain],
        "document_type": [document_type],
        "document_description": ["A deterministic test note"],
        "entities": [entities],
        "text": [text],
    }


def concatenate_rows(*rows: dict[str, list[str]]) -> Dataset:
    columns = rows[0].keys()
    return Dataset.from_dict(
        {column: [value for row in rows for value in row[column]] for column in columns}
    )


def make_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    duplicate_text = "Contact Renée at renee@example.com."
    train = concatenate_rows(
        source_row(
            "train-1",
            duplicate_text,
            "[{'entity': 'renee@example.com', 'types': ['email']}, "
            "{'entity': 'Renée', 'types': ['first_name']}]",
        ),
        source_row(
            "train-2",
            "John called John.",
            "[{'entity': 'John', 'types': ['name']}]",
        ),
        source_row(
            "train-3",
            "No matching value.",
            "[{'entity': 'Missing', 'types': ['name']}]",
        ),
    )
    validation = concatenate_rows(
        source_row(
            "valid-1",
            duplicate_text,
            '[{"entity":"Renée","types":["name"]}]',
        )
    )
    test = concatenate_rows(
        source_row(
            "test-1",
            "The conference opens on 4 April 2027.",
            "[{'entity': '4 April 2027', 'types': ['date']}]",
        )
    )
    dataset = DatasetDict({"train": train, "validation": validation, "test": test})
    snapshot = tmp_path / "snapshot"
    dataset.save_to_disk(str(snapshot))
    lock = tmp_path / "source-lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "example/source",
                "resolved_revision": "abc123",
                "snapshot_path": str(snapshot),
                "split_counts": {"train": 3, "validation": 1, "test": 1},
                "review_blockers": [],
            }
        ),
        encoding="utf-8",
    )
    return snapshot, lock


def test_length_summary_uses_nearest_observed_percentiles() -> None:
    summary = _length_summary([1, 2, 3, 4, 10])

    assert summary["p50"] == 3
    assert summary["p75"] == 4
    assert summary["max"] == 10
    assert summary["mean"] == 4.0


def test_prompt_overhead_counts_input_ids_not_mapping_fields() -> None:
    assert _prompt_overhead_tokens(MappingChatTemplateTokenizer()) == 17


def test_inspection_writes_complete_review_bundle(tmp_path: Path) -> None:
    _, lock = make_snapshot(tmp_path)
    output = tmp_path / "inspection"

    outcome = inspect_source_dataset(
        InspectionConfig(
            lock_file=lock,
            tokenizer_path=tmp_path / "unused",
            output_dir=output,
            seed=42,
            tokenizer_batch_size=2,
        ),
        tokenizer=FakeTokenizer(),
    )

    expected_files = {
        "summary.md",
        "statistics.json",
        "label_inventory.csv",
        "quality_issues.jsonl",
        "split_overlap.json",
        "review_samples.jsonl",
        "inspection_manifest.json",
    }
    assert {path.name for path in output.iterdir()} == expected_files
    assert outcome.record_count == 5
    assert outcome.exit_code == 2
    assert outcome.blocker_count >= 1

    statistics = json.loads((output / "statistics.json").read_text(encoding="utf-8"))
    assert statistics["overall"]["entity_occurrences"] == {
        "zero": 1,
        "unique": 4,
        "multiple": 1,
    }
    assert statistics["overall"]["negative_after_filtering"]["count"] == 1
    assert statistics["overall"]["canonical_label_frequency"]["PERSON_NAME"] == 4
    assert statistics["overall"]["source_order_violation_records"] == 1
    assert statistics["outliers"]["highest_entity_counts"][0]["entity_count"] == 2

    overlap = json.loads((output / "split_overlap.json").read_text(encoding="utf-8"))
    assert overlap["exact_text_duplicates"]["cross_split_group_count"] == 1


def test_review_selection_is_deterministic(tmp_path: Path) -> None:
    _, lock = make_snapshot(tmp_path)
    first = tmp_path / "inspection-first"
    second = tmp_path / "inspection-second"
    base = dict(lock_file=lock, tokenizer_path=tmp_path / "unused", seed=42)

    inspect_source_dataset(
        InspectionConfig(output_dir=first, **base), tokenizer=FakeTokenizer()
    )
    inspect_source_dataset(
        InspectionConfig(output_dir=second, **base), tokenizer=FakeTokenizer()
    )

    assert (first / "review_samples.jsonl").read_bytes() == (
        second / "review_samples.jsonl"
    ).read_bytes()
    assert (first / "statistics.json").read_bytes() == (
        second / "statistics.json"
    ).read_bytes()
