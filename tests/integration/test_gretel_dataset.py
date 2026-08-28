from __future__ import annotations

import os

import pytest
from datasets import load_dataset
from huggingface_hub import HfApi

from pii_redactor.data.constants import DEFAULT_DATASET_ID, REQUIRED_COLUMNS
from pii_redactor.data.models import SourceRecord


@pytest.mark.network
def test_current_gretel_schema_and_parser() -> None:
    if os.environ.get("RUN_NETWORK_TESTS") != "1":
        pytest.skip("set RUN_NETWORK_TESTS=1 to query the real Gretel dataset")

    info = HfApi().dataset_info(DEFAULT_DATASET_ID, revision="main")
    dataset = load_dataset(
        DEFAULT_DATASET_ID,
        revision=info.sha,
        split="train[:1]",
    )

    assert REQUIRED_COLUMNS <= set(dataset.column_names)
    record = SourceRecord.from_mapping(dataset[0])
    assert record.uid
    assert record.text
    assert record.entities
