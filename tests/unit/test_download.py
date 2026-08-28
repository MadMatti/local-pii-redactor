from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset, DatasetDict

from pii_redactor.data.download import (
    DownloadConfig,
    DownloadError,
    download_source_dataset,
)


def make_dataset() -> DatasetDict:
    row = {
        "uid": ["uid-1"],
        "domain": ["test"],
        "document_type": ["Note"],
        "document_description": ["A test note"],
        "entities": ["[{'entity': 'Alice', 'types': ['first_name']}]"],
        "text": ["Alice wrote the note."],
        "extra": ["recorded"],
    }
    return DatasetDict(
        {
            "train": Dataset.from_dict(row),
            "validation": Dataset.from_dict(row),
            "test": Dataset.from_dict(row),
        }
    )


@dataclass
class FakeInfo:
    sha: str
    card_data: dict[str, str]


class FakeApi:
    def __init__(self, sha: str) -> None:
        self.sha = sha

    def dataset_info(self, **_: Any) -> FakeInfo:
        return FakeInfo(sha=self.sha, card_data={"license": "apache-2.0"})


def make_config(tmp_path: Path, **overrides: Any) -> DownloadConfig:
    values = {
        "dataset_id": "example/source",
        "requested_revision": "main",
        "cache_dir": tmp_path / "cache",
        "snapshot_root": tmp_path / "snapshots",
        "lock_file": tmp_path / "intermediate" / "source-lock.json",
    }
    values.update(overrides)
    return DownloadConfig(**values)


def test_download_pins_revision_and_writes_lock(tmp_path: Path, capsys: Any) -> None:
    dataset = make_dataset()
    calls: list[tuple[str, str, str]] = []

    def load_fn(dataset_id: str, *, revision: str, cache_dir: str) -> DatasetDict:
        calls.append((dataset_id, revision, cache_dir))
        return dataset

    config = make_config(tmp_path)
    outcome = download_source_dataset(
        config,
        api=FakeApi("abc123"),
        load_dataset_fn=load_fn,
    )

    lock = json.loads(config.lock_file.read_text(encoding="utf-8"))
    assert calls == [("example/source", "abc123", str(config.cache_dir))]
    assert outcome.resolved_revision == "abc123"
    assert outcome.snapshot_path.exists()
    assert lock["resolved_revision"] == "abc123"
    assert lock["extra_columns_by_split"]["train"] == ["extra"]
    assert len(outcome.review_blockers) == 3  # toy split sizes differ from source
    assert "Alice" not in capsys.readouterr().out


def test_download_is_idempotent_for_same_revision(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    first = download_source_dataset(
        config,
        api=FakeApi("abc123"),
        load_dataset_fn=lambda *args, **kwargs: make_dataset(),
    )

    def fail_if_called(*args: Any, **kwargs: Any) -> DatasetDict:
        raise AssertionError("remote dataset should not be loaded again")

    second = download_source_dataset(
        config,
        api=FakeApi("abc123"),
        load_dataset_fn=fail_if_called,
    )

    assert not first.reused_snapshot
    assert second.reused_snapshot
    assert second.snapshot_path == first.snapshot_path


def test_download_refuses_changed_revision_without_refresh(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    download_source_dataset(
        config,
        api=FakeApi("abc123"),
        load_dataset_fn=lambda *args, **kwargs: make_dataset(),
    )

    with pytest.raises(DownloadError, match="--refresh-revision"):
        download_source_dataset(
            config,
            api=FakeApi("def456"),
            load_dataset_fn=lambda *args, **kwargs: make_dataset(),
        )


def test_offline_requires_and_reuses_lock(tmp_path: Path) -> None:
    online = make_config(tmp_path)
    download_source_dataset(
        online,
        api=FakeApi("abc123"),
        load_dataset_fn=lambda *args, **kwargs: make_dataset(),
    )

    offline = make_config(tmp_path, offline=True)
    outcome = download_source_dataset(offline)

    assert outcome.reused_snapshot
    assert outcome.resolved_revision == "abc123"


def test_offline_without_lock_fails(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="existing source lock"):
        download_source_dataset(make_config(tmp_path, offline=True))
