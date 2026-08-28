"""Revision-pinned download and local snapshot management."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

from datasets import DatasetDict, load_dataset, load_from_disk
from huggingface_hub import HfApi

from .constants import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATASET_ID,
    DEFAULT_LOCK_FILE,
    DEFAULT_REQUESTED_REVISION,
    DEFAULT_SNAPSHOT_ROOT,
    EXPECTED_SPLIT_COUNTS,
    PROJECT_ROOT,
    REQUIRED_COLUMNS,
)


class DownloadError(RuntimeError):
    """Raised when acquisition cannot produce a trustworthy local snapshot."""


@dataclass(frozen=True)
class DownloadConfig:
    dataset_id: str = DEFAULT_DATASET_ID
    requested_revision: str = DEFAULT_REQUESTED_REVISION
    cache_dir: Path = DEFAULT_CACHE_DIR
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT
    lock_file: Path = DEFAULT_LOCK_FILE
    offline: bool = False
    refresh_revision: bool = False


@dataclass(frozen=True)
class DownloadOutcome:
    resolved_revision: str
    snapshot_path: Path
    lock_file: Path
    split_counts: dict[str, int]
    review_blockers: tuple[dict[str, Any], ...]
    reused_snapshot: bool

    @property
    def exit_code(self) -> int:
        return 2 if self.review_blockers else 0


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_snapshot_path(lock: dict[str, Any]) -> Path:
    snapshot = Path(lock["snapshot_path"])
    if snapshot.is_absolute():
        return snapshot
    return PROJECT_ROOT / snapshot


def read_source_lock(lock_file: Path = DEFAULT_LOCK_FILE) -> dict[str, Any]:
    try:
        with lock_file.open(encoding="utf-8") as handle:
            lock = json.load(handle)
    except FileNotFoundError as exc:
        raise DownloadError(f"source lock does not exist: {lock_file}") from exc
    except json.JSONDecodeError as exc:
        raise DownloadError(f"source lock is invalid JSON: {lock_file}") from exc
    if not isinstance(lock, dict) or "resolved_revision" not in lock:
        raise DownloadError(f"source lock has an invalid schema: {lock_file}")
    return lock


def _validate_dataset_structure(
    dataset: DatasetDict,
) -> tuple[
    dict[str, int],
    dict[str, list[str]],
    dict[str, list[str]],
    list[dict[str, Any]],
]:
    if not isinstance(dataset, DatasetDict):
        raise DownloadError("load_dataset did not return a DatasetDict")

    blockers: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    columns_by_split: dict[str, list[str]] = {}
    extra_columns_by_split: dict[str, list[str]] = {}

    missing_splits = sorted(set(EXPECTED_SPLIT_COUNTS) - set(dataset))
    if missing_splits:
        raise DownloadError(f"dataset is missing required splits: {missing_splits}")

    for split in EXPECTED_SPLIT_COUNTS:
        split_dataset = dataset[split]
        split_counts[split] = len(split_dataset)
        columns = sorted(split_dataset.column_names)
        columns_by_split[split] = columns
        missing_columns = sorted(REQUIRED_COLUMNS - set(columns))
        if missing_columns:
            raise DownloadError(
                f"split {split!r} is missing required columns: {missing_columns}"
            )
        extra_columns_by_split[split] = sorted(set(columns) - REQUIRED_COLUMNS)
        expected_count = EXPECTED_SPLIT_COUNTS[split]
        if split_counts[split] != expected_count:
            blockers.append(
                {
                    "type": "split_count_mismatch",
                    "split": split,
                    "expected": expected_count,
                    "actual": split_counts[split],
                }
            )

    return split_counts, columns_by_split, extra_columns_by_split, blockers


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_save_dataset(dataset: DatasetDict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=".dataset-", dir=destination.parent)
    )
    temporary_path.rmdir()
    try:
        dataset.save_to_disk(str(temporary_path))
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise


def _dataset_license(info: Any) -> str | None:
    card_data = getattr(info, "card_data", None)
    if card_data is None:
        return None
    if isinstance(card_data, dict):
        return card_data.get("license")
    return getattr(card_data, "license", None)


def _outcome_from_local_lock(
    config: DownloadConfig,
    lock: dict[str, Any],
    load_from_disk_fn: Callable[[str], Any],
) -> DownloadOutcome:
    if lock.get("dataset_id") != config.dataset_id:
        raise DownloadError(
            "existing source lock belongs to a different dataset: "
            f"{lock.get('dataset_id')!r}"
        )
    snapshot_path = resolve_snapshot_path(lock)
    if not snapshot_path.exists():
        raise DownloadError(f"locked snapshot does not exist: {snapshot_path}")
    dataset = load_from_disk_fn(str(snapshot_path))
    split_counts, _, _, blockers = _validate_dataset_structure(dataset)
    return DownloadOutcome(
        resolved_revision=str(lock["resolved_revision"]),
        snapshot_path=snapshot_path,
        lock_file=config.lock_file,
        split_counts=split_counts,
        review_blockers=tuple(blockers),
        reused_snapshot=True,
    )


def download_source_dataset(
    config: DownloadConfig,
    *,
    api: Any | None = None,
    load_dataset_fn: Callable[..., Any] = load_dataset,
    load_from_disk_fn: Callable[[str], Any] = load_from_disk,
) -> DownloadOutcome:
    """Resolve, download, validate, snapshot, and lock the source dataset."""

    existing_lock = None
    if config.lock_file.exists():
        existing_lock = read_source_lock(config.lock_file)

    if config.offline:
        if existing_lock is None:
            raise DownloadError("offline mode requires an existing source lock")
        return _outcome_from_local_lock(config, existing_lock, load_from_disk_fn)

    api = api or HfApi()
    info = api.dataset_info(
        repo_id=config.dataset_id,
        revision=config.requested_revision,
    )
    resolved_revision = getattr(info, "sha", None)
    if not resolved_revision:
        raise DownloadError("Hugging Face did not return an immutable revision SHA")

    if existing_lock is not None:
        locked_revision = existing_lock.get("resolved_revision")
        if locked_revision == resolved_revision:
            return _outcome_from_local_lock(config, existing_lock, load_from_disk_fn)
        if not config.refresh_revision:
            raise DownloadError(
                "the remote revision differs from the existing source lock "
                f"({locked_revision} -> {resolved_revision}); rerun with "
                "--refresh-revision to create and select a new snapshot"
            )

    dataset = load_dataset_fn(
        config.dataset_id,
        revision=resolved_revision,
        cache_dir=str(config.cache_dir),
    )
    split_counts, columns, extras, blockers = _validate_dataset_structure(dataset)

    snapshot_path = config.snapshot_root / resolved_revision / "dataset"
    reused_snapshot = snapshot_path.exists()
    if reused_snapshot:
        saved_dataset = load_from_disk_fn(str(snapshot_path))
        saved_counts, saved_columns, _, saved_blockers = _validate_dataset_structure(
            saved_dataset
        )
        if saved_counts != split_counts or saved_columns != columns:
            raise DownloadError(
                f"existing snapshot does not match downloaded metadata: {snapshot_path}"
            )
        blockers.extend(saved_blockers)
    else:
        _atomic_save_dataset(dataset, snapshot_path)

    manifest = {
        "schema_version": 1,
        "dataset_id": config.dataset_id,
        "requested_revision": config.requested_revision,
        "resolved_revision": resolved_revision,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "license": _dataset_license(info),
        "snapshot_path": _display_path(snapshot_path),
        "cache_dir": _display_path(config.cache_dir),
        "split_counts": split_counts,
        "expected_split_counts": EXPECTED_SPLIT_COUNTS,
        "columns_by_split": columns,
        "required_columns": sorted(REQUIRED_COLUMNS),
        "extra_columns_by_split": extras,
        "review_blockers": blockers,
        "library_versions": {
            "datasets": _package_version("datasets"),
            "huggingface_hub": _package_version("huggingface-hub"),
            "python": ".".join(map(str, os.sys.version_info[:3])),
        },
    }
    _atomic_write_json(config.lock_file, manifest)

    return DownloadOutcome(
        resolved_revision=resolved_revision,
        snapshot_path=snapshot_path,
        lock_file=config.lock_file,
        split_counts=split_counts,
        review_blockers=tuple(blockers),
        reused_snapshot=reused_snapshot,
    )
