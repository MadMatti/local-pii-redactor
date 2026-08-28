"""Reproducible construction of MLX chat datasets and calibration text."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from pii_redactor.schema import EntityType

from .constants import (
    DATASET_BUILD_SEED,
    DATASET_BUILD_VERSION,
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_LOCK_FILE,
    DEFAULT_PROCESSED_DIR,
    EXCLUDED_SOURCE_LABELS,
    MAIN_STAGE_COUNTS,
    POLICY_VERSION,
    PROMPT_VERSION,
    SMOKE_STAGE_COUNTS,
    TARGET_NEGATIVE_RATIO,
    V0_STAGE_COUNTS,
    SOURCE_LABEL_MAPPING,
)
from .models import SourceRecord
from .prepared import PreparedSample, source_record_to_sample
from .synthetic import (
    generate_calibration_texts,
    generate_negative_samples,
    generate_positive_samples,
)


class DatasetBuildError(RuntimeError):
    """Raised when a reproducible build cannot safely continue."""


@dataclass(frozen=True)
class BuildResult:
    manifest_path: Path
    rejected_path: Path
    counts: dict[str, dict[str, int]]
    rejected_count: int


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for record in records:
            handle.write(_json_bytes(record))
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_shuffle(samples: Sequence[PreparedSample], key: str) -> list[PreparedSample]:
    shuffled = list(samples)
    random.Random(f"{DATASET_BUILD_SEED}:{key}").shuffle(shuffled)
    return shuffled


def _cover_labels(
    candidates: list[PreparedSample], *, minimum: int, key: str
) -> tuple[list[PreparedSample], list[PreparedSample]]:
    """Select a deterministic prefix that meets per-label occurrence coverage."""

    remaining = _stable_shuffle(candidates, f"coverage:{key}")
    selected: list[PreparedSample] = []
    counts: Counter[EntityType] = Counter()
    unselected: list[PreparedSample] = []
    for sample in remaining:
        useful = any(counts[label] < minimum for label in sample.entity_types)
        if useful:
            selected.append(sample)
            counts.update(sample.entity_types)
        else:
            unselected.append(sample)
    missing = [label.value for label in EntityType if counts[label] < minimum]
    if missing:
        raise DatasetBuildError(f"cannot meet label coverage for {key}: {missing}")
    return selected, unselected


def _select_stage(
    candidates: Sequence[PreparedSample],
    *,
    total: int,
    minimum_per_label: int,
    key: str,
) -> list[PreparedSample]:
    negative_target = round(total * TARGET_NEGATIVE_RATIO)
    negatives = _stable_shuffle(
        [sample for sample in candidates if sample.is_negative], f"{key}:negative"
    )
    positives = [sample for sample in candidates if not sample.is_negative]
    if len(negatives) < negative_target:
        raise DatasetBuildError(f"not enough negative candidates for {key}")

    covered, remaining = _cover_labels(
        positives, minimum=minimum_per_label, key=key
    )
    positive_target = total - negative_target
    if len(covered) > positive_target:
        raise DatasetBuildError(f"label coverage exceeds positive capacity for {key}")
    fill = _stable_shuffle(remaining, f"{key}:positive-fill")
    chosen = covered + fill[: positive_target - len(covered)]
    if len(chosen) != positive_target:
        raise DatasetBuildError(f"not enough positive candidates for {key}")
    return _stable_shuffle(
        chosen + negatives[:negative_target], f"{key}:final-order"
    )


def _source_samples(
    snapshot_path: Path,
) -> tuple[dict[str, list[PreparedSample]], list[dict[str, Any]]]:
    from datasets import load_from_disk

    dataset = load_from_disk(str(snapshot_path))
    split_names = {"train": "train", "validation": "valid", "test": "test"}
    output: dict[str, list[PreparedSample]] = {name: [] for name in split_names.values()}
    rejected: list[dict[str, Any]] = []
    for source_split, output_split in split_names.items():
        for row_index, row in enumerate(dataset[source_split]):
            try:
                record = SourceRecord.from_mapping(row)
                output[output_split].append(
                    source_record_to_sample(record, split=output_split)
                )
            except Exception as exc:
                rejected.append(
                    {
                        "source": "gretel",
                        "source_split": source_split,
                        "row_index": row_index,
                        "source_uid": row.get("uid"),
                        "error_type": type(exc).__name__,
                        "reason": str(exc),
                    }
                )
    return output, rejected


def _build_main_split(
    source: Sequence[PreparedSample], *, split: str, total: int
) -> list[PreparedSample]:
    negative_target = round(total * TARGET_NEGATIVE_RATIO)
    source_negatives = [sample for sample in source if sample.is_negative]
    source_positives = [sample for sample in source if not sample.is_negative]
    retained_source_negatives = _stable_shuffle(
        source_negatives, f"main:{split}:source-negative"
    )[:negative_target]
    synthetic_negative_count = negative_target - len(retained_source_negatives)

    synthetic_positive_count = max(len(EntityType) * 3, total // 10)
    positive_target = total - negative_target
    synthetic_positive_count = min(synthetic_positive_count, positive_target)
    source_positive_count = positive_target - synthetic_positive_count
    retained_source_positives = _stable_shuffle(
        source_positives, f"main:{split}:source-positive"
    )[:source_positive_count]
    if len(retained_source_positives) != source_positive_count:
        raise DatasetBuildError(f"not enough source positives for main {split}")

    synthetic_negatives = generate_negative_samples(
        split=split, count=synthetic_negative_count, seed=DATASET_BUILD_SEED
    )
    synthetic_positives = generate_positive_samples(
        split=split, count=synthetic_positive_count, seed=DATASET_BUILD_SEED
    )
    combined = (
        retained_source_negatives
        + retained_source_positives
        + synthetic_negatives
        + synthetic_positives
    )
    return _select_stage(
        combined,
        total=total,
        minimum_per_label=20 if split == "train" else 5,
        key=f"main:{split}",
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _sample_stats(samples: Sequence[PreparedSample]) -> dict[str, Any]:
    labels = Counter(
        entity.type.value for sample in samples for entity in sample.entities
    )
    kinds = Counter(sample.synthetic_kind for sample in samples if sample.synthetic_kind)
    return {
        "records": len(samples),
        "negative_records": sum(sample.is_negative for sample in samples),
        "negative_ratio": round(
            sum(sample.is_negative for sample in samples) / len(samples), 6
        ),
        "source_records": sum(sample.source == "gretel" for sample in samples),
        "synthetic_records": sum(sample.source == "synthetic" for sample in samples),
        "label_occurrences": dict(sorted(labels.items())),
        "synthetic_kinds": dict(sorted(kinds.items())),
    }


def _synthetic_review_records(
    main: dict[str, list[PreparedSample]],
) -> list[dict[str, Any]]:
    """Create a deterministic, reason-tagged bundle for the human data gate."""

    synthetic = [
        sample
        for split in ("train", "valid", "test")
        for sample in main[split]
        if sample.source == "synthetic"
    ]
    selected: dict[str, tuple[PreparedSample, set[str]]] = {}

    def select(samples: Sequence[PreparedSample], count: int, reason: str) -> None:
        for sample in _stable_shuffle(samples, f"review:{reason}")[:count]:
            if sample.sample_id not in selected:
                selected[sample.sample_id] = (sample, set())
            selected[sample.sample_id][1].add(reason)

    select(synthetic, 100, "random_synthetic")
    for kind in sorted({sample.synthetic_kind for sample in synthetic if sample.synthetic_kind}):
        select(
            [sample for sample in synthetic if sample.synthetic_kind == kind],
            10,
            f"synthetic_kind:{kind}",
        )
    for label in EntityType:
        select(
            [sample for sample in synthetic if label in sample.entity_types],
            5,
            f"canonical_label:{label.value}",
        )
    longest = sorted(
        synthetic, key=lambda sample: (-len(sample.text), sample.sample_id)
    )
    select(longest, 30, "longest_synthetic")

    output: list[dict[str, Any]] = []
    for sample_id in sorted(selected):
        sample, reasons = selected[sample_id]
        record = sample.to_json_record()
        record["review_reasons"] = sorted(reasons)
        output.append(record)
    return output


def build_datasets(
    *,
    lock_file: Path = DEFAULT_LOCK_FILE,
    output_dir: Path = DEFAULT_PROCESSED_DIR,
    calibration_dir: Path = DEFAULT_CALIBRATION_DIR,
) -> BuildResult:
    """Build full-source, main, V0, smoke, and calibration artifacts."""

    if not lock_file.is_file():
        raise DatasetBuildError(f"source lock does not exist: {lock_file}")
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    snapshot_path = Path(lock["snapshot_path"])
    if not snapshot_path.is_absolute():
        snapshot_path = lock_file.resolve().parents[3] / snapshot_path
    if not snapshot_path.is_dir():
        raise DatasetBuildError(f"saved snapshot does not exist: {snapshot_path}")

    source, rejected = _source_samples(snapshot_path)
    full_dir = output_dir / "full_source"
    for split, samples in source.items():
        _write_jsonl(
            full_dir / f"{split}.jsonl",
            (sample.to_json_record() for sample in samples),
        )

    rejected_path = output_dir / "rejected_source.jsonl"
    _write_jsonl(rejected_path, rejected)

    main: dict[str, list[PreparedSample]] = {}
    for split, count in MAIN_STAGE_COUNTS.items():
        main[split] = _build_main_split(source[split], split=split, total=count)
        _write_jsonl(
            output_dir / f"{split}.jsonl",
            (sample.to_json_record() for sample in main[split]),
        )

    v0: dict[str, list[PreparedSample]] = {}
    smoke: dict[str, list[PreparedSample]] = {}
    for split in ("train", "valid", "test"):
        v0[split] = _select_stage(
            main[split],
            total=V0_STAGE_COUNTS[split],
            minimum_per_label=10 if split == "train" else 2,
            key=f"v0:{split}",
        )
        smoke[split] = _select_stage(
            v0[split],
            total=SMOKE_STAGE_COUNTS[split],
            minimum_per_label=2 if split == "train" else 1,
            key=f"smoke:{split}",
        )
        _write_jsonl(
            output_dir / "stages" / "v0" / f"{split}.jsonl",
            (sample.to_json_record() for sample in v0[split]),
        )
        _write_jsonl(
            output_dir / "stages" / "smoke" / f"{split}.jsonl",
            (sample.to_json_record() for sample in smoke[split]),
        )

    synthetic_review_path = output_dir / "synthetic_review_samples.jsonl"
    synthetic_review_records = _synthetic_review_records(main)
    _write_jsonl(synthetic_review_path, synthetic_review_records)

    calibration_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = calibration_dir / "pii_calibration.txt"
    calibration_text = "\n\n<|calibration_document|>\n\n".join(
        generate_calibration_texts(count=500, seed=DATASET_BUILD_SEED)
    )
    calibration_path.write_text(calibration_text + "\n", encoding="utf-8")

    files = sorted(
        [
            *output_dir.glob("*.jsonl"),
            *full_dir.glob("*.jsonl"),
            *(output_dir / "stages").glob("*/*.jsonl"),
            calibration_path,
        ]
    )
    repository_root = lock_file.resolve().parents[3]
    artifacts = {
        _relative(path, repository_root): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    }
    counts = {
        "full_source": {split: len(samples) for split, samples in source.items()},
        "main": {split: len(samples) for split, samples in main.items()},
        "v0": {split: len(samples) for split, samples in v0.items()},
        "smoke": {split: len(samples) for split, samples in smoke.items()},
    }
    manifest = {
        "schema_version": 1,
        "build_version": DATASET_BUILD_VERSION,
        "seed": DATASET_BUILD_SEED,
        "source": {
            "dataset_id": lock["dataset_id"],
            "resolved_revision": lock["resolved_revision"],
            "source_lock": _relative(lock_file, repository_root),
            "source_lock_sha256": _sha256(lock_file),
            "input_counts": lock["split_counts"],
            "converted_counts": counts["full_source"],
            "rejected_records": len(rejected),
        },
        "policy": {
            "policy_version": POLICY_VERSION,
            "prompt_version": PROMPT_VERSION,
            "canonical_labels": [label.value for label in EntityType],
            "source_label_mapping": {
                label: canonical.value
                for label, canonical in sorted(SOURCE_LABEL_MAPPING.items())
            },
            "excluded_source_labels": sorted(EXCLUDED_SOURCE_LABELS),
            "source_defect_policy": "reject_record_without_repair",
        },
        "selection": {
            "official_split_boundaries_preserved": True,
            "synthetic_template_families_split_isolated": True,
            "target_negative_ratio": TARGET_NEGATIVE_RATIO,
            "nested_stages": "smoke_subset_of_v0_subset_of_main",
        },
        "counts": counts,
        "statistics": {
            "main": {split: _sample_stats(samples) for split, samples in main.items()},
            "v0": {split: _sample_stats(samples) for split, samples in v0.items()},
            "smoke": {split: _sample_stats(samples) for split, samples in smoke.items()},
        },
        "calibration": {
            "records": 500,
            "uses_frozen_test_data": False,
            "separator": "<|calibration_document|>",
        },
        "human_review": {
            "synthetic_review_records": len(synthetic_review_records),
            "artifact": _relative(synthetic_review_path, repository_root),
            "selection": [
                "100 deterministic random synthetic records",
                "up to 10 per synthetic kind",
                "up to 5 per canonical label",
                "30 longest synthetic records",
            ],
        },
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    calibration_manifest = {
        "schema_version": 1,
        "build_version": DATASET_BUILD_VERSION,
        "seed": DATASET_BUILD_SEED,
        "records": 500,
        "uses_frozen_test_data": False,
        "artifact": artifacts[_relative(calibration_path, repository_root)],
    }
    _write_json(calibration_dir / "manifest.json", calibration_manifest)
    return BuildResult(
        manifest_path=manifest_path,
        rejected_path=rejected_path,
        counts=counts,
        rejected_count=len(rejected),
    )
