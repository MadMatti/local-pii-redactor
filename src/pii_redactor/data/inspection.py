"""Deterministic inspection and human-review artifacts for the Gretel dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from datasets import DatasetDict, load_from_disk
from pydantic import ValidationError
from tqdm import tqdm
from transformers import AutoTokenizer

from pii_redactor.schema import EntityType

from .constants import (
    CANDIDATE_SYSTEM_PROMPT,
    DEFAULT_LOCK_FILE,
    DEFAULT_TOKENIZER_PATH,
    EXCLUDED_SOURCE_LABELS,
    EXPECTED_SPLIT_COUNTS,
    INSPECTION_SEED,
    KNOWN_SOURCE_LABELS,
    PROJECT_ROOT,
    SOURCE_LABEL_MAPPING,
    TRUNCATION_THRESHOLDS,
)
from .download import DownloadError, read_source_lock, resolve_snapshot_path
from .models import SourceRecord
from .source import (
    EntityParseError,
    classify_source_types,
    count_exact_occurrences,
    normalize_for_duplicate,
)


BLOCKING_ISSUE_TYPES = frozenset(
    {
        "ambiguous_canonical_mapping",
        "cross_split_exact_duplicate",
        "cross_split_normalized_duplicate",
        "duplicate_uid",
        "entity_not_found",
        "malformed_record",
        "split_count_mismatch",
        "unknown_source_label",
    }
)


class InspectionError(RuntimeError):
    """Raised when inspection cannot complete and write review artifacts."""


@dataclass(frozen=True, order=True)
class RecordRef:
    split: str
    index: int
    uid: str

    def as_dict(self) -> dict[str, Any]:
        return {"split": self.split, "index": self.index, "uid": self.uid}


@dataclass
class MetricsAccumulator:
    record_count: int = 0
    char_lengths: list[int] = field(default_factory=list)
    input_token_lengths: list[int] = field(default_factory=list)
    output_token_lengths: list[int] = field(default_factory=list)
    estimated_sequence_lengths: list[int] = field(default_factory=list)
    entity_counts: list[int] = field(default_factory=list)
    source_labels: Counter[str] = field(default_factory=Counter)
    canonical_labels: Counter[str] = field(default_factory=Counter)
    domains: Counter[str] = field(default_factory=Counter)
    document_types: Counter[str] = field(default_factory=Counter)
    unknown_labels: Counter[str] = field(default_factory=Counter)
    negative_records: int = 0
    mixed_included_excluded_records: int = 0
    malformed_records: int = 0
    zero_occurrence_entities: int = 0
    unique_occurrence_entities: int = 0
    multiple_occurrence_entities: int = 0
    multi_type_entities: int = 0
    source_order_violation_records: int = 0

    def merge(self, other: "MetricsAccumulator") -> None:
        self.record_count += other.record_count
        self.char_lengths.extend(other.char_lengths)
        self.input_token_lengths.extend(other.input_token_lengths)
        self.output_token_lengths.extend(other.output_token_lengths)
        self.estimated_sequence_lengths.extend(other.estimated_sequence_lengths)
        self.entity_counts.extend(other.entity_counts)
        self.source_labels.update(other.source_labels)
        self.canonical_labels.update(other.canonical_labels)
        self.domains.update(other.domains)
        self.document_types.update(other.document_types)
        self.unknown_labels.update(other.unknown_labels)
        self.negative_records += other.negative_records
        self.mixed_included_excluded_records += other.mixed_included_excluded_records
        self.malformed_records += other.malformed_records
        self.zero_occurrence_entities += other.zero_occurrence_entities
        self.unique_occurrence_entities += other.unique_occurrence_entities
        self.multiple_occurrence_entities += other.multiple_occurrence_entities
        self.multi_type_entities += other.multi_type_entities
        self.source_order_violation_records += other.source_order_violation_records


@dataclass(frozen=True)
class InspectionConfig:
    lock_file: Path = DEFAULT_LOCK_FILE
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH
    output_dir: Path | None = None
    seed: int = INSPECTION_SEED
    tokenizer_batch_size: int = 512


@dataclass(frozen=True)
class InspectionOutcome:
    output_dir: Path
    resolved_revision: str
    record_count: int
    issue_count: int
    blocker_count: int
    review_sample_count: int

    @property
    def exit_code(self) -> int:
        return 2 if self.blocker_count else 0


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _length_summary(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        return {
            "count": 0,
            "p50": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "mean": 0.0,
        }
    data = np.asarray(values, dtype=np.int64)
    percentiles = np.percentile(
        data,
        [50, 75, 90, 95, 99],
        method="nearest",
    )
    return {
        "count": int(data.size),
        "p50": int(percentiles[0]),
        "p75": int(percentiles[1]),
        "p90": int(percentiles[2]),
        "p95": int(percentiles[3]),
        "p99": int(percentiles[4]),
        "max": int(data.max()),
        "mean": round(float(data.mean()), 3),
    }


def summarize_metrics(metrics: MetricsAccumulator) -> dict[str, Any]:
    record_count = metrics.record_count
    negative_percentage = (
        round(100.0 * metrics.negative_records / record_count, 3)
        if record_count
        else 0.0
    )
    return {
        "record_count": record_count,
        "character_lengths": _length_summary(metrics.char_lengths),
        "input_token_lengths": _length_summary(metrics.input_token_lengths),
        "candidate_output_token_lengths": _length_summary(
            metrics.output_token_lengths
        ),
        "estimated_total_sequence_lengths": _length_summary(
            metrics.estimated_sequence_lengths
        ),
        "entities_per_document": _length_summary(metrics.entity_counts),
        "source_label_frequency": dict(sorted(metrics.source_labels.items())),
        "canonical_label_frequency": dict(
            sorted(metrics.canonical_labels.items())
        ),
        "domain_frequency": dict(sorted(metrics.domains.items())),
        "document_type_frequency": dict(sorted(metrics.document_types.items())),
        "unknown_label_frequency": dict(sorted(metrics.unknown_labels.items())),
        "negative_after_filtering": {
            "count": metrics.negative_records,
            "percentage": negative_percentage,
        },
        "records_with_included_and_excluded_labels": (
            metrics.mixed_included_excluded_records
        ),
        "malformed_records": metrics.malformed_records,
        "entity_occurrences": {
            "zero": metrics.zero_occurrence_entities,
            "unique": metrics.unique_occurrence_entities,
            "multiple": metrics.multiple_occurrence_entities,
        },
        "multi_type_entities": metrics.multi_type_entities,
        "source_order_violation_records": metrics.source_order_violation_records,
        "estimated_truncation_counts": {
            str(threshold): sum(
                length >= threshold for length in metrics.estimated_sequence_lengths
            )
            for threshold in TRUNCATION_THRESHOLDS
        },
    }


def _token_lengths(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    batch_size: int,
) -> Iterator[int]:
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        for token_ids in encoded["input_ids"]:
            yield len(token_ids)


def _prompt_overhead_tokens(tokenizer: Any) -> int:
    messages = [
        {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
    ]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        if hasattr(encoded, "keys") and "input_ids" in encoded:
            input_ids = encoded["input_ids"]
            if input_ids and isinstance(input_ids[0], (list, tuple, np.ndarray)):
                return len(input_ids[0])
            return len(input_ids)
        return len(encoded)
    except (AttributeError, TypeError, ValueError):
        return len(
            tokenizer.encode(CANDIDATE_SYSTEM_PROMPT, add_special_tokens=False)
        ) + 16


def _candidate_output(record: SourceRecord) -> tuple[str, set[str], set[str]]:
    entities: list[dict[str, str]] = []
    included_labels: set[str] = set()
    excluded_labels: set[str] = set()
    for entity in record.entities:
        classification = classify_source_types(entity.types)
        included_labels.update(classification.included_labels)
        excluded_labels.update(classification.excluded_labels)
        if len(classification.canonical_types) == 1:
            entities.append(
                {
                    "type": classification.canonical_types[0].value,
                    "text": entity.entity,
                }
            )
    return (
        json.dumps({"entities": entities}, ensure_ascii=False, separators=(",", ":")),
        included_labels,
        excluded_labels,
    )


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _atomic_write_text(path, content)


def _choose(
    refs: Iterable[RecordRef],
    count: int,
    rng: random.Random,
) -> list[RecordRef]:
    unique = sorted(set(refs))
    if len(unique) <= count:
        return unique
    return sorted(rng.sample(unique, count))


def _duplicate_groups(
    groups: dict[str, list[RecordRef]],
) -> tuple[list[list[RecordRef]], list[list[RecordRef]]]:
    duplicates: list[list[RecordRef]] = []
    cross_split: list[list[RecordRef]] = []
    for key in sorted(groups):
        refs = sorted(set(groups[key]))
        if len(refs) < 2:
            continue
        duplicates.append(refs)
        if len({ref.split for ref in refs}) > 1:
            cross_split.append(refs)
    return duplicates, cross_split


def _group_examples(groups: Sequence[Sequence[RecordRef]], cap: int = 100) -> list[Any]:
    return [[ref.as_dict() for ref in group[:10]] for group in groups[:cap]]


def _issue(
    issue_type: str,
    ref: RecordRef | None,
    details: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "issue_type": issue_type,
        "severity": "blocker" if issue_type in BLOCKING_ISSUE_TYPES else "warning",
        "details": details,
    }
    if ref is not None:
        result.update(ref.as_dict())
    return result


def _inspection_output_dir(config: InspectionConfig, revision: str) -> Path:
    if config.output_dir is not None:
        return config.output_dir
    return (
        PROJECT_ROOT
        / "data"
        / "intermediate"
        / "gretel-pii-masking-en-v1"
        / revision
        / "inspection"
    )


def _load_tokenizer(tokenizer_path: Path) -> Any:
    if not tokenizer_path.exists():
        raise InspectionError(f"tokenizer path does not exist: {tokenizer_path}")
    return AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
    )


def inspect_source_dataset(
    config: InspectionConfig,
    *,
    tokenizer: Any | None = None,
    load_from_disk_fn: Any = load_from_disk,
) -> InspectionOutcome:
    """Inspect the pinned source dataset and write a deterministic review bundle."""

    try:
        lock = read_source_lock(config.lock_file)
    except DownloadError as exc:
        raise InspectionError(str(exc)) from exc
    revision = str(lock["resolved_revision"])
    snapshot_path = resolve_snapshot_path(lock)
    if not snapshot_path.exists():
        raise InspectionError(f"snapshot does not exist: {snapshot_path}")

    dataset = load_from_disk_fn(str(snapshot_path))
    if not isinstance(dataset, DatasetDict):
        raise InspectionError("locked snapshot is not a DatasetDict")
    tokenizer = tokenizer or _load_tokenizer(config.tokenizer_path)
    output_dir = _inspection_output_dir(config, revision)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_overhead = _prompt_overhead_tokens(tokenizer)
    metrics_by_split: dict[str, MetricsAccumulator] = {}
    overall = MetricsAccumulator()
    issues: list[dict[str, Any]] = []
    all_refs: list[RecordRef] = []
    source_label_refs: dict[str, list[RecordRef]] = defaultdict(list)
    negative_refs: list[RecordRef] = []
    zero_occurrence_refs: list[RecordRef] = []
    multiple_occurrence_refs: list[RecordRef] = []
    malformed_refs: list[RecordRef] = []
    multi_type_refs: list[RecordRef] = []
    source_order_violation_refs: list[RecordRef] = []
    long_records: list[tuple[int, RecordRef]] = []
    long_output_records: list[tuple[int, RecordRef]] = []
    high_entity_records: list[tuple[int, RecordRef]] = []
    uid_groups: dict[str, list[RecordRef]] = defaultdict(list)
    exact_text_groups: dict[str, list[RecordRef]] = defaultdict(list)
    normalized_text_groups: dict[str, list[RecordRef]] = defaultdict(list)
    template_groups: dict[str, list[RecordRef]] = defaultdict(list)

    for lock_blocker in lock.get("review_blockers", []):
        issues.append(
            _issue(
                str(lock_blocker.get("type", "source_lock_blocker")),
                None,
                dict(lock_blocker),
            )
        )

    for split in EXPECTED_SPLIT_COUNTS:
        if split not in dataset:
            raise InspectionError(f"snapshot is missing required split: {split}")
        split_dataset = dataset[split]
        metrics = MetricsAccumulator()
        metrics_by_split[split] = metrics

        if len(split_dataset) != EXPECTED_SPLIT_COUNTS[split]:
            issues.append(
                _issue(
                    "split_count_mismatch",
                    None,
                    {
                        "split": split,
                        "expected": EXPECTED_SPLIT_COUNTS[split],
                        "actual": len(split_dataset),
                    },
                )
            )

        texts = split_dataset["text"]
        input_lengths = _token_lengths(
            tokenizer,
            texts,
            batch_size=config.tokenizer_batch_size,
        )
        rows = zip(split_dataset, input_lengths, strict=True)
        for index, (row, input_token_length) in enumerate(
            tqdm(rows, total=len(split_dataset), desc=f"Inspecting {split}")
        ):
            raw_uid = row.get("uid")
            uid = raw_uid if isinstance(raw_uid, str) else f"<invalid-{index}>"
            ref = RecordRef(split=split, index=index, uid=uid)
            all_refs.append(ref)
            raw_text = row.get("text")
            text = raw_text if isinstance(raw_text, str) else ""

            metrics.record_count += 1
            metrics.char_lengths.append(len(text))
            metrics.input_token_lengths.append(input_token_length)
            if isinstance(row.get("domain"), str):
                metrics.domains[row["domain"]] += 1
            if isinstance(row.get("document_type"), str):
                metrics.document_types[row["document_type"]] += 1

            uid_groups[uid].append(ref)
            exact_text_groups[_hash_text(text)].append(ref)
            normalized_text_groups[_hash_text(normalize_for_duplicate(text))].append(
                ref
            )
            template_value = "\x1f".join(
                [
                    str(row.get("domain", "")),
                    str(row.get("document_type", "")),
                    normalize_for_duplicate(str(row.get("document_description", ""))),
                ]
            )
            template_groups[_hash_text(template_value)].append(ref)

            try:
                record = SourceRecord.from_mapping(row)
            except (EntityParseError, ValidationError, KeyError, TypeError) as exc:
                metrics.malformed_records += 1
                metrics.entity_counts.append(0)
                metrics.negative_records += 1
                malformed_refs.append(ref)
                empty_output = '{"entities":[]}'
                output_length = len(
                    tokenizer.encode(empty_output, add_special_tokens=False)
                )
                metrics.output_token_lengths.append(output_length)
                estimated = prompt_overhead + input_token_length + output_length
                metrics.estimated_sequence_lengths.append(estimated)
                long_records.append((estimated, ref))
                long_output_records.append((output_length, ref))
                high_entity_records.append((0, ref))
                issues.append(
                    _issue(
                        "malformed_record",
                        ref,
                        {"error_type": type(exc).__name__, "message": str(exc)},
                    )
                )
                continue

            metrics.entity_counts.append(len(record.entities))
            high_entity_records.append((len(record.entities), ref))
            candidate_output, included_in_record, excluded_in_record = _candidate_output(
                record
            )
            output_length = len(
                tokenizer.encode(candidate_output, add_special_tokens=False)
            )
            estimated = prompt_overhead + input_token_length + output_length
            metrics.output_token_lengths.append(output_length)
            metrics.estimated_sequence_lengths.append(estimated)
            long_records.append((estimated, ref))
            long_output_records.append((output_length, ref))

            canonical_count = 0
            unique_positions_in_annotation_order: list[int] = []
            for entity_index, entity in enumerate(record.entities):
                if len(entity.types) > 1:
                    metrics.multi_type_entities += 1
                    multi_type_refs.append(ref)

                classification = classify_source_types(entity.types)
                for source_label in entity.types:
                    metrics.source_labels[source_label] += 1
                    source_label_refs[source_label].append(ref)
                for canonical in classification.canonical_types:
                    metrics.canonical_labels[canonical.value] += 1
                    canonical_count += 1
                for unknown in classification.unknown_labels:
                    metrics.unknown_labels[unknown] += 1
                    issues.append(
                        _issue(
                            "unknown_source_label",
                            ref,
                            {"entity_index": entity_index, "source_label": unknown},
                        )
                    )
                if classification.is_ambiguous:
                    issues.append(
                        _issue(
                            "ambiguous_canonical_mapping",
                            ref,
                            {
                                "entity_index": entity_index,
                                "source_types": list(entity.types),
                                "canonical_types": [
                                    item.value for item in classification.canonical_types
                                ],
                            },
                        )
                    )

                occurrences = count_exact_occurrences(record.text, entity.entity)
                if occurrences == 0:
                    metrics.zero_occurrence_entities += 1
                    zero_occurrence_refs.append(ref)
                    issues.append(
                        _issue(
                            "entity_not_found",
                            ref,
                            {
                                "entity_index": entity_index,
                                "entity": entity.entity,
                                "source_types": list(entity.types),
                            },
                        )
                    )
                elif occurrences == 1:
                    metrics.unique_occurrence_entities += 1
                    unique_positions_in_annotation_order.append(
                        record.text.find(entity.entity)
                    )
                else:
                    metrics.multiple_occurrence_entities += 1
                    multiple_occurrence_refs.append(ref)

            if any(
                current < previous
                for previous, current in zip(
                    unique_positions_in_annotation_order,
                    unique_positions_in_annotation_order[1:],
                )
            ):
                metrics.source_order_violation_records += 1
                source_order_violation_refs.append(ref)
                issues.append(
                    _issue(
                        "source_annotation_order_violation",
                        ref,
                        {
                            "unique_entity_positions_in_annotation_order": (
                                unique_positions_in_annotation_order
                            )
                        },
                    )
                )

            if canonical_count == 0:
                metrics.negative_records += 1
                negative_refs.append(ref)
            if included_in_record and excluded_in_record:
                metrics.mixed_included_excluded_records += 1

        overall.merge(metrics)

    uid_duplicates, uid_cross = _duplicate_groups(uid_groups)
    exact_duplicates, exact_cross = _duplicate_groups(exact_text_groups)
    normalized_duplicates, normalized_cross = _duplicate_groups(
        normalized_text_groups
    )
    template_duplicates, template_cross = _duplicate_groups(template_groups)

    for group in uid_duplicates:
        issues.append(
            _issue(
                "duplicate_uid",
                None,
                {"records": [ref.as_dict() for ref in group[:20]]},
            )
        )
    for group in exact_cross:
        issues.append(
            _issue(
                "cross_split_exact_duplicate",
                None,
                {"records": [ref.as_dict() for ref in group[:20]]},
            )
        )
    for group in normalized_cross:
        issues.append(
            _issue(
                "cross_split_normalized_duplicate",
                None,
                {"records": [ref.as_dict() for ref in group[:20]]},
            )
        )

    split_overlap = {
        "duplicate_uids": {
            "group_count": len(uid_duplicates),
            "cross_split_group_count": len(uid_cross),
            "examples": _group_examples(uid_duplicates),
        },
        "exact_text_duplicates": {
            "group_count": len(exact_duplicates),
            "cross_split_group_count": len(exact_cross),
            "examples": _group_examples(exact_cross),
        },
        "normalized_text_duplicates": {
            "group_count": len(normalized_duplicates),
            "cross_split_group_count": len(normalized_cross),
            "examples": _group_examples(normalized_cross),
        },
        "template_groups": {
            "group_count": len(template_duplicates),
            "cross_split_group_count": len(template_cross),
            "examples": _group_examples(template_cross),
            "note": (
                "Template groups use domain, document type, and normalized document "
                "description; reuse is diagnostic and is not automatically blocking."
            ),
        },
    }

    rng = random.Random(config.seed)
    review_reasons: dict[RecordRef, set[str]] = defaultdict(set)

    def select(refs: Iterable[RecordRef], count: int, reason: str) -> None:
        for selected in _choose(refs, count, rng):
            review_reasons[selected].add(reason)

    select(all_refs, 100, "random")
    for source_label in sorted(SOURCE_LABEL_MAPPING):
        select(source_label_refs[source_label], 20, f"included_label:{source_label}")
    for source_label in sorted(EXCLUDED_SOURCE_LABELS):
        select(source_label_refs[source_label], 10, f"excluded_label:{source_label}")
    select(negative_refs, 50, "negative_after_filtering")
    select(zero_occurrence_refs, 100, "entity_not_found")
    select(malformed_refs, 100, "malformed_record")
    select(multiple_occurrence_refs, 100, "multiple_exact_occurrences")
    select(multi_type_refs, 200, "multi_type_entity")
    select(source_order_violation_refs, 100, "source_annotation_order_violation")
    for _, ref in sorted(long_records, key=lambda item: (-item[0], item[1]))[:30]:
        review_reasons[ref].add("longest_estimated_sequence")
    for _, ref in sorted(
        long_output_records, key=lambda item: (-item[0], item[1])
    )[:30]:
        review_reasons[ref].add("longest_candidate_output")
    for _, ref in sorted(
        high_entity_records, key=lambda item: (-item[0], item[1])
    )[:30]:
        review_reasons[ref].add("highest_entity_count")
    for label in ("cross_split_exact_duplicate", "cross_split_normalized_duplicate"):
        groups = exact_cross if label.endswith("exact_duplicate") else normalized_cross
        for group in groups[:50]:
            for ref in group[:10]:
                review_reasons[ref].add(label)

    review_rows: list[dict[str, Any]] = []
    for ref in sorted(review_reasons):
        row = dataset[ref.split][ref.index]
        parsed_entities: list[dict[str, Any]] | None = None
        parse_error: str | None = None
        candidate_labels: list[str] = []
        source_labels: list[str] = []
        would_be_negative = True
        try:
            record = SourceRecord.from_mapping(row)
            parsed_entities = [entity.model_dump(mode="json") for entity in record.entities]
            for entity in record.entities:
                classification = classify_source_types(entity.types)
                source_labels.extend(entity.types)
                candidate_labels.extend(
                    canonical.value for canonical in classification.canonical_types
                )
            would_be_negative = not candidate_labels
        except (EntityParseError, ValidationError, KeyError, TypeError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

        review_rows.append(
            {
                "split": ref.split,
                "index": ref.index,
                "uid": ref.uid,
                "selection_reasons": sorted(review_reasons[ref]),
                "domain": row.get("domain"),
                "document_type": row.get("document_type"),
                "document_description": row.get("document_description"),
                "text": row.get("text"),
                "raw_entities": row.get("entities"),
                "parsed_entities": parsed_entities,
                "parse_error": parse_error,
                "source_labels": sorted(set(source_labels)),
                "candidate_canonical_labels": sorted(set(candidate_labels)),
                "would_be_negative": would_be_negative,
            }
        )

    issues.sort(
        key=lambda item: (
            item["severity"],
            item["issue_type"],
            item.get("split", ""),
            item.get("index", -1),
        )
    )
    blocker_count = sum(item["severity"] == "blocker" for item in issues)

    statistics = {
        "schema_version": 1,
        "dataset": {
            "dataset_id": lock["dataset_id"],
            "resolved_revision": revision,
            "split_counts": lock.get("split_counts", {}),
        },
        "tokenizer": {
            "path": str(config.tokenizer_path),
            "class": type(tokenizer).__name__,
            "prompt_overhead_tokens": prompt_overhead,
            "sequence_estimate_method": (
                "prompt/template overhead with empty user and assistant plus exact "
                "standalone input and candidate-output token counts"
            ),
        },
        "candidate_policy": {
            "included_source_labels": sorted(SOURCE_LABEL_MAPPING),
            "excluded_source_labels": sorted(EXCLUDED_SOURCE_LABELS),
            "known_source_label_count": len(KNOWN_SOURCE_LABELS),
            "canonical_labels": sorted(item.value for item in EntityType),
        },
        "splits": {
            split: summarize_metrics(metrics)
            for split, metrics in metrics_by_split.items()
        },
        "overall": summarize_metrics(overall),
        "quality": {
            "issue_count": len(issues),
            "blocker_count": blocker_count,
            "issue_type_frequency": dict(
                sorted(Counter(item["issue_type"] for item in issues).items())
            ),
        },
        "outliers": {
            "highest_entity_counts": [
                {"entity_count": count, **ref.as_dict()}
                for count, ref in sorted(
                    high_entity_records, key=lambda item: (-item[0], item[1])
                )[:30]
            ],
            "longest_candidate_outputs": [
                {"output_tokens": count, **ref.as_dict()}
                for count, ref in sorted(
                    long_output_records, key=lambda item: (-item[0], item[1])
                )[:30]
            ],
            "longest_estimated_sequences": [
                {"estimated_sequence_tokens": count, **ref.as_dict()}
                for count, ref in sorted(
                    long_records, key=lambda item: (-item[0], item[1])
                )[:30]
            ],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = output_dir / "statistics.json"
    labels_path = output_dir / "label_inventory.csv"
    issues_path = output_dir / "quality_issues.jsonl"
    overlap_path = output_dir / "split_overlap.json"
    review_path = output_dir / "review_samples.jsonl"
    summary_path = output_dir / "summary.md"
    manifest_path = output_dir / "inspection_manifest.json"

    _write_json(statistics_path, statistics)
    _write_json(overlap_path, split_overlap)
    _write_jsonl(issues_path, issues)
    _write_jsonl(review_path, review_rows)

    label_rows: list[list[Any]] = []
    observed_labels = overall.source_labels
    for label in sorted(KNOWN_SOURCE_LABELS | set(observed_labels)):
        canonical = SOURCE_LABEL_MAPPING.get(label)
        if canonical:
            policy = "included"
        elif label in EXCLUDED_SOURCE_LABELS:
            policy = "excluded"
        else:
            policy = "unknown"
        label_rows.append(
            [
                label,
                policy,
                canonical.value if canonical else "",
                observed_labels.get(label, 0),
                overall.unknown_labels.get(label, 0),
            ]
        )
    csv_lines: list[str] = []
    with tempfile.NamedTemporaryFile(
        mode="w+", encoding="utf-8", newline="", delete=True
    ) as csv_buffer:
        writer = csv.writer(csv_buffer)
        writer.writerow(
            ["source_label", "policy", "canonical_label", "entity_count", "unknown_count"]
        )
        writer.writerows(label_rows)
        csv_buffer.flush()
        csv_buffer.seek(0)
        csv_lines.append(csv_buffer.read())
    _atomic_write_text(labels_path, "".join(csv_lines))

    overall_summary = statistics["overall"]
    summary = f"""# Gretel source dataset inspection

## Source

- Dataset: `{lock['dataset_id']}`
- Resolved revision: `{revision}`
- Records: {overall.record_count:,}
- Tokenizer: `{config.tokenizer_path}` ({type(tokenizer).__name__})
- Inspection seed: {config.seed}

## Candidate policy

- Canonical labels: {len(EntityType)}
- Included source labels: {len(SOURCE_LABEL_MAPPING)}
- Explicitly excluded source labels: {len(EXCLUDED_SOURCE_LABELS)}
- Records becoming negative after filtering: {overall.negative_records:,} ({overall_summary['negative_after_filtering']['percentage']:.3f}%)
- Records containing included and excluded labels: {overall.mixed_included_excluded_records:,}

## Lengths

| Measure | P50 | P75 | P90 | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Characters | {overall_summary['character_lengths']['p50']} | {overall_summary['character_lengths']['p75']} | {overall_summary['character_lengths']['p90']} | {overall_summary['character_lengths']['p95']} | {overall_summary['character_lengths']['p99']} | {overall_summary['character_lengths']['max']} |
| Input tokens | {overall_summary['input_token_lengths']['p50']} | {overall_summary['input_token_lengths']['p75']} | {overall_summary['input_token_lengths']['p90']} | {overall_summary['input_token_lengths']['p95']} | {overall_summary['input_token_lengths']['p99']} | {overall_summary['input_token_lengths']['max']} |
| Candidate output tokens | {overall_summary['candidate_output_token_lengths']['p50']} | {overall_summary['candidate_output_token_lengths']['p75']} | {overall_summary['candidate_output_token_lengths']['p90']} | {overall_summary['candidate_output_token_lengths']['p95']} | {overall_summary['candidate_output_token_lengths']['p99']} | {overall_summary['candidate_output_token_lengths']['max']} |
| Estimated total sequence | {overall_summary['estimated_total_sequence_lengths']['p50']} | {overall_summary['estimated_total_sequence_lengths']['p75']} | {overall_summary['estimated_total_sequence_lengths']['p90']} | {overall_summary['estimated_total_sequence_lengths']['p95']} | {overall_summary['estimated_total_sequence_lengths']['p99']} | {overall_summary['estimated_total_sequence_lengths']['max']} |

Estimated total sequence length is diagnostic only. It combines a fixed chat-template/prompt overhead with standalone input and candidate-output token counts; no training JSONL has been generated.

## Quality gate

- Quality issues: {len(issues):,}
- Review blockers: {blocker_count:,}
- Malformed records: {overall.malformed_records:,}
- Entity values absent from source: {overall.zero_occurrence_entities:,}
- Entity values with multiple exact occurrences: {overall.multiple_occurrence_entities:,}
- Multi-type entities: {overall.multi_type_entities:,}
- Records whose source annotations are not in text order: {overall.source_order_violation_records:,}
- Cross-split exact duplicate groups: {len(exact_cross):,}
- Cross-split normalized duplicate groups: {len(normalized_cross):,}
- Cross-split template groups: {len(template_cross):,}

## Human review

Review `{review_path.name}` together with `{issues_path.name}` and `{overlap_path.name}` before approving label boundaries, repeated-value behavior, or dataset conversion. This inspection intentionally stops before writing MLX training JSONL or synthetic examples.
"""
    _atomic_write_text(summary_path, summary)

    artifact_paths = [
        summary_path,
        statistics_path,
        labels_path,
        issues_path,
        overlap_path,
        review_path,
    ]
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_lock": str(config.lock_file),
        "dataset_id": lock["dataset_id"],
        "resolved_revision": revision,
        "snapshot_path": str(snapshot_path),
        "seed": config.seed,
        "tokenizer_path": str(config.tokenizer_path),
        "library_versions": {
            "datasets": _package_version("datasets"),
            "numpy": _package_version("numpy"),
            "pydantic": _package_version("pydantic"),
            "transformers": _package_version("transformers"),
        },
        "artifacts": {
            path.name: {"sha256": _hash_file(path), "bytes": path.stat().st_size}
            for path in artifact_paths
        },
        "record_count": overall.record_count,
        "issue_count": len(issues),
        "blocker_count": blocker_count,
        "review_sample_count": len(review_rows),
    }
    _write_json(manifest_path, manifest)

    return InspectionOutcome(
        output_dir=output_dir,
        resolved_revision=revision,
        record_count=overall.record_count,
        issue_count=len(issues),
        blocker_count=blocker_count,
        review_sample_count=len(review_rows),
    )
