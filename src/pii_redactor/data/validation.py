"""Strict structural, semantic, split, and tokenizer validation for prepared data."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pii_redactor.schema import EntityType

from .constants import (
    CANDIDATE_SYSTEM_PROMPT,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_TOKENIZER_PATH,
    MAIN_STAGE_COUNTS,
    POLICY_VERSION,
    PROMPT_VERSION,
    SMOKE_STAGE_COUNTS,
    V0_STAGE_COUNTS,
)


class DatasetValidationError(RuntimeError):
    """Raised when validation cannot be run due to configuration or I/O."""


@dataclass(frozen=True)
class ValidationResult:
    exit_code: int
    report_path: Path
    statistics_path: Path
    error_count: int
    warning_count: int


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {key: 0 for key in ("p50", "p75", "p90", "p95", "p99", "max")}
    ordered = sorted(values)

    def nearest_rank(percent: float) -> int:
        return ordered[max(0, math.ceil(percent * len(ordered)) - 1)]

    return {
        "p50": nearest_rank(0.50),
        "p75": nearest_rank(0.75),
        "p90": nearest_rank(0.90),
        "p95": nearest_rank(0.95),
        "p99": nearest_rank(0.99),
        "max": ordered[-1],
    }


def _load_and_validate_file(
    path: Path,
    *,
    expected_split: str,
    errors: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    labels: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    negative_count = 0
    character_lengths: list[int] = []
    for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_number}: invalid JSON: {exc}")
            continue
        prefix = f"{path}:{line_number}"
        if not isinstance(record, dict) or set(record) != {
            "sample_id",
            "messages",
            "metadata",
        }:
            errors.append(f"{prefix}: unexpected top-level shape")
            continue
        sample_id = record["sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            errors.append(f"{prefix}: sample_id must be a nonempty string")
            continue
        if sample_id in sample_ids:
            errors.append(f"{prefix}: duplicate sample_id {sample_id}")
        sample_ids.add(sample_id)

        messages = record["messages"]
        expected_roles = ["system", "user", "assistant"]
        if (
            not isinstance(messages, list)
            or len(messages) != 3
            or [item.get("role") for item in messages if isinstance(item, dict)]
            != expected_roles
            or any(
                not isinstance(item, dict)
                or set(item) != {"role", "content"}
                or not isinstance(item["content"], str)
                for item in messages
            )
        ):
            errors.append(f"{prefix}: messages must be system/user/assistant strings")
            continue
        if messages[0]["content"] != CANDIDATE_SYSTEM_PROMPT:
            errors.append(f"{prefix}: system prompt differs from the canonical prompt")
        text = messages[1]["content"]
        character_lengths.append(len(text))
        try:
            assistant = json.loads(messages[2]["content"])
        except json.JSONDecodeError:
            errors.append(f"{prefix}: assistant content is not JSON")
            continue
        if not isinstance(assistant, dict) or set(assistant) != {"entities"}:
            errors.append(f"{prefix}: assistant JSON must contain only entities")
            continue
        assistant_entities = assistant["entities"]
        if not isinstance(assistant_entities, list):
            errors.append(f"{prefix}: assistant entities must be a list")
            continue

        metadata = record["metadata"]
        required_metadata = {
            "source",
            "source_split",
            "is_negative",
            "policy_version",
            "prompt_version",
            "entity_spans",
        }
        if not isinstance(metadata, dict) or not required_metadata <= set(metadata):
            errors.append(f"{prefix}: metadata is missing required fields")
            continue
        if metadata["source_split"] != expected_split:
            errors.append(f"{prefix}: source_split does not match file split")
        if metadata["policy_version"] != POLICY_VERSION:
            errors.append(f"{prefix}: policy_version is not canonical")
        if metadata["prompt_version"] != PROMPT_VERSION:
            errors.append(f"{prefix}: prompt_version is not canonical")
        spans = metadata["entity_spans"]
        if not isinstance(spans, list) or len(spans) != len(assistant_entities):
            errors.append(f"{prefix}: assistant entities and metadata spans differ")
            continue
        previous_end = -1
        reconstructed: list[dict[str, str]] = []
        for index, span in enumerate(spans):
            if not isinstance(span, dict) or set(span) != {
                "type",
                "text",
                "start",
                "end",
            }:
                errors.append(f"{prefix}: span {index} has an unexpected shape")
                continue
            label = span["type"]
            value = span["text"]
            start = span["start"]
            end = span["end"]
            if label not in {item.value for item in EntityType}:
                errors.append(f"{prefix}: span {index} has unknown label {label}")
            if (
                not isinstance(value, str)
                or not value
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end <= start
                or end > len(text)
                or text[start:end] != value
            ):
                errors.append(f"{prefix}: span {index} is not an exact source substring")
            if start < previous_end:
                errors.append(f"{prefix}: spans are overlapping or out of source order")
            previous_end = end
            labels[label] += 1
            reconstructed.append({"type": label, "text": value})
        if assistant_entities != reconstructed:
            errors.append(f"{prefix}: assistant entities do not exactly match span order")
        is_negative = not spans
        if metadata["is_negative"] is not is_negative:
            errors.append(f"{prefix}: is_negative disagrees with entity spans")
        negative_count += is_negative
        if metadata.get("synthetic_kind"):
            kinds[metadata["synthetic_kind"]] += 1
        records.append(record)

    statistics = {
        "records": len(records),
        "negative_records": negative_count,
        "negative_ratio": round(negative_count / len(records), 6) if records else 0,
        "label_occurrences": dict(sorted(labels.items())),
        "synthetic_kinds": dict(sorted(kinds.items())),
        "character_lengths": _percentiles(character_lengths),
    }
    return records, statistics


def _validate_collection_splits(
    name: str,
    records: dict[str, list[dict[str, Any]]],
    errors: list[str],
) -> None:
    exact_owner: dict[str, str] = {}
    normalized_owner: dict[str, str] = {}
    family_owner: dict[str, str] = {}
    sample_id_owner: dict[str, str] = {}
    for split, split_records in records.items():
        for record in split_records:
            sample_id = record["sample_id"]
            if sample_id in sample_id_owner:
                errors.append(
                    f"{name}: sample_id {sample_id} occurs in "
                    f"{sample_id_owner[sample_id]} and {split}"
                )
            sample_id_owner.setdefault(sample_id, split)
            text = record["messages"][1]["content"]
            exact = hashlib.sha256(text.encode("utf-8")).hexdigest()
            normalized = hashlib.sha256(
                _normalized_text(text).encode("utf-8")
            ).hexdigest()
            if exact in exact_owner:
                errors.append(
                    f"{name}: exact text is duplicated in {exact_owner[exact]} and {split}"
                )
            exact_owner.setdefault(exact, split)
            if normalized in normalized_owner:
                errors.append(
                    f"{name}: normalized text is duplicated in "
                    f"{normalized_owner[normalized]} and {split}"
                )
            normalized_owner.setdefault(normalized, split)
            family = record["metadata"].get("template_family")
            if family:
                if family in family_owner and family_owner[family] != split:
                    errors.append(
                        f"{name}: template family {family} occurs across splits"
                    )
                family_owner.setdefault(family, split)


def _token_statistics(
    records: dict[str, list[dict[str, Any]]], tokenizer_path: Path
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if not tokenizer_path.is_dir():
        raise DatasetValidationError(f"tokenizer directory does not exist: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=False
    )
    output: dict[str, Any] = {}
    for split, split_records in records.items():
        lengths: list[int] = []
        prompt_lengths: list[int] = []
        for record in split_records:
            token_ids = tokenizer.apply_chat_template(
                record["messages"], tokenize=True, return_dict=False
            )
            lengths.append(len(token_ids))
            prompt_ids = tokenizer.apply_chat_template(
                record["messages"][:-1],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            prompt_lengths.append(len(prompt_ids))
        output[split] = {
            **_percentiles(lengths),
            "at_or_above_768": sum(length >= 768 for length in lengths),
            "at_or_above_1024": sum(length >= 1024 for length in lengths),
            "at_or_above_1280": sum(length >= 1280 for length in lengths),
            "prompt_at_or_above_768": sum(
                length >= 768 for length in prompt_lengths
            ),
            "prompt_at_or_above_1024": sum(
                length >= 1024 for length in prompt_lengths
            ),
            "prompt_max": max(prompt_lengths, default=0),
        }
    return output


def validate_prepared_datasets(
    *,
    output_dir: Path = DEFAULT_PROCESSED_DIR,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
) -> ValidationResult:
    """Validate every emitted artifact and always write a readable report."""

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise DatasetValidationError(f"dataset manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repository_root = output_dir.resolve().parents[1]
    errors: list[str] = []
    warnings: list[str] = []

    for relative_path, expected in manifest["artifacts"].items():
        path = repository_root / relative_path
        if not path.is_file():
            errors.append(f"manifest artifact is missing: {relative_path}")
        elif _sha256(path) != expected["sha256"]:
            errors.append(f"manifest hash mismatch: {relative_path}")

    specifications = {
        "main": (output_dir, MAIN_STAGE_COUNTS),
        "v0": (output_dir / "stages" / "v0", V0_STAGE_COUNTS),
        "smoke": (output_dir / "stages" / "smoke", SMOKE_STAGE_COUNTS),
    }
    all_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    statistics: dict[str, Any] = {}
    canonical = {item.value for item in EntityType}
    for name, (directory, expected_counts) in specifications.items():
        collection: dict[str, list[dict[str, Any]]] = {}
        statistics[name] = {}
        for split, expected_count in expected_counts.items():
            path = directory / f"{split}.jsonl"
            if not path.is_file():
                errors.append(f"missing dataset file: {path}")
                collection[split] = []
                continue
            records, file_stats = _load_and_validate_file(
                path, expected_split=split, errors=errors
            )
            collection[split] = records
            statistics[name][split] = file_stats
            if len(records) != expected_count:
                errors.append(
                    f"{name}/{split}: expected {expected_count} records, got {len(records)}"
                )
            ratio = file_stats["negative_ratio"]
            if not 0.20 <= ratio <= 0.30:
                errors.append(f"{name}/{split}: negative ratio {ratio} is outside 20-30%")
            missing = canonical - set(file_stats["label_occurrences"])
            if missing:
                errors.append(f"{name}/{split}: missing labels {sorted(missing)}")
        _validate_collection_splits(name, collection, errors)
        all_records[name] = collection

    for split in ("train", "valid", "test"):
        main_ids = {record["sample_id"] for record in all_records["main"][split]}
        v0_ids = {record["sample_id"] for record in all_records["v0"][split]}
        smoke_ids = {record["sample_id"] for record in all_records["smoke"][split]}
        if not v0_ids <= main_ids:
            errors.append(f"{split}: V0 is not a subset of main")
        if not smoke_ids <= v0_ids:
            errors.append(f"{split}: smoke is not a subset of V0")

    full_statistics: dict[str, Any] = {}
    full_records: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "valid", "test"):
        path = output_dir / "full_source" / f"{split}.jsonl"
        records, file_stats = _load_and_validate_file(
            path, expected_split=split, errors=errors
        )
        expected = manifest["counts"]["full_source"][split]
        if len(records) != expected:
            errors.append(f"full_source/{split}: expected {expected}, got {len(records)}")
        full_records[split] = records
        full_statistics[split] = file_stats
    _validate_collection_splits("full_source", full_records, errors)
    statistics["full_source"] = full_statistics

    statistics["token_lengths"] = _token_statistics(
        all_records["main"], tokenizer_path
    )
    if manifest["source"]["rejected_records"]:
        warnings.append(
            f"{manifest['source']['rejected_records']} source records were rejected "
            "without repair; "
            "see rejected_source.jsonl"
        )
    statistics["validation"] = {
        "status": "passed" if not errors else "failed",
        "errors": len(errors),
        "warnings": len(warnings),
    }

    report_dir = output_dir / "validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = report_dir / "statistics.json"
    statistics_path.write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = report_dir / "validation_report.md"
    lines = [
        "# Prepared dataset validation",
        "",
        f"**Status:** {'PASS' if not errors else 'FAIL'}",
        "",
        f"- Errors: {len(errors)}",
        f"- Warnings: {len(warnings)}",
        f"- Source revision: `{manifest['source']['resolved_revision']}`",
        f"- Build version: `{manifest['build_version']}`",
        "",
        "## Dataset stages",
        "",
        "| Stage | Split | Records | Negatives | Negative ratio |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for stage in ("main", "v0", "smoke", "full_source"):
        for split in ("train", "valid", "test"):
            item = statistics[stage][split]
            lines.append(
                f"| {stage} | {split} | {item['records']} | "
                f"{item['negative_records']} | {item['negative_ratio']:.1%} |"
            )
    lines.extend(["", "## Token lengths (main, full chat sequence)", ""])
    for split, item in statistics["token_lengths"].items():
        lines.append(
            f"- **{split}:** P95 {item['p95']}, P99 {item['p99']}, max {item['max']}; "
            f">=768: {item['at_or_above_768']}, >=1024: {item['at_or_above_1024']}; "
            f"prompt max {item['prompt_max']}, prompt >=768: "
            f"{item['prompt_at_or_above_768']}"
        )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in errors[:200])
        if len(errors) > 200:
            lines.append(f"- … {len(errors) - 200} additional errors omitted")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ValidationResult(
        exit_code=0 if not errors else 2,
        report_path=report_path,
        statistics_path=statistics_path,
        error_count=len(errors),
        warning_count=len(warnings),
    )
