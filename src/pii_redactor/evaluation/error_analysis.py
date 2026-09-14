"""Deterministic, occurrence-level error review for frozen PII predictions.

Cause names describe heuristics, not established model failure mechanisms.
Only ``review_samples.jsonl`` contains source text and generated responses.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pii_redactor.data.prepared import sample_from_json_record
from pii_redactor.schema import EntityType

from .metrics import (
    AlignedPrediction,
    EvaluationError,
    align_predictions,
    evaluate_predictions,
    load_predictions,
    sha256_file,
)
from .parsing import parse_prediction


@dataclass(frozen=True)
class ErrorAnalysis:
    statistics: dict[str, Any]
    review_samples: tuple[dict[str, Any], ...]


def _group_key(kind: str, label: str | None, cause: str) -> str:
    return f"{kind}/{label or 'UNASSIGNED'}/{cause}"


def _overlaps(item: AlignedPrediction, gold: dict[str, Any]) -> bool:
    return (
        item.start is not None
        and item.end is not None
        and item.start < gold["end"]
        and gold["start"] < item.end
    )


def _sample_errors(row: dict[str, Any], raw_output: str) -> dict[str, Any]:
    sample = sample_from_json_record(row)
    parsed = parse_prediction(raw_output)
    aligned = align_predictions(sample.text, parsed.entities) if parsed.schema_valid else ()
    truth = {
        (entity.type.value, entity.start, entity.end): entity.metadata_dict()
        for entity in sample.entities
    }
    matched: set[tuple[str, int, int]] = set()
    unmatched_predictions: list[tuple[int, AlignedPrediction]] = []
    for index, item in enumerate(aligned):
        key = (item.entity.type.value, item.start, item.end)
        if item.is_source_member and key in truth and key not in matched:
            matched.add(key)
        else:
            unmatched_predictions.append((index, item))

    errors: list[dict[str, Any]] = []
    for key in sorted(truth.keys() - matched, key=lambda item: (item[1], item[2], item[0])):
        label, start, end = key
        gold = truth[key]
        if not parsed.schema_valid:
            cause = "invalid_schema"
        elif any(
            item.start == start and item.end == end and item.entity.type.value != label
            for _, item in unmatched_predictions
        ):
            cause = "wrong_label"
        elif any(
            item.entity.type.value == label and _overlaps(item, gold)
            for _, item in unmatched_predictions
        ):
            cause = "boundary_mismatch"
        elif any(
            item.is_source_member
            and item.entity.type.value == label
            and item.entity.text == gold["text"]
            for item in aligned
        ):
            cause = "repeated_occurrence_omission"
        else:
            cause = "missed_entity"
        errors.append({"kind": "FN", "label": label, "suspected_cause": cause, "entity": gold})

    for index, item in unmatched_predictions:
        label = item.entity.type.value
        if not item.is_source_member:
            cause = (
                "overpredicted_occurrence"
                if item.entity.text in sample.text
                else "hallucinated_text"
            )
        elif any(
            gold["start"] == item.start
            and gold["end"] == item.end
            and gold["type"] != label
            for gold in truth.values()
        ):
            cause = "wrong_label"
        elif any(
            gold["type"] == label and _overlaps(item, gold)
            for gold in truth.values()
        ):
            cause = "boundary_mismatch"
        else:
            cause = "spurious_entity"
        errors.append(
            {
                "kind": "FP",
                "label": label,
                "suspected_cause": cause,
                "prediction_index": index,
                "entity": {
                    "type": label,
                    "text": item.entity.text,
                    "start": item.start,
                    "end": item.end,
                },
            }
        )

    # A malformed response on a negative document has no FN or FP under the
    # strict evaluator. Retain it as a separate document-level schema issue.
    if not parsed.schema_valid:
        errors.append(
            {"kind": "SCHEMA", "label": None, "suspected_cause": "invalid_schema"}
        )
    return {
        "sample_id": sample.sample_id,
        "source": sample.source,
        "source_split": sample.split,
        "is_negative": sample.is_negative,
        "text": sample.text,
        "ground_truth": [entity.metadata_dict() for entity in sample.entities],
        "raw_output": raw_output,
        "json_valid": parsed.json_valid,
        "schema_valid": parsed.schema_valid,
        "parse_error": parsed.error,
        "tp": len(matched),
        "errors": errors,
    }


def analyze_errors(
    dataset_path: Path,
    prediction_path: Path,
    *,
    expected_dataset_sha256: str,
    seed: int = 42,
    samples_per_group: int = 20,
    max_review_records: int = 500,
) -> ErrorAnalysis:
    """Validate frozen inputs and group every exact FN/FP before sampling.

    Group sampling ranks IDs by SHA-256(seed, group, ID), then visits groups
    round-robin until the global cap is reached. Counts always cover all input
    records; sampling caps only affect the sensitive human-review artifact.
    """

    if (
        not isinstance(expected_dataset_sha256, str)
        or len(expected_dataset_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_dataset_sha256)
    ):
        raise EvaluationError("expected dataset SHA-256 must be 64 lowercase hexadecimal characters")
    if samples_per_group < 1 or max_review_records < 1:
        raise EvaluationError("review sample limits must be positive")

    # Reuse authoritative coverage/hash checks and explicitly reconcile every
    # sample and label, preventing drift from the existing scoring contract.
    evaluated = evaluate_predictions(
        dataset_path, prediction_path, expected_dataset_sha256=expected_dataset_sha256
    )
    predictions = load_predictions(prediction_path)
    expected_samples = {row["sample_id"]: row for row in evaluated.sample_results}
    reviewed: dict[str, dict[str, Any]] = {}
    group_members: dict[str, set[str]] = defaultdict(set)
    group_counts: Counter[str] = Counter()
    groups: dict[str, dict[str, Any]] = {}
    label_counts = {label.value: Counter() for label in EntityType}

    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sample_id = row["sample_id"]
            result = _sample_errors(row, predictions[sample_id])
            counts = Counter(error["kind"].lower() for error in result["errors"])
            if any(
                actual != expected_samples[sample_id][metric]
                for metric, actual in (("tp", result["tp"]), ("fn", counts["fn"]), ("fp", counts["fp"]))
            ):
                raise EvaluationError("error-analysis counts do not reconcile with exact evaluation")
            if not result["errors"]:
                continue
            reviewed[sample_id] = result
            for error in result["errors"]:
                kind, label, cause = error["kind"], error["label"], error["suspected_cause"]
                group = _group_key(kind, label, cause)
                group_members[group].add(sample_id)
                group_counts[group] += 1
                groups[group] = {"kind": kind, "label": label, "suspected_cause": cause}
                if label is not None:
                    label_counts[label][kind.lower()] += 1

    for label, counts in label_counts.items():
        if any(counts[kind] != evaluated.metrics["per_class"][label][kind] for kind in ("fn", "fp")):
            raise EvaluationError("per-label error counts do not reconcile with exact evaluation")
    # Catch input changes between scoring and collecting review evidence.
    if (
        sha256_file(dataset_path) != expected_dataset_sha256
        or sha256_file(prediction_path) != evaluated.metrics["predictions"]["sha256"]
    ):
        raise EvaluationError("frozen inputs changed during error analysis")

    ranked: dict[str, list[str]] = {}
    for group, members in sorted(group_members.items()):
        ranked[group] = sorted(
            members,
            key=lambda sample_id: (
                hashlib.sha256(json.dumps([seed, group, sample_id]).encode()).hexdigest(),
                sample_id,
            ),
        )[:samples_per_group]
    selected: dict[str, set[str]] = defaultdict(set)
    for rank in range(samples_per_group):
        for group, members in ranked.items():
            if rank >= len(members):
                continue
            sample_id = members[rank]
            if sample_id in selected or len(selected) < max_review_records:
                selected[sample_id].add(group)
    review_samples = tuple(
        {**reviewed[sample_id], "selection_reasons": sorted(reasons)}
        for sample_id, reasons in sorted(selected.items())
    )
    selected_group_counts = Counter(group for reasons in selected.values() for group in reasons)
    group_statistics = [
        {
            "group": group,
            **groups[group],
            "errors": group_counts[group],
            "records": len(group_members[group]),
            "selected_records": selected_group_counts[group],
        }
        for group in sorted(groups)
    ]
    statistics = {
        "schema_version": 1,
        "dataset": evaluated.metrics["dataset"],
        "predictions": evaluated.metrics["predictions"],
        "exact_metrics": evaluated.metrics,
        "totals": {
            "fn": evaluated.metrics["micro"]["fn"],
            "fp": evaluated.metrics["micro"]["fp"],
            "schema_invalid_records": sum(not row["schema_valid"] for row in evaluated.sample_results),
            "records_with_issues": len(reviewed),
            "review_records": len(review_samples),
        },
        "sampling": {
            "seed": seed,
            "samples_per_group": samples_per_group,
            "max_review_records": max_review_records,
            "algorithm": "sha256_ranked_ids_then_sorted_group_round_robin_v1",
            "unit": "documents; selection reasons are capped nominations per group",
            "unrepresented_groups": sum(row["selected_records"] == 0 for row in group_statistics),
        },
        "cause_policy": {
            "status": "heuristic; suspected causes require human review",
            "fn_precedence": ["invalid_schema", "wrong_label", "boundary_mismatch", "repeated_occurrence_omission", "missed_entity"],
            "fp_precedence": ["hallucinated_text_or_overpredicted_occurrence", "wrong_label", "boundary_mismatch", "spurious_entity"],
            "boundary_mismatch": "same-label overlapping spans with unequal boundaries",
            "wrong_label": "identical span with a different canonical label",
            "overpredicted_occurrence": "exact text exists but all assignable spans were consumed by earlier predictions, possibly under another label",
            "invalid_schema": "all gold entities count as FN and no predictions count as FP, matching the strict evaluator; SCHEMA groups count documents separately",
            "pairing": "causes are assigned independently to each FN/FP; overlapping evidence is not a one-to-one causal pairing",
        },
        "groups": group_statistics,
    }
    return ErrorAnalysis(statistics=statistics, review_samples=review_samples)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _report(statistics: dict[str, Any]) -> str:
    totals = statistics["totals"]
    lines = [
        "# PII error analysis",
        "",
        f"Analyzed {statistics['dataset']['records']} records. Exact false negatives: {totals['fn']}; exact false positives: {totals['fp']}.",
        f"Schema-invalid documents: {totals['schema_invalid_records']}. Selected review documents: {totals['review_records']}.",
        "",
        "Suspected causes are deterministic heuristics, not verified explanations. Exact counts reconcile with the frozen evaluator.",
        "A wrong label or boundary can produce both an FN and an FP. Schema issues are document counts and must not be added to entity FN/FP counts.",
        "Invalid schema discards the whole prediction under strict scoring, including potentially recognizable entities.",
        "",
        "## Groups",
        "",
        "| Kind | Label | Suspected cause | Errors | Documents | Sample nominations |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in statistics["groups"]:
        lines.append(f"| {row['kind']} | {row['label'] or 'UNASSIGNED'} | {row['suspected_cause']} | {row['errors']} | {row['records']} | {row['selected_records']} |")
    sampling = statistics["sampling"]
    lines.extend([
        "",
        "## Human review",
        "",
        "`review_samples.jsonl` contains sensitive source text, gold spans, raw outputs, and every error for each selected document. Keep it local.",
        f"Seed: {sampling['seed']}; at most {sampling['samples_per_group']} nominations per group and {sampling['max_review_records']} unique documents overall.",
        "Documents are deduplicated by sample ID and carry all selected group reasons. A selected document may incidentally contain errors from other groups.",
        f"Groups without a selected document after the global cap: {sampling['unrepresented_groups']}.",
        "Check source annotations and span boundaries before treating any heuristic group as an established model defect.",
        "",
    ])
    return "\n".join(lines)


def write_error_analysis(result: ErrorAnalysis, output_dir: Path) -> Path:
    """Write deterministic artifacts; reuse identical files, never replace others.

    Exclusive file creation plus byte-for-byte checks permits an interrupted
    identical run to finish without overwriting a completed or different run.
    """

    payloads = {
        "statistics.json": _json_bytes(result.statistics),
        "summary.md": _report(result.statistics).encode("utf-8"),
        "review_samples.jsonl": "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in result.review_samples
        ).encode("utf-8"),
    }
    manifest = {
        "schema_version": 1,
        "dataset_sha256": result.statistics["dataset"]["sha256"],
        "prediction_sha256": result.statistics["predictions"]["sha256"],
        "implementation_sha256": sha256_file(Path(__file__)),
        "sampling": result.statistics["sampling"],
        "totals": result.statistics["totals"],
        "artifacts": {
            name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "contains_source_text": name == "review_samples.jsonl"}
            for name, payload in sorted(payloads.items())
        },
    }
    payloads["inspection_manifest.json"] = _json_bytes(manifest)
    if output_dir.is_symlink():
        raise EvaluationError("error-analysis output directory must not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.name not in payloads or path.is_symlink() or not path.is_file():
            raise EvaluationError("error-analysis output directory contains unexpected entries")
        if path.read_bytes() != payloads[path.name]:
            raise EvaluationError("refusing to overwrite a different error-analysis run")
    for name, payload in payloads.items():
        path = output_dir / name
        try:
            with path.open("xb") as handle:
                handle.write(payload)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != payload:
                raise EvaluationError("refusing to overwrite a different error-analysis artifact")
    return output_dir / "inspection_manifest.json"
