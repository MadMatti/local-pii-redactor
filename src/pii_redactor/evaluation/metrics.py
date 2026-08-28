"""Exact occurrence-level metrics for frozen PII extraction evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pii_redactor.data.prepared import SpanEntity, sample_from_json_record
from pii_redactor.schema import EntityType

from .parsing import PredictionEntity, parse_prediction


class EvaluationError(RuntimeError):
    """Raised when frozen evaluation inputs are incomplete or inconsistent."""


@dataclass(frozen=True)
class AlignedPrediction:
    entity: PredictionEntity
    start: int | None
    end: int | None

    @property
    def is_source_member(self) -> bool:
        return self.start is not None and self.end is not None


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, Any]
    sample_results: tuple[dict[str, Any], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _all_occurrences(text: str, value: str) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    cursor = 0
    while value and (position := text.find(value, cursor)) >= 0:
        output.append((position, position + len(value)))
        cursor = position + len(value)
    return output


def align_predictions(
    text: str, entities: Iterable[PredictionEntity]
) -> tuple[AlignedPrediction, ...]:
    """Assign repeated predictions to distinct exact source occurrences."""

    used_spans: set[tuple[int, int]] = set()
    aligned: list[AlignedPrediction] = []
    for entity in entities:
        available = [
            span
            for span in _all_occurrences(text, entity.text)
            if span not in used_spans
        ]
        if not available:
            aligned.append(AlignedPrediction(entity=entity, start=None, end=None))
            continue
        start, end = available[0]
        used_spans.add((start, end))
        aligned.append(
            AlignedPrediction(entity=entity, start=start, end=end)
        )
    return tuple(aligned)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 8),
        "recall": round(recall, 8),
        "f1": round(f1, 8),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvaluationError(f"JSONL file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise EvaluationError(f"{path}:{line_number}: record must be an object")
        rows.append(decoded)
    return rows


def load_predictions(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for line_number, row in enumerate(_read_jsonl(path), 1):
        if not {"sample_id", "raw_output"} <= set(row):
            raise EvaluationError(
                f"{path}:{line_number}: prediction needs sample_id and raw_output"
            )
        sample_id = row.get("sample_id")
        raw_output = row.get("raw_output")
        if not isinstance(sample_id, str) or not isinstance(raw_output, str):
            raise EvaluationError(
                f"{path}:{line_number}: prediction fields must be strings"
            )
        if sample_id in output:
            raise EvaluationError(f"duplicate prediction sample_id: {sample_id}")
        output[sample_id] = raw_output
    return output


def _span_key(entity: SpanEntity) -> tuple[str, int, int]:
    return (entity.type.value, entity.start, entity.end)


def _source_order_valid(aligned: tuple[AlignedPrediction, ...]) -> bool:
    if any(not item.is_source_member for item in aligned):
        return False
    starts = [item.start for item in aligned]
    return all(
        current is not None and previous is not None and current >= previous
        for previous, current in zip(starts, starts[1:])
    )


def _evaluate_sample(
    dataset_record: dict[str, Any], raw_output: str
) -> tuple[
    dict[str, Any],
    Counter[str],
    Counter[str],
    Counter[str],
    Counter[str],
    Counter[str],
    Counter[str],
]:
    sample = sample_from_json_record(dataset_record)
    parsed = parse_prediction(raw_output)
    aligned = align_predictions(sample.text, parsed.entities) if parsed.schema_valid else ()
    truth = {_span_key(entity) for entity in sample.entities}
    matched: set[tuple[str, int, int]] = set()
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    relaxed_tp: Counter[str] = Counter()
    relaxed_fp: Counter[str] = Counter()
    relaxed_fn: Counter[str] = Counter()
    for item in aligned:
        if not item.is_source_member:
            fp[item.entity.type.value] += 1
            continue
        key = (item.entity.type.value, item.start, item.end)
        if key in truth and key not in matched:
            matched.add(key)
            tp[item.entity.type.value] += 1
        else:
            fp[item.entity.type.value] += 1
    for label, _, _ in truth - matched:
        fn[label] += 1

    relaxed_matched: set[int] = set()
    for item in aligned:
        label = item.entity.type.value
        if not item.is_source_member:
            relaxed_fp[label] += 1
            continue
        candidate_index = next(
            (
                index
                for index, gold in enumerate(sample.entities)
                if index not in relaxed_matched
                and gold.type is item.entity.type
                and item.start is not None
                and item.end is not None
                and item.start < gold.end
                and gold.start < item.end
            ),
            None,
        )
        if candidate_index is None:
            relaxed_fp[label] += 1
        else:
            relaxed_matched.add(candidate_index)
            relaxed_tp[label] += 1
    for index, gold in enumerate(sample.entities):
        if index not in relaxed_matched:
            relaxed_fn[gold.type.value] += 1

    true_count = len(truth)
    predicted_count = len(aligned)
    true_positive_count = sum(tp.values())
    false_positive_count = sum(fp.values())
    false_negative_count = sum(fn.values())
    hallucinated = sum(not item.is_source_member for item in aligned)
    source_order_valid = _source_order_valid(aligned) if parsed.schema_valid else False
    complete_recall = false_negative_count == 0
    exact_document = complete_recall and false_positive_count == 0 and parsed.schema_valid
    result = {
        "sample_id": sample.sample_id,
        "source_split": sample.split,
        "ground_truth_entities": true_count,
        "predicted_entities": predicted_count,
        "tp": true_positive_count,
        "fp": false_positive_count,
        "fn": false_negative_count,
        "relaxed_tp": sum(relaxed_tp.values()),
        "relaxed_fp": sum(relaxed_fp.values()),
        "relaxed_fn": sum(relaxed_fn.values()),
        "json_valid": parsed.json_valid,
        "schema_valid": parsed.schema_valid,
        "source_order_valid": source_order_valid,
        "hallucinated_substrings": hallucinated,
        "complete_recall": complete_recall,
        "exact_document_match": exact_document,
        "is_negative": sample.is_negative,
        "parse_error": parsed.error,
    }
    return result, tp, fp, fn, relaxed_tp, relaxed_fp, relaxed_fn


def evaluate_predictions(
    dataset_path: Path,
    prediction_path: Path,
    *,
    expected_dataset_sha256: str | None = None,
) -> EvaluationResult:
    """Evaluate one complete prediction file against a frozen prepared split."""

    actual_hash = sha256_file(dataset_path)
    if expected_dataset_sha256 and actual_hash != expected_dataset_sha256:
        raise EvaluationError(
            f"dataset hash mismatch: expected {expected_dataset_sha256}, got {actual_hash}"
        )
    dataset_rows = _read_jsonl(dataset_path)
    predictions = load_predictions(prediction_path)
    dataset_ids = [row.get("sample_id") for row in dataset_rows]
    if any(not isinstance(sample_id, str) for sample_id in dataset_ids):
        raise EvaluationError("dataset contains an invalid sample_id")
    if len(dataset_ids) != len(set(dataset_ids)):
        raise EvaluationError("dataset contains duplicate sample IDs")
    missing = set(dataset_ids) - set(predictions)
    extra = set(predictions) - set(dataset_ids)
    if missing or extra:
        raise EvaluationError(
            f"prediction coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    sample_results: list[dict[str, Any]] = []
    total_tp: Counter[str] = Counter()
    total_fp: Counter[str] = Counter()
    total_fn: Counter[str] = Counter()
    total_relaxed_tp: Counter[str] = Counter()
    total_relaxed_fp: Counter[str] = Counter()
    total_relaxed_fn: Counter[str] = Counter()
    for row in dataset_rows:
        sample_id = row["sample_id"]
        result, tp, fp, fn, relaxed_tp, relaxed_fp, relaxed_fn = _evaluate_sample(
            row, predictions[sample_id]
        )
        sample_results.append(result)
        total_tp.update(tp)
        total_fp.update(fp)
        total_fn.update(fn)
        total_relaxed_tp.update(relaxed_tp)
        total_relaxed_fp.update(relaxed_fp)
        total_relaxed_fn.update(relaxed_fn)

    per_class = {
        label.value: _prf(
            total_tp[label.value], total_fp[label.value], total_fn[label.value]
        )
        for label in EntityType
    }
    micro = _prf(sum(total_tp.values()), sum(total_fp.values()), sum(total_fn.values()))
    relaxed_micro = _prf(
        sum(total_relaxed_tp.values()),
        sum(total_relaxed_fp.values()),
        sum(total_relaxed_fn.values()),
    )
    macro = {
        metric: round(
            sum(per_class[label.value][metric] for label in EntityType)
            / len(EntityType),
            8,
        )
        for metric in ("precision", "recall", "f1")
    }
    pii_records = [row for row in sample_results if not row["is_negative"]]
    negative_records = [row for row in sample_results if row["is_negative"]]
    metrics = {
        "schema_version": 1,
        "dataset": {
            "path": str(dataset_path),
            "sha256": actual_hash,
            "records": len(dataset_rows),
        },
        "predictions": {
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
        },
        "micro": micro,
        "relaxed_overlap_micro": relaxed_micro,
        "macro": macro,
        "per_class": per_class,
        "documents": {
            "pii_records": len(pii_records),
            "complete_document_recall": round(
                _safe_ratio(
                    sum(row["complete_recall"] for row in pii_records),
                    len(pii_records),
                ),
                8,
            ),
            "exact_document_match": round(
                _safe_ratio(
                    sum(row["exact_document_match"] for row in sample_results),
                    len(sample_results),
                ),
                8,
            ),
            "pii_free_records": len(negative_records),
            "pii_free_false_positive_rate": round(
                _safe_ratio(
                    sum(row["predicted_entities"] > 0 for row in negative_records),
                    len(negative_records),
                ),
                8,
            ),
        },
        "validity": {
            "json_valid_rate": round(
                _safe_ratio(
                    sum(row["json_valid"] for row in sample_results),
                    len(sample_results),
                ),
                8,
            ),
            "schema_valid_rate": round(
                _safe_ratio(
                    sum(row["schema_valid"] for row in sample_results),
                    len(sample_results),
                ),
                8,
            ),
            "source_order_valid_rate": round(
                _safe_ratio(
                    sum(row["source_order_valid"] for row in sample_results),
                    len(sample_results),
                ),
                8,
            ),
            "hallucinated_substrings": sum(
                row["hallucinated_substrings"] for row in sample_results
            ),
        },
    }
    return EvaluationResult(metrics=metrics, sample_results=tuple(sample_results))


def write_evaluation(
    result: EvaluationResult, output_dir: Path, *, title: str
) -> tuple[Path, Path, Path]:
    """Write machine-readable metrics, error analysis, and a compact report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    samples_path = output_dir / "sample_results.jsonl"
    report_path = output_dir / "report.md"
    metrics_path.write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    samples_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in result.sample_results
        ),
        encoding="utf-8",
    )
    micro = result.metrics["micro"]
    documents = result.metrics["documents"]
    validity = result.metrics["validity"]
    lines = [
        f"# {title}",
        "",
        f"- Records: {result.metrics['dataset']['records']}",
        f"- Micro precision: {micro['precision']:.4f}",
        f"- Micro recall: {micro['recall']:.4f}",
        f"- Micro F1: {micro['f1']:.4f}",
        f"- Relaxed overlap F1: {result.metrics['relaxed_overlap_micro']['f1']:.4f}",
        f"- Complete-document recall: {documents['complete_document_recall']:.4f}",
        f"- PII-free false-positive rate: {documents['pii_free_false_positive_rate']:.4f}",
        f"- JSON validity: {validity['json_valid_rate']:.4f}",
        f"- Schema validity: {validity['schema_valid_rate']:.4f}",
        f"- Hallucinated substrings: {validity['hallucinated_substrings']}",
        "",
        "## Per-class exact metrics",
        "",
        "| Label | Precision | Recall | F1 | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in EntityType:
        item = result.metrics["per_class"][label.value]
        lines.append(
            f"| {label.value} | {item['precision']:.4f} | {item['recall']:.4f} | "
            f"{item['f1']:.4f} | {item['tp']} | {item['fp']} | {item['fn']} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics_path, samples_path, report_path
