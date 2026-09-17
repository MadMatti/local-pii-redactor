#!/usr/bin/env python3
"""Compare frozen predictions with explicit recall and validity regression gates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pii_redactor.evaluation.metrics import EvaluationError, evaluate_predictions, load_predictions
from pii_redactor.evaluation.parsing import parse_prediction


def compare(dataset, reference, candidate, expected_hash, *, recall_tolerance=0.0,
            document_tolerance=0.0):
    for value in (recall_tolerance, document_tolerance):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise EvaluationError("tolerances must be finite fractions between zero and one")
    ref = evaluate_predictions(dataset, reference, expected_dataset_sha256=expected_hash)
    cand = evaluate_predictions(dataset, candidate, expected_dataset_sha256=expected_hash)
    paths = [("micro", "recall", recall_tolerance),
             ("documents", "complete_document_recall", document_tolerance),
             ("validity", "schema_valid_rate", 0.0)]
    gates = []
    for section, metric, allowed in paths:
        left, right = ref.metrics[section][metric], cand.metrics[section][metric]
        gates.append({"metric": metric, "reference": left, "candidate": right,
                      "delta": round(right-left, 8), "maximum_drop": allowed,
                      "passed": right + allowed + 1e-8 >= left})
    reference_outputs, candidate_outputs = load_predictions(reference), load_predictions(candidate)
    reference_samples = {r["sample_id"]: r for r in ref.sample_results}
    changes = []
    for row in cand.sample_results:
        sample_id = row["sample_id"]
        left, right = reference_outputs[sample_id], candidate_outputs[sample_id]
        if left == right:
            continue
        parsed_left, parsed_right = parse_prediction(left), parse_prediction(right)
        equivalent = (parsed_left.schema_valid and parsed_right.schema_valid
                      and parsed_left.entities == parsed_right.entities)
        previous = reference_samples[sample_id]
        changes.append({"sample_id": sample_id, "equivalent_entity_sequence": equivalent,
                        "tp_delta": row["tp"]-previous["tp"],
                        "fp_delta": row["fp"]-previous["fp"],
                        "fn_delta": row["fn"]-previous["fn"],
                        "schema_valid_before": previous["schema_valid"],
                        "schema_valid_after": row["schema_valid"]})
    return {
        "schema_version": 1,
        "status": "passed" if all(g["passed"] for g in gates) else "review_required",
        "dataset": ref.metrics["dataset"], "gates": gates,
        "reference_metrics": ref.metrics, "candidate_metrics": cand.metrics,
        "output_changes": {"raw_changed_records": len(changes),
                           "entity_sequence_changed_records": sum(not r["equivalent_entity_sequence"] for r in changes),
                           "records": changes},
        "per_class_deltas": {
            label: {metric: round(cand.metrics["per_class"][label][metric] - values[metric], 8)
                    for metric in ("precision", "recall", "f1", "tp", "fp", "fn")}
            for label, values in ref.metrics["per_class"].items()
        },
        "interpretation": "Task-score gate, not bitwise numerical equivalence or deployment approval. Review changed outputs and per-class deltas.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-recall-drop", type=float, default=0.0)
    parser.add_argument("--maximum-document-recall-drop", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        output = args.output_dir.resolve()
        allowed = (PROJECT_ROOT / "evaluation/results").resolve()
        if output == allowed or not output.is_relative_to(allowed) or output.is_symlink():
            raise EvaluationError("parity reports must stay under local evaluation/results")
        result = compare(args.dataset, args.reference_predictions, args.candidate_predictions,
                         args.expected_dataset_sha256, recall_tolerance=args.maximum_recall_drop,
                         document_tolerance=args.maximum_document_recall_drop)
        serialized = json.dumps(result, sort_keys=True, indent=2) + "\n"
        report = output / "parity.json"
        if output.exists() and (set(p.name for p in output.iterdir()) != {"parity.json"}
                                or report.is_symlink() or report.read_text() != serialized):
            raise EvaluationError("refusing to replace different parity evidence")
        output.mkdir(parents=True, exist_ok=True)
        report.write_text(serialized)
    except (EvaluationError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Parity check failed ({type(exc).__name__}); inspect frozen inputs and output path.", file=sys.stderr)
        return 1
    print(f"Parity status: {result['status']}")
    print(f"Changed outputs: {result['output_changes']['raw_changed_records']}")
    print(f"Report: {report}")
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
