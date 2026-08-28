"""Frozen exact-match evaluation and deterministic baselines."""

from .metrics import EvaluationError, EvaluationResult, evaluate_predictions
from .parsing import ParsedPrediction, PredictionEntity, parse_prediction

__all__ = [
    "EvaluationError",
    "EvaluationResult",
    "ParsedPrediction",
    "PredictionEntity",
    "evaluate_predictions",
    "parse_prediction",
]
