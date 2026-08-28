"""Strict parsing for model-generated PII responses."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pii_redactor.schema import EntityType


@dataclass(frozen=True)
class PredictionEntity:
    type: EntityType
    text: str


@dataclass(frozen=True)
class ParsedPrediction:
    raw_output: str
    json_valid: bool
    schema_valid: bool
    entities: tuple[PredictionEntity, ...]
    error: str | None = None


def parse_prediction(raw_output: str) -> ParsedPrediction:
    """Parse only the frozen response schema; do not repair model output."""

    if not isinstance(raw_output, str):
        return ParsedPrediction(
            raw_output="",
            json_valid=False,
            schema_valid=False,
            entities=(),
            error="raw output must be a string",
        )
    try:
        decoded = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        return ParsedPrediction(
            raw_output=raw_output,
            json_valid=False,
            schema_valid=False,
            entities=(),
            error=f"invalid JSON: {exc.msg}",
        )
    if not isinstance(decoded, dict) or set(decoded) != {"entities"}:
        return ParsedPrediction(
            raw_output=raw_output,
            json_valid=True,
            schema_valid=False,
            entities=(),
            error="response must be an object containing only 'entities'",
        )
    raw_entities = decoded["entities"]
    if not isinstance(raw_entities, list):
        return ParsedPrediction(
            raw_output=raw_output,
            json_valid=True,
            schema_valid=False,
            entities=(),
            error="entities must be a list",
        )
    entities: list[PredictionEntity] = []
    for index, item in enumerate(raw_entities):
        if not isinstance(item, dict) or set(item) != {"type", "text"}:
            return ParsedPrediction(
                raw_output=raw_output,
                json_valid=True,
                schema_valid=False,
                entities=(),
                error=f"entity {index} must contain only type and text",
            )
        if not isinstance(item["text"], str) or not item["text"]:
            return ParsedPrediction(
                raw_output=raw_output,
                json_valid=True,
                schema_valid=False,
                entities=(),
                error=f"entity {index} text must be a nonempty string",
            )
        try:
            entity_type = EntityType(item["type"])
        except (TypeError, ValueError):
            return ParsedPrediction(
                raw_output=raw_output,
                json_valid=True,
                schema_valid=False,
                entities=(),
                error=f"entity {index} contains an unknown type",
            )
        entities.append(PredictionEntity(type=entity_type, text=item["text"]))
    return ParsedPrediction(
        raw_output=raw_output,
        json_valid=True,
        schema_valid=True,
        entities=tuple(entities),
    )
