"""Safe parsing and diagnostics for Gretel source annotations."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from pii_redactor.schema import EntityType

from .constants import EXCLUDED_SOURCE_LABELS, SOURCE_LABEL_MAPPING
from .models import SourceEntity


class EntityParseError(ValueError):
    """Raised when a source `entities` value is not safely parseable."""


@dataclass(frozen=True)
class SourceTypeClassification:
    """Classification of one source entity's type list."""

    canonical_types: tuple[EntityType, ...]
    included_labels: tuple[str, ...]
    excluded_labels: tuple[str, ...]
    unknown_labels: tuple[str, ...]

    @property
    def is_ambiguous(self) -> bool:
        return len(self.canonical_types) > 1


def _decode_entities(value: Any) -> Any:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        raise EntityParseError("entities must be a nonempty string or a list")

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise EntityParseError("entities is neither JSON nor a Python literal") from exc


def parse_entities(value: Any) -> tuple[SourceEntity, ...]:
    """Parse source annotations without evaluating executable Python."""

    decoded = _decode_entities(value)
    if not isinstance(decoded, list):
        raise EntityParseError("entities must decode to a list")

    parsed: list[SourceEntity] = []
    for index, item in enumerate(decoded):
        if not isinstance(item, dict):
            raise EntityParseError(f"entity at index {index} must be an object")
        if set(item) != {"entity", "types"}:
            raise EntityParseError(
                f"entity at index {index} must contain only 'entity' and 'types'"
            )
        raw_types = item["types"]
        if not isinstance(raw_types, (list, tuple)):
            raise EntityParseError(f"types at index {index} must be a list")
        try:
            parsed.append(
                SourceEntity(entity=item["entity"], types=tuple(raw_types))
            )
        except (ValidationError, TypeError) as exc:
            raise EntityParseError(f"invalid entity at index {index}: {exc}") from exc
    return tuple(parsed)


def classify_source_types(types: tuple[str, ...]) -> SourceTypeClassification:
    """Map source labels to the candidate V1 policy without repairing labels."""

    included: list[str] = []
    excluded: list[str] = []
    unknown: list[str] = []
    canonical: list[EntityType] = []

    for source_type in types:
        if source_type in SOURCE_LABEL_MAPPING:
            included.append(source_type)
            mapped = SOURCE_LABEL_MAPPING[source_type]
            if mapped not in canonical:
                canonical.append(mapped)
        elif source_type in EXCLUDED_SOURCE_LABELS:
            excluded.append(source_type)
        else:
            unknown.append(source_type)

    return SourceTypeClassification(
        canonical_types=tuple(canonical),
        included_labels=tuple(included),
        excluded_labels=tuple(excluded),
        unknown_labels=tuple(unknown),
    )


def count_exact_occurrences(text: str, value: str) -> int:
    """Count non-overlapping exact occurrences without normalizing source text."""

    if not value:
        return 0
    count = 0
    start = 0
    while True:
        position = text.find(value, start)
        if position < 0:
            return count
        count += 1
        start = position + len(value)


def normalize_for_duplicate(value: str) -> str:
    """Normalize only for duplicate diagnostics, never for training output."""

    return re.sub(r"\s+", " ", value).strip().casefold()
