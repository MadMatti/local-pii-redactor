"""Canonical prepared samples and deterministic source-span conversion."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from pii_redactor.schema import EntityType

from .constants import (
    CANDIDATE_SYSTEM_PROMPT,
    POLICY_VERSION,
    PROMPT_VERSION,
)
from .models import SourceRecord
from .source import classify_source_types


class PreparationError(ValueError):
    """Raised when exact deterministic preparation is not possible."""


@dataclass(frozen=True, order=True)
class SpanEntity:
    """One canonical entity anchored to the immutable source text."""

    start: int
    end: int
    type: EntityType
    text: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise PreparationError("entity span must be positive and nonempty")
        if self.end - self.start != len(self.text):
            raise PreparationError("entity span length does not match entity text")

    def response_dict(self) -> dict[str, str]:
        return {"type": self.type.value, "text": self.text}

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "text": self.text,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class PreparedSample:
    """One MLX-compatible chat example with evaluation metadata."""

    sample_id: str
    text: str
    entities: tuple[SpanEntity, ...]
    source: str
    split: str
    source_uid: str | None = None
    template_family: str | None = None
    synthetic_kind: str | None = None

    @property
    def is_negative(self) -> bool:
        return not self.entities

    @property
    def assistant_content(self) -> str:
        return json.dumps(
            {"entities": [entity.response_dict() for entity in self.entities]},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @property
    def entity_types(self) -> frozenset[EntityType]:
        return frozenset(entity.type for entity in self.entities)

    def to_json_record(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source": self.source,
            "source_split": self.split,
            "is_negative": self.is_negative,
            "policy_version": POLICY_VERSION,
            "prompt_version": PROMPT_VERSION,
            "entity_spans": [entity.metadata_dict() for entity in self.entities],
        }
        if self.source_uid is not None:
            metadata["source_uid"] = self.source_uid
        if self.template_family is not None:
            metadata["template_family"] = self.template_family
        if self.synthetic_kind is not None:
            metadata["synthetic_kind"] = self.synthetic_kind
        return {
            "sample_id": self.sample_id,
            "messages": [
                {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT},
                {"role": "user", "content": self.text},
                {"role": "assistant", "content": self.assistant_content},
            ],
            "metadata": metadata,
        }


def _all_occurrences(text: str, value: str) -> list[int]:
    positions: list[int] = []
    cursor = 0
    while True:
        position = text.find(value, cursor)
        if position < 0:
            return positions
        positions.append(position)
        cursor = position + len(value)


def _validate_nonoverlapping(entities: Iterable[SpanEntity]) -> None:
    ordered = sorted(entities)
    for previous, current in zip(ordered, ordered[1:]):
        if current.start < previous.end:
            raise PreparationError(
                "canonical source entities overlap: "
                f"{previous.type.value}@{previous.start}:{previous.end} and "
                f"{current.type.value}@{current.start}:{current.end}"
            )


def source_record_to_sample(
    record: SourceRecord,
    *,
    split: str,
) -> PreparedSample:
    """Filter, align, and source-order one Gretel record exactly."""

    annotations_by_text: dict[str, list[EntityType]] = defaultdict(list)
    for annotation in record.entities:
        classification = classify_source_types(annotation.types)
        if classification.unknown_labels:
            raise PreparationError(
                f"unknown source labels: {classification.unknown_labels}"
            )
        if classification.is_ambiguous:
            raise PreparationError(
                "source entity maps to multiple canonical types: "
                f"{classification.canonical_types}"
            )
        if not classification.canonical_types:
            continue
        annotations_by_text[annotation.entity].append(
            classification.canonical_types[0]
        )

    spans: list[SpanEntity] = []
    for entity_text, canonical_types in annotations_by_text.items():
        positions = _all_occurrences(record.text, entity_text)
        if len(positions) != len(canonical_types):
            raise PreparationError(
                f"annotation/occurrence mismatch for {entity_text!r}: "
                f"annotations={len(canonical_types)}, occurrences={len(positions)}"
            )
        if len(set(canonical_types)) > 1:
            raise PreparationError(
                "repeated exact text has conflicting canonical labels: "
                f"{entity_text!r}"
            )
        for position, canonical_type in zip(positions, canonical_types, strict=True):
            spans.append(
                SpanEntity(
                    start=position,
                    end=position + len(entity_text),
                    type=canonical_type,
                    text=entity_text,
                )
            )

    spans.sort()
    _validate_nonoverlapping(spans)
    return PreparedSample(
        sample_id=f"gretel:{split}:{record.uid}",
        text=record.text,
        entities=tuple(spans),
        source="gretel",
        split=split,
        source_uid=record.uid,
    )


def synthetic_sample_id(
    *,
    split: str,
    template_family: str,
    variant: int,
    text: str,
) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"synthetic:{PROMPT_VERSION}:{split}:{template_family}:{variant:05d}:{digest}"


def sample_from_text_and_values(
    *,
    split: str,
    template_family: str,
    variant: int,
    synthetic_kind: str,
    text: str,
    values: Iterable[tuple[EntityType, str]],
) -> PreparedSample:
    """Create a synthetic sample by anchoring every declared occurrence."""

    spans: list[SpanEntity] = []
    for entity_type, value in values:
        positions = _all_occurrences(text, value)
        if not positions:
            raise PreparationError(
                f"synthetic entity {value!r} is absent from template {template_family}"
            )
        for position in positions:
            spans.append(
                SpanEntity(
                    start=position,
                    end=position + len(value),
                    type=entity_type,
                    text=value,
                )
            )
    unique_spans = sorted(set(spans))
    _validate_nonoverlapping(unique_spans)
    return PreparedSample(
        sample_id=synthetic_sample_id(
            split=split,
            template_family=template_family,
            variant=variant,
            text=text,
        ),
        text=text,
        entities=tuple(unique_spans),
        source="synthetic",
        split=split,
        template_family=template_family,
        synthetic_kind=synthetic_kind,
    )


def sample_from_json_record(record: Mapping[str, Any]) -> PreparedSample:
    """Reconstruct a prepared sample from emitted JSONL for tests and tools."""

    messages = record["messages"]
    metadata = record["metadata"]
    spans = tuple(
        SpanEntity(
            start=item["start"],
            end=item["end"],
            type=EntityType(item["type"]),
            text=item["text"],
        )
        for item in metadata["entity_spans"]
    )
    return PreparedSample(
        sample_id=record["sample_id"],
        text=messages[1]["content"],
        entities=spans,
        source=metadata["source"],
        split=metadata["source_split"],
        source_uid=metadata.get("source_uid"),
        template_family=metadata.get("template_family"),
        synthetic_kind=metadata.get("synthetic_kind"),
    )
