"""Typed representations of Gretel source records."""

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, field_validator


class SourceEntity(BaseModel):
    """One unmodified entity annotation from the source dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    entity: str
    types: tuple[str, ...]

    @field_validator("entity")
    @classmethod
    def validate_entity(cls, value: str) -> str:
        if not value:
            raise ValueError("entity must be a nonempty string")
        return value

    @field_validator("types")
    @classmethod
    def validate_types(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("types must contain at least one source label")
        if any(not item for item in value):
            raise ValueError("types must contain only nonempty strings")
        return value


class SourceRecord(BaseModel):
    """The documented source columns after safe entity parsing."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    uid: str
    domain: str
    document_type: str
    document_description: str
    text: str
    entities: tuple[SourceEntity, ...]

    @field_validator(
        "uid",
        "domain",
        "document_type",
        "document_description",
        "text",
    )
    @classmethod
    def validate_string_fields(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("source record fields must be strings")
        return value

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SourceRecord":
        """Create a record while deliberately ignoring documented extra columns."""

        from .source import parse_entities

        return cls(
            uid=row["uid"],
            domain=row["domain"],
            document_type=row["document_type"],
            document_description=row["document_description"],
            text=row["text"],
            entities=parse_entities(row["entities"]),
        )
