import json

import pytest

from pii_redactor.evaluation.parsing import parse_prediction
from pii_redactor.schema import EntityType


def test_parse_prediction_accepts_only_frozen_shape() -> None:
    parsed = parse_prediction(
        '{"entities":[{"type":"EMAIL","text":"ada@example.net"}]}'
    )
    assert parsed.json_valid
    assert parsed.schema_valid
    assert parsed.entities[0].type is EntityType.EMAIL


@pytest.mark.parametrize(
    ("raw", "json_valid"),
    [
        ("```json\n{\"entities\":[]}\n```", False),
        ('{"entities":[],"explanation":"none"}', True),
        ('{"entities":[{"type":"UNKNOWN","text":"x"}]}', True),
        ('{"entities":[{"type":"EMAIL","text":""}]}', True),
    ],
)
def test_parse_prediction_rejects_repairs_extras_unknowns_and_empty_text(
    raw: str, json_valid: bool
) -> None:
    parsed = parse_prediction(raw)
    assert parsed.json_valid is json_valid
    assert not parsed.schema_valid
    assert parsed.entities == ()
