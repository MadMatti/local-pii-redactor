from pii_redactor.data.builder import _select_stage
from pii_redactor.data.synthetic import (
    generate_negative_samples,
    generate_positive_samples,
)
from pii_redactor.schema import EntityType


def test_stage_selection_is_deterministic_balanced_and_label_complete() -> None:
    candidates = generate_positive_samples(split="train", count=420, seed=42)
    candidates += generate_negative_samples(split="train", count=180, seed=42)
    first = _select_stage(
        candidates, total=400, minimum_per_label=2, key="unit-stage"
    )
    second = _select_stage(
        candidates, total=400, minimum_per_label=2, key="unit-stage"
    )
    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
    assert sum(sample.is_negative for sample in first) == 100
    assert {entity.type for sample in first for entity in sample.entities} == set(
        EntityType
    )
