from pii_redactor.data.synthetic import (
    generate_negative_samples,
    generate_positive_samples,
)
from pii_redactor.schema import EntityType


def test_negative_generation_is_deterministic_and_contains_long_and_hard_cases() -> None:
    first = generate_negative_samples(split="train", count=100, seed=42)
    second = generate_negative_samples(split="train", count=100, seed=42)
    assert [sample.to_json_record() for sample in first] == [
        sample.to_json_record() for sample in second
    ]
    assert all(sample.is_negative for sample in first)
    assert {sample.synthetic_kind for sample in first} >= {
        "long_negative",
        "hard_negative",
    }


def test_positive_generation_covers_every_label_and_exact_occurrence() -> None:
    samples = generate_positive_samples(split="valid", count=140, seed=42)
    assert {entity.type for sample in samples for entity in sample.entities} == set(
        EntityType
    )
    for sample in samples:
        for entity in sample.entities:
            assert sample.text[entity.start : entity.end] == entity.text
    repeated = [sample for sample in samples if sample.synthetic_kind == "repeated_positive"]
    assert repeated
    assert all(len(sample.entities) == 3 for sample in repeated)


def test_template_families_are_split_isolated() -> None:
    train = generate_positive_samples(split="train", count=50, seed=42)
    valid = generate_positive_samples(split="valid", count=50, seed=42)
    assert {sample.template_family for sample in train}.isdisjoint(
        {sample.template_family for sample in valid}
    )
