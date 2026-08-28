"""Deterministic, split-isolated synthetic additions for dataset V1."""

from __future__ import annotations

import random
from collections.abc import Iterable

from pii_redactor.schema import EntityType

from .prepared import PreparedSample, sample_from_text_and_values


_SPLIT_WORD = {"train": "amber", "valid": "cobalt", "test": "silver"}
_RARE_SOURCE_LABELS = {
    EntityType.BANK_ACCOUNT,
    EntityType.CREDIT_CARD,
    EntityType.USERNAME,
    EntityType.VEHICLE_ID,
}

_NEGATIVE_PATTERNS = {
    "train": (
        "The project team reviewed the quarterly roadmap and assigned follow-up actions.",
        "A blank username field appears beside the save button; no user value was entered.",
        "The guide explains that an IPv4 header contains source and destination fields.",
        "The address section is optional and was deliberately left empty.",
        "The report was approved in April, but it contains no birth date.",
        "The account-number label is visible while the corresponding value remains blank.",
    ),
    "valid": (
        "Reviewers discussed the release checklist and left implementation notes for tomorrow.",
        "The login form shows a username caption with an empty input control.",
        "This chapter compares IPv4 and IPv6 packet-header terminology without listing an address.",
        "The mailing-address box has not been completed.",
        "A meeting occurred during autumn; no person's birth date is stated.",
        "The bank-account field is disabled and contains no value.",
    ),
    "test": (
        "The operations group summarized routine maintenance tasks and closed the review.",
        "A username placeholder is displayed, although nobody supplied an identifier.",
        "The note describes IP routing concepts without including a network address.",
        "The postal-address heading is present but its field is empty.",
        "The event happened in winter and does not disclose anybody's date of birth.",
        "The payment account entry was intentionally left unfilled.",
    ),
}

_LONG_FILLER = {
    "train": (
        "The working group examined process quality, documentation clarity, change control, "
        "and routine operational checks. Each section records general observations, open "
        "questions, and nonpersonal recommendations for the next review cycle."
    ),
    "valid": (
        "The assessment describes ordinary planning, equipment care, version tracking, and "
        "team coordination. Its paragraphs contain general guidance and deliberately omit "
        "details about individual people or accounts."
    ),
    "test": (
        "The handbook covers standard workflows, quality gates, issue triage, and release "
        "preparation. The material is procedural and contains no information tied to a "
        "specific person, device, vehicle, or financial record."
    ),
}


def _marker(split: str, variant: int) -> str:
    """Return a benign uniqueness marker without phone-like punctuation."""

    return f"Reference {_SPLIT_WORD[split]}-{variant:05d}."


def _long_text(split: str, variant: int, *, insertion: str = "") -> str:
    filler = _LONG_FILLER[split]
    paragraphs = [filler for _ in range(18)]
    if insertion:
        paragraphs[(variant * 7) % len(paragraphs)] = insertion
    return f"{_marker(split, variant)}\n\n" + "\n\n".join(paragraphs)


def generate_negative_samples(
    *, split: str, count: int, seed: int
) -> list[PreparedSample]:
    """Generate simple, hard, and long PII-free records deterministically."""

    if split not in _NEGATIVE_PATTERNS:
        raise ValueError(f"unsupported split: {split}")
    rng = random.Random(f"negative:{seed}:{split}")
    hard_indices = {1, 2, 3, 4, 5}
    samples: list[PreparedSample] = []
    long_count = count * 15 // 100
    for variant in range(count):
        if variant < long_count:
            kind = "long_negative"
            family = f"{split}:negative:long:{variant % 5}"
            text = _long_text(split, variant)
        else:
            pattern_index = rng.randrange(len(_NEGATIVE_PATTERNS[split]))
            kind = (
                "hard_negative"
                if pattern_index in hard_indices
                else "simple_negative"
            )
            family = f"{split}:negative:{kind}:{pattern_index}"
            text = f"{_NEGATIVE_PATTERNS[split][pattern_index]} {_marker(split, variant)}"
        samples.append(
            sample_from_text_and_values(
                split=split,
                template_family=family,
                variant=variant,
                synthetic_kind=kind,
                text=text,
                values=(),
            )
        )
    return samples


def _value(entity_type: EntityType, split: str, variant: int) -> str:
    split_code = {"train": 17, "valid": 43, "test": 71}[split]
    number = variant + split_code * 10_000
    short = number % 90_000 + 10_000
    values = {
        EntityType.PERSON_NAME: f"Avery Northwood {split_code}-{variant}",
        EntityType.EMAIL: f"avery.{split}.{variant}@example.net",
        EntityType.PHONE_NUMBER: f"+41 44 {short // 100:03d} {short % 100:02d} 10",
        EntityType.ADDRESS: f"{variant + 10} {_SPLIT_WORD[split].title()} Meadow Lane",
        EntityType.DATE_OF_BIRTH: (
            f"{(variant % 27) + 1:02d}.{(variant % 12) + 1:02d}."
            f"{1960 + variant % 40}"
        ),
        EntityType.IP_ADDRESS: f"10.{split_code}.{variant // 250 % 250}.{variant % 250 + 1}",
        EntityType.USERNAME: f"{_SPLIT_WORD[split]}_user_{variant:05d}",
        EntityType.CREDIT_CARD: (
            f"{4000 + split_code:04d} {variant % 10_000:04d} "
            f"{(variant * 7) % 10_000:04d} {(variant * 13) % 10_000:04d}"
        ),
        EntityType.BANK_ACCOUNT: f"CH{split_code:02d} 0900 0000 {number:09d}",
        EntityType.NATIONAL_ID: f"NAT-{split_code}-{variant:07d}",
        EntityType.MEDICAL_ID: f"MED-{split_code}-{variant:07d}",
        EntityType.CUSTOMER_ID: f"CUS-{split_code}-{variant:07d}",
        EntityType.EMPLOYEE_ID: f"EMP-{split_code}-{variant:07d}",
        EntityType.VEHICLE_ID: f"ZH-{split_code}-{variant:06d}",
    }
    return values[entity_type]


def _positive_sentence(
    entity_type: EntityType, value: str, split: str, variant: int
) -> str:
    labels = {
        EntityType.PERSON_NAME: "The registered contact is",
        EntityType.EMAIL: "Send the confirmation to",
        EntityType.PHONE_NUMBER: "The callback number is",
        EntityType.ADDRESS: "Deliver the sealed packet to",
        EntityType.DATE_OF_BIRTH: "The recorded date of birth is",
        EntityType.IP_ADDRESS: "The connection originated from",
        EntityType.USERNAME: "The active username is",
        EntityType.CREDIT_CARD: "The card number on the form is",
        EntityType.BANK_ACCOUNT: "The designated bank account is",
        EntityType.NATIONAL_ID: "The national identifier is",
        EntityType.MEDICAL_ID: "The medical record identifier is",
        EntityType.CUSTOMER_ID: "The customer identifier is",
        EntityType.EMPLOYEE_ID: "The employee identifier is",
        EntityType.VEHICLE_ID: "The vehicle identifier is",
    }
    return f"{labels[entity_type]} {value}. {_marker(split, variant)}"


def generate_positive_samples(
    *, split: str, count: int, seed: int
) -> list[PreparedSample]:
    """Generate split-isolated label coverage, repeats, and long contexts."""

    if split not in _NEGATIVE_PATTERNS:
        raise ValueError(f"unsupported split: {split}")
    labels = list(EntityType)
    rng = random.Random(f"positive:{seed}:{split}")
    samples: list[PreparedSample] = []
    for variant in range(count):
        entity_type = labels[variant % len(labels)]
        value = _value(entity_type, split, variant)
        mode = variant % 10
        if mode == 0:
            kind = "repeated_positive"
            family = f"{split}:positive:repeated:{entity_type.value}"
            text = (
                f"The first record gives {value}. A later audit repeats {value}, and the "
                f"final confirmation again lists {value}. {_marker(split, variant)}"
            )
        elif mode == 1:
            kind = "long_positive"
            family = f"{split}:positive:long:{entity_type.value}"
            insertion = _positive_sentence(entity_type, value, split, variant)
            text = _long_text(split, variant, insertion=insertion)
        else:
            kind = (
                "rare_positive"
                if entity_type in _RARE_SOURCE_LABELS
                else "standard_positive"
            )
            family = f"{split}:positive:standard:{entity_type.value}:{mode}"
            text = _positive_sentence(entity_type, value, split, variant)
        samples.append(
            sample_from_text_and_values(
                split=split,
                template_family=family,
                variant=variant,
                synthetic_kind=kind,
                text=text,
                values=((entity_type, value),),
            )
        )
    rng.shuffle(samples)
    return samples


def generate_calibration_texts(*, count: int, seed: int) -> Iterable[str]:
    """Yield a balanced corpus independent of the frozen test split."""

    positives = generate_positive_samples(split="train", count=count // 2, seed=seed + 1)
    negatives = generate_negative_samples(
        split="train", count=count - len(positives), seed=seed + 1
    )
    combined = [sample.text for sample in positives + negatives]
    random.Random(f"calibration:{seed}").shuffle(combined)
    return combined
