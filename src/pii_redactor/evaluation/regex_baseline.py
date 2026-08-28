"""Conservative deterministic regex baseline for syntactic PII classes."""

from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Pattern

from pii_redactor.schema import EntityType


REGEX_BASELINE_VERSION = "regex-baseline-v1"


@dataclass(frozen=True, order=True)
class RegexDetection:
    start: int
    end: int
    type: EntityType
    text: str


@dataclass(frozen=True)
class PatternRule:
    entity_type: EntityType
    pattern: Pattern[str]
    group: int = 0
    predicate: Callable[[str], bool] | None = None


def _valid_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value.strip("[]"))
    except ValueError:
        return False
    return True


def _luhn_or_formatted(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, character in enumerate(digits):
        digit = int(character)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0 or " " in value


def _valid_username(value: str) -> bool:
    return value.casefold() not in {"caption", "field", "input", "placeholder"}


_RULES = (
    PatternRule(
        EntityType.EMAIL,
        re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])"),
    ),
    PatternRule(
        EntityType.IP_ADDRESS,
        re.compile(
            r"(?<![\w:])(?:\d{1,3}(?:\.\d{1,3}){3}|"
            r"[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{0,4}){2,7})(?![\w:])"
        ),
        predicate=_valid_ip,
    ),
    PatternRule(
        EntityType.BANK_ACCOUNT,
        re.compile(r"(?<!\w)[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}(?!\w)"),
    ),
    PatternRule(
        EntityType.CREDIT_CARD,
        re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
        predicate=_luhn_or_formatted,
    ),
    PatternRule(
        EntityType.NATIONAL_ID,
        re.compile(
            r"(?<!\w)(?:NAT-[A-Z0-9-]{4,}|\d{3}-\d{2}-\d{4})(?!\w)",
            re.IGNORECASE,
        ),
    ),
    PatternRule(
        EntityType.MEDICAL_ID,
        re.compile(r"(?<!\w)(?:MED|MRN)[- :][A-Z0-9-]{4,}(?!\w)", re.IGNORECASE),
    ),
    PatternRule(
        EntityType.CUSTOMER_ID,
        re.compile(r"(?<!\w)(?:CUS|CUSTOMER)[- :][A-Z0-9-]{4,}(?!\w)", re.IGNORECASE),
    ),
    PatternRule(
        EntityType.EMPLOYEE_ID,
        re.compile(r"(?<!\w)(?:EMP|EMPLOYEE)[- :][A-Z0-9-]{3,}(?!\w)", re.IGNORECASE),
    ),
    PatternRule(
        EntityType.VEHICLE_ID,
        re.compile(
            r"(?<!\w)(?:[A-HJ-NPR-Z0-9]{17}|[A-Z]{1,3}-[A-Z0-9-]{3,10})(?!\w)"
        ),
    ),
    PatternRule(
        EntityType.DATE_OF_BIRTH,
        re.compile(
            r"(?:date of birth|birth date|\bDOB\b|born)\s*(?:is|:|was)?\s*"
            r"(\d{1,4}[./-]\d{1,2}[./-]\d{1,4})",
            re.IGNORECASE,
        ),
        group=1,
    ),
    PatternRule(
        EntityType.USERNAME,
        re.compile(
            r"(?:username|user name)\s*(?:(?:is|:|=)\s*)?"
            r"([A-Za-z0-9_.-]{3,64})",
            re.IGNORECASE,
        ),
        group=1,
        predicate=_valid_username,
    ),
    PatternRule(
        EntityType.ADDRESS,
        re.compile(
            r"(?<!\w)\d{1,6}\s+[A-Za-z][A-Za-z .'-]{1,60}\s+"
            r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Boulevard|Blvd|Drive|Dr)\b",
            re.IGNORECASE,
        ),
    ),
    PatternRule(
        EntityType.PHONE_NUMBER,
        re.compile(r"(?<![\w\d])\+?\d(?:[ ()-]*\d){7,14}(?!\d)"),
    ),
)


def _overlaps(candidate: RegexDetection, accepted: Iterable[RegexDetection]) -> bool:
    return any(
        candidate.start < existing.end and existing.start < candidate.end
        for existing in accepted
    )


def regex_detect(text: str) -> tuple[RegexDetection, ...]:
    """Return non-overlapping detections in exact source order."""

    accepted: list[RegexDetection] = []
    for rule in _RULES:
        for match in rule.pattern.finditer(text):
            start, end = match.span(rule.group)
            value = text[start:end]
            if rule.predicate and not rule.predicate(value):
                continue
            candidate = RegexDetection(
                start=start,
                end=end,
                type=rule.entity_type,
                text=value,
            )
            if not _overlaps(candidate, accepted):
                accepted.append(candidate)
    return tuple(sorted(accepted))


def regex_response(text: str) -> str:
    detections = regex_detect(text)
    return json.dumps(
        {
            "entities": [
                {"type": detection.type.value, "text": detection.text}
                for detection in detections
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
