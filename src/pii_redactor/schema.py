"""Shared domain types for the Local PII Redactor."""

from enum import StrEnum


class EntityType(StrEnum):
    """Canonical V1 entity labels approved for dataset inspection."""

    PERSON_NAME = "PERSON_NAME"
    EMAIL = "EMAIL"
    PHONE_NUMBER = "PHONE_NUMBER"
    ADDRESS = "ADDRESS"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    IP_ADDRESS = "IP_ADDRESS"
    USERNAME = "USERNAME"
    CREDIT_CARD = "CREDIT_CARD"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    NATIONAL_ID = "NATIONAL_ID"
    MEDICAL_ID = "MEDICAL_ID"
    CUSTOMER_ID = "CUSTOMER_ID"
    EMPLOYEE_ID = "EMPLOYEE_ID"
    VEHICLE_ID = "VEHICLE_ID"
