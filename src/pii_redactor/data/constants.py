"""Constants for the Gretel source dataset and candidate V1 policy."""

from pathlib import Path

from pii_redactor.schema import EntityType

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DATASET_ID = "gretelai/gretel-pii-masking-en-v1"
DEFAULT_REQUESTED_REVISION = "main"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "huggingface"
DEFAULT_SNAPSHOT_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "gretel-pii-masking-en-v1"
)
DEFAULT_LOCK_FILE = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "gretel-pii-masking-en-v1"
    / "source-lock.json"
)
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "models" / "base-qwen3-1.7b-4bit"

EXPECTED_SPLIT_COUNTS = {
    "train": 50_000,
    "validation": 5_000,
    "test": 5_000,
}
REQUIRED_COLUMNS = {
    "uid",
    "domain",
    "document_type",
    "document_description",
    "entities",
    "text",
}

SOURCE_LABEL_MAPPING: dict[str, EntityType] = {
    "first_name": EntityType.PERSON_NAME,
    "last_name": EntityType.PERSON_NAME,
    "name": EntityType.PERSON_NAME,
    "email": EntityType.EMAIL,
    "phone_number": EntityType.PHONE_NUMBER,
    "street_address": EntityType.ADDRESS,
    "address": EntityType.ADDRESS,
    "date_of_birth": EntityType.DATE_OF_BIRTH,
    "ipv4": EntityType.IP_ADDRESS,
    "ipv6": EntityType.IP_ADDRESS,
    "user_name": EntityType.USERNAME,
    "credit_card_number": EntityType.CREDIT_CARD,
    "account_number": EntityType.BANK_ACCOUNT,
    "bank_routing_number": EntityType.BANK_ACCOUNT,
    "swift_bic": EntityType.BANK_ACCOUNT,
    "ssn": EntityType.NATIONAL_ID,
    "national_id": EntityType.NATIONAL_ID,
    "tax_id": EntityType.NATIONAL_ID,
    "medical_record_number": EntityType.MEDICAL_ID,
    "health_plan_beneficiary_number": EntityType.MEDICAL_ID,
    "customer_id": EntityType.CUSTOMER_ID,
    "employee_id": EntityType.EMPLOYEE_ID,
    "license_plate": EntityType.VEHICLE_ID,
    "vehicle_identifier": EntityType.VEHICLE_ID,
}

EXCLUDED_SOURCE_LABELS = frozenset(
    {
        "date",
        "date_time",
        "time",
        "city",
        "state",
        "country",
        "postcode",
        "coordinate",
        "company_name",
        "url",
        "device_identifier",
        "unique_identifier",
        "biometric_identifier",
        "certificate_license_number",
        "api_key",
        "password",
        "cvv",
        "pin",
    }
)

KNOWN_SOURCE_LABELS = frozenset(SOURCE_LABEL_MAPPING) | EXCLUDED_SOURCE_LABELS
if len(KNOWN_SOURCE_LABELS) != 42:  # pragma: no cover - import-time invariant
    raise RuntimeError("The Gretel source-label policy must classify 42 labels")

CANDIDATE_SYSTEM_PROMPT = """You are a PII detection model.

Identify personally identifiable information explicitly present in the provided text.

Return only JSON matching this structure:

{"entities":[{"type":"ENTITY_TYPE","text":"EXACT_SOURCE_TEXT"}]}

Allowed entity types:
PERSON_NAME, EMAIL, PHONE_NUMBER, ADDRESS, DATE_OF_BIRTH,
IP_ADDRESS, USERNAME, CREDIT_CARD, BANK_ACCOUNT, NATIONAL_ID,
MEDICAL_ID, CUSTOMER_ID, EMPLOYEE_ID, VEHICLE_ID.

Rules:
- Copy entity text exactly from the input.
- Do not normalize or rewrite entities.
- Include every occurrence.
- Return entities in source order.
- Do not infer information that is not explicitly present.
- If no PII is present, return {"entities":[]}.
- Do not provide explanations."""

INSPECTION_SEED = 42
TRUNCATION_THRESHOLDS = (768, 1024, 1280)

POLICY_VERSION = "pii-taxonomy-v1"
PROMPT_VERSION = "pii-extraction-v1"
DATASET_BUILD_VERSION = "prepared-dataset-v1"
DATASET_BUILD_SEED = 42

DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "data" / "calibration"

MAIN_STAGE_COUNTS = {"train": 20_000, "valid": 2_000, "test": 2_000}
V0_STAGE_COUNTS = {"train": 5_000, "valid": 500, "test": 500}
SMOKE_STAGE_COUNTS = {"train": 500, "valid": 100, "test": 100}
TARGET_NEGATIVE_RATIO = 0.25
