import re
from datetime import date
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
PHONE_DIGITS_REGEX = re.compile(r"^[0-9]{7,15}$")


class KYCApplicationCreate(BaseModel):
    """
    Input schema for applicant onboarding.
    Submitted via multipart form data alongside document and selfie files.
    """
    account_type: str = Field(
        default="savings",
        description="Type of account requested (e.g., savings, current, credit_card)",
    )
    legal_name: str = Field(
        ...,
        min_length=2,
        max_length=128,
        description="Full legal name as printed on government identification",
    )
    dob: str = Field(
        ...,
        description="Date of birth in YYYY-MM-DD format",
    )
    phone: str = Field(
        ...,
        min_length=7,
        max_length=20,
        description="Applicant mobile phone number",
    )
    email: str = Field(
        ...,
        min_length=5,
        max_length=128,
        description="Applicant email address",
    )
    id_type: str = Field(
        default="aadhaar",
        description="Type of government ID presented (e.g. aadhaar, pan, passport, voter_id, ssn)",
    )
    id_number: str = Field(
        ...,
        min_length=4,
        max_length=64,
        description="Government identification document number",
    )
    device_fingerprint: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Client device/browser fingerprint identifier",
    )
    ip_address: str | None = Field(
        default=None,
        description="Client IPv4 or IPv6 network address",
    )
    client_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional client telemetry (user agent, screen dimensions, language)",
    )

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        clean = v.strip().lower()
        if not EMAIL_REGEX.match(clean):
            raise ValueError(f"Invalid email address format: '{v}'")
        return clean

    @field_validator("phone")
    @classmethod
    def validate_phone_format(cls, v: str) -> str:
        digits = re.sub(r"[\s\-\(\)\+]", "", v)
        if not PHONE_DIGITS_REGEX.match(digits):
            raise ValueError(f"Phone number must contain between 7 and 15 digits: '{v}'")
        return v.strip()

    @field_validator("dob")
    @classmethod
    def validate_dob_format(cls, v: str) -> str:
        try:
            parsed = date.fromisoformat(v.strip())
            if parsed > date.today():
                raise ValueError("Date of birth cannot be in the future.")
            if parsed.year < 1900:
                raise ValueError("Date of birth year must be 1900 or later.")
        except Exception as e:
            raise ValueError(f"DOB must be in valid YYYY-MM-DD format: {e}") from e
        return v.strip()


class KYCApplicationReceipt(BaseModel):
    """
    Immediate synchronous response returned to the applicant upon successful submission.
    """
    applicant_id: str
    account_type: str
    status: str
    risk_tier: str
    risk_score: float | None
    reason_codes: list[str]
    document_artifact_uri: str
    selfie_artifact_uri: str
    created_at: str
    message: str

    model_config = ConfigDict(from_attributes=True)


class KYCApplicationDetail(BaseModel):
    """
    Comprehensive KYC application details viewable by analysts and auditing systems.
    Zero raw identity numbers, emails, or phone numbers are exposed; only pseudonymized hashes and last 4.
    """
    id: str
    applicant_id: str
    account_type: str
    id_type: str
    claimed_name: str
    claimed_dob: str
    id_number_last4: str
    identity_hash: str
    phone_hash: str
    email_hash: str
    device_fingerprint_hash: str
    ip_address_hash: str | None
    document_artifact_uri: str
    selfie_artifact_uri: str
    document_mime_type: str
    selfie_mime_type: str
    status: str
    risk_tier: str
    risk_score: float | None
    reasons: list[str]
    metadata: dict[str, Any]
    created_at: str | None
    updated_at: str | None

    model_config = ConfigDict(from_attributes=True)
