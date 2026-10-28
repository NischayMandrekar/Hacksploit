from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


@dataclass
class FieldContradictionReport:
    """Detailed report comparing claimed application demographics against extracted document fields."""
    name_match_score: float
    dob_match_score: float
    id_checksum_valid: bool
    id_match: bool
    extracted_name: str | None = None
    extracted_dob: str | None = None
    extracted_id_number: str | None = None
    claimed_name: str | None = None
    claimed_dob: str | None = None
    claimed_id_number: str | None = None
    contradictions_detected: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def has_contradiction(self) -> bool:
        return len(self.contradictions_detected) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name_match_score": round(self.name_match_score, 4),
            "dob_match_score": round(self.dob_match_score, 4),
            "id_checksum_valid": self.id_checksum_valid,
            "id_match": self.id_match,
            "extracted_name": self.extracted_name,
            "extracted_dob": self.extracted_dob,
            "extracted_id_number": self.extracted_id_number,
            "claimed_name": self.claimed_name,
            "claimed_dob": self.claimed_dob,
            "claimed_id_number": self.claimed_id_number,
            "contradictions_detected": self.contradictions_detected,
            "has_contradiction": self.has_contradiction,
            "reasons": self.reasons,
        }


@dataclass
class SecureQRVerificationResult:
    """Outcome of offline RSA-2048 digital signature verification on Aadhaar Secure QR codes."""
    status: str  # "VERIFIED", "TAMPERED", "UNAVAILABLE", "INVALID_FORMAT"
    is_valid: bool
    reason_codes: list[str] = field(default_factory=list)
    extracted_demographics: dict[str, str] = field(default_factory=dict)
    signature_algorithm: str = "RSA-2048-SHA256"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "is_valid": self.is_valid,
            "reason_codes": self.reason_codes,
            "extracted_demographics": self.extracted_demographics,
            "signature_algorithm": self.signature_algorithm,
            "details": self.details,
        }


class DocumentForensicAssessment(BaseModel):
    """Pydantic model representing forensic assessment for an applicant document."""
    applicant_id: str
    document_type: str = "aadhaar"
    has_contradictions: bool = Field(default=False)
    contradiction_reasons: list[str] = Field(default_factory=list)
    name_similarity: float = Field(default=1.0)
    dob_match: bool = Field(default=True)
    id_checksum_valid: bool = Field(default=True)
    secure_qr_status: str = Field(default="UNAVAILABLE")
    reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)
