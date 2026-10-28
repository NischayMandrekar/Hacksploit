from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from app.core.forensics import EvidenceStatus


class RiskTier(StrEnum):
    """Calibrated operational risk tier."""
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


@dataclass
class EvidenceAttribution:
    """Detailed attribution for an evaluated evidence signal."""
    rule_id: str
    signal_name: str
    severity: str  # "REJECT", "REVIEW", "INFO"
    score_impact: float
    description: str
    evidence_status: EvidenceStatus = EvidenceStatus.PRESENT
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "signal_name": self.signal_name,
            "severity": self.severity,
            "score_impact": self.score_impact,
            "description": self.description,
            "evidence_status": self.evidence_status.value,
            "details": self.details,
        }


@dataclass
class RiskAssessment:
    """Standard output contract for all fraud detection models (M0-M3)."""
    applicant_id: str
    model_name: str
    risk_score: float  # Calibrated probability in [0.0, 1.0]
    risk_tier: RiskTier
    reason_codes: list[str]
    attributions: list[EvidenceAttribution]
    missing_signals: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicant_id": self.applicant_id,
            "model_name": self.model_name,
            "risk_score": round(self.risk_score, 4),
            "risk_tier": self.risk_tier.value,
            "reason_codes": self.reason_codes,
            "attributions": [a.to_dict() for a in self.attributions],
            "missing_signals": self.missing_signals,
            "execution_time_ms": round(self.execution_time_ms, 3),
            "summary": self.summary,
        }


@dataclass
class ApplicationEvidence:
    """
    Unified multi-source evidence container.
    Aggregates tabular applicant inputs, relational graph topology,
    document forensics, image integrity, and behavioral velocity.
    """
    applicant_id: str

    # 1. Claimed Demographics
    claimed_name: str | None = None
    claimed_dob: str | None = None
    id_type: str | None = None
    id_number_last4: str | None = None

    # 2. Relational Graph Metrics
    shared_phone_degree: int = 1
    shared_device_degree: int = 1
    shared_identity_degree: int = 0
    shared_email_degree: int = 0
    has_confirmed_fraud_neighbor: bool = False
    fraud_neighbor_count: int = 0
    ego_density: float = 0.0
    cycle_count: float = 0.0

    # 3. Document Forensics & Checksums
    document_extracted_name: str | None = None
    document_extracted_dob: str | None = None
    ocr_name_similarity: float | None = None
    dob_mismatch_years: float | None = None
    id_checksum_valid: bool | None = None
    secure_qr_status: str = "UNAVAILABLE"  # "VERIFIED", "TAMPERED", "UNAVAILABLE"
    secure_qr_valid: bool | None = None

    # 4. Media Forensics & AI Generation
    image_dimension_status: EvidenceStatus = EvidenceStatus.UNAVAILABLE
    exif_status: EvidenceStatus = EvidenceStatus.UNAVAILABLE
    jpeg_quality_status: EvidenceStatus = EvidenceStatus.UNAVAILABLE
    ela_status: EvidenceStatus = EvidenceStatus.UNAVAILABLE
    ela_max_diff: float | None = None
    fft_status: EvidenceStatus = EvidenceStatus.UNAVAILABLE
    fft_grid_ratio: float | None = None
    face_similarity_status: EvidenceStatus = EvidenceStatus.UNAVAILABLE
    face_cosine_similarity: float | None = None

    # 5. Behavioral Velocity
    velocity_1h_applications: int = 1
    velocity_24h_applications: int = 1
    device_velocity_1h: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicant_id": self.applicant_id,
            "claimed_name": self.claimed_name,
            "claimed_dob": self.claimed_dob,
            "id_type": self.id_type,
            "id_number_last4": self.id_number_last4,
            "shared_phone_degree": self.shared_phone_degree,
            "shared_device_degree": self.shared_device_degree,
            "shared_identity_degree": self.shared_identity_degree,
            "shared_email_degree": self.shared_email_degree,
            "has_confirmed_fraud_neighbor": self.has_confirmed_fraud_neighbor,
            "fraud_neighbor_count": self.fraud_neighbor_count,
            "ego_density": self.ego_density,
            "cycle_count": self.cycle_count,
            "ocr_name_similarity": self.ocr_name_similarity,
            "dob_mismatch_years": self.dob_mismatch_years,
            "id_checksum_valid": self.id_checksum_valid,
            "secure_qr_status": self.secure_qr_status,
            "secure_qr_valid": self.secure_qr_valid,
            "image_dimension_status": self.image_dimension_status.value,
            "exif_status": self.exif_status.value,
            "jpeg_quality_status": self.jpeg_quality_status.value,
            "ela_status": self.ela_status.value,
            "fft_status": self.fft_status.value,
            "face_similarity_status": self.face_similarity_status.value,
            "face_cosine_similarity": self.face_cosine_similarity,
            "velocity_1h_applications": self.velocity_1h_applications,
            "velocity_24h_applications": self.velocity_24h_applications,
            "device_velocity_1h": self.device_velocity_1h,
        }


class FraudModel(Protocol):
    """
    Protocol for all fraud prediction models (M0 Rules, M1 Tabular, M2 Graph, M3 Fusion).
    ARCHITECTURE.md §2:
    Guarantee: Returns calibrated risk_score, risk_tier, and evidence_attributions.
    """

    def predict_risk(self, features: ApplicationEvidence) -> RiskAssessment:
        """Evaluates multi-source evidence and produces an explainable, calibrated risk assessment."""
        ...
