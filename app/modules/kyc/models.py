import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    Index,
    JSON,
    String,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, validates

from app.infrastructure.database import Base
from app.modules.graph.models import validate_pseudonymized_hash
from app.modules.identity.models import ensure_utc, utcnow


class KYCApplication(Base):
    """
    Authoritative record for an onboarding KYC application.
    Contains customer intake metadata, artifact references, and screening verdicts.
    All sensitive identifiers are stored as salted cryptographic HMAC-SHA256 digests.
    """
    __tablename__ = "kyc_applications"
    __table_args__ = (
        Index("ix_kyc_applications_applicant_id", "applicant_id", unique=True),
        Index("ix_kyc_applications_identity_hash", "identity_hash"),
        Index("ix_kyc_applications_phone_hash", "phone_hash"),
        Index("ix_kyc_applications_email_hash", "email_hash"),
        Index("ix_kyc_applications_device_hash", "device_fingerprint_hash"),
        Index("ix_kyc_applications_status_created", "status", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    applicant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    account_type: Mapped[str] = mapped_column(String(32), nullable=False, default="savings")
    id_type: Mapped[str] = mapped_column(String(32), nullable=False, default="aadhaar")

    # Name and DOB for forensic OCR matching
    claimed_name: Mapped[str] = mapped_column(String(128), nullable=False)
    claimed_dob: Mapped[str] = mapped_column(String(32), nullable=False)
    id_number_last4: Mapped[str] = mapped_column(String(8), nullable=False)

    # Pseudonymized Identifier Hashes (HMAC-SHA256 hex digests)
    identity_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    phone_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    email_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    device_fingerprint_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    ip_address_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Storage URIs for document scans and selfies (outside DB rows)
    document_artifact_uri: Mapped[str] = mapped_column(String(512), nullable=False)
    selfie_artifact_uri: Mapped[str] = mapped_column(String(512), nullable=False)
    document_mime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    selfie_mime_type: Mapped[str] = mapped_column(String(64), nullable=False)

    # Ingestion Screening Verdict
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    risk_tier: Mapped[str] = mapped_column(String(32), nullable=False, default="UNKNOWN")
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    reasons_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    @validates("identity_hash", "phone_hash", "email_hash", "device_fingerprint_hash", "ip_address_hash")
    def _validate_hashes(self, key: str, value: str | None) -> str | None:
        return validate_pseudonymized_hash(value, key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "applicant_id": self.applicant_id,
            "account_type": self.account_type,
            "id_type": self.id_type,
            "claimed_name": self.claimed_name,
            "claimed_dob": self.claimed_dob,
            "id_number_last4": self.id_number_last4,
            "identity_hash": self.identity_hash,
            "phone_hash": self.phone_hash,
            "email_hash": self.email_hash,
            "device_fingerprint_hash": self.device_fingerprint_hash,
            "ip_address_hash": self.ip_address_hash,
            "document_artifact_uri": self.document_artifact_uri,
            "selfie_artifact_uri": self.selfie_artifact_uri,
            "document_mime_type": self.document_mime_type,
            "selfie_mime_type": self.selfie_mime_type,
            "status": self.status,
            "risk_tier": self.risk_tier,
            "risk_score": self.risk_score,
            "reasons": list(self.reasons_json) if self.reasons_json else [],
            "metadata": dict(self.metadata_json) if self.metadata_json else {},
            "created_at": ensure_utc(self.created_at).isoformat() if self.created_at else None,
            "updated_at": ensure_utc(self.updated_at).isoformat() if self.updated_at else None,
        }
