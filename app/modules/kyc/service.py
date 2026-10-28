import logging
import uuid
from typing import Any
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import (
    ApplicantNotFoundError,
    ArtifactStorageError,
    DuplicateApplicationError,
)
from app.core.security import hash_pii
from app.infrastructure.jobs.registry import ExecutionContext, default_registry
from app.infrastructure.jobs.service import enqueue_job
from app.infrastructure.storage.artifact_store import ArtifactStore, LocalDiskArtifactStore
from app.modules.forensics.ocr import FieldContradictionDetector, OCRExtractor
from app.modules.forensics.qr import AadhaarSecureQRVerifier
from app.modules.graph.models import ApplicantEdge, ApplicantNode, EdgeType, NodeType
from app.modules.graph.rules import M0RuleEngine
from app.modules.kyc.models import KYCApplication
from app.modules.kyc.schemas import (
    KYCApplicationCreate,
    KYCApplicationReceipt,
)

logger = logging.getLogger(__name__)


# Register baseline handler for async document forensics if not already present
def _default_document_forensics_handler(
    payload: dict[str, Any], context: ExecutionContext
) -> dict[str, Any]:
    context.check()
    applicant_id = payload.get("applicant_id", "unknown")
    doc_uri = payload.get("document_artifact_uri")
    claimed_name = payload.get("claimed_name", "")
    claimed_dob = payload.get("claimed_dob", "")
    id_type = payload.get("id_type", "aadhaar")

    logger.info("Executing async document forensics task", extra={"applicant_id": applicant_id})

    # 1. Load document artifact if URI present
    doc_bytes = b""
    if doc_uri:
        try:
            store = LocalDiskArtifactStore()
            doc_bytes = store.load_artifact(doc_uri)
        except Exception as e:
            logger.warning("Could not load document artifact %s: %s", doc_uri, str(e))

    # 2. Offline Secure QR verification
    qr_verifier = AadhaarSecureQRVerifier()
    qr_result = qr_verifier.verify_from_image(doc_bytes) if doc_bytes else None

    # 3. OCR and Field contradiction detection
    ocr_extractor = OCRExtractor()
    extracted_doc = ocr_extractor.extract_fields(doc_bytes, "image/jpeg") if doc_bytes else None
    extracted_fields = dict(extracted_doc.fields) if extracted_doc else {}

    if qr_result and qr_result.is_valid:
        extracted_fields.update(qr_result.extracted_demographics)

    contradiction_detector = FieldContradictionDetector()
    report = contradiction_detector.evaluate(
        claimed_name=claimed_name,
        claimed_dob=claimed_dob,
        claimed_id_number=payload.get("claimed_id_number", ""),
        extracted_fields=extracted_fields,
        id_type=id_type,
    )

    return {
        "applicant_id": applicant_id,
        "forensics_status": "completed",
        "secure_qr_status": qr_result.status if qr_result else "UNAVAILABLE",
        "has_contradictions": report.has_contradiction,
        "contradictions": report.contradictions_detected,
        "reasons": report.reasons,
    }


if "kyc.document_forensics" not in default_registry._handlers:
    default_registry.register("kyc.document_forensics", _default_document_forensics_handler)


class KYCApplicationService:
    """
    Core domain service orchestrating synchronous KYC ingestion:
    1. Validates intake payloads and binary artifact headers/magic bytes.
    2. Writes binary document and selfie images to ArtifactStore.
    3. Derives salted HMAC-SHA256 digests for all PII.
    4. Persists the applicant node and typed edges in the relational identity graph.
    5. Evaluates synchronous 1-hop screening rules (M0) in < 15ms.
    6. Enqueues asynchronous forensic extraction in the background lease queue.
    """

    def __init__(
        self,
        db: Session,
        artifact_store: ArtifactStore,
        rule_engine: M0RuleEngine | None = None,
    ) -> None:
        self.db = db
        self.artifact_store = artifact_store
        self.rule_engine = rule_engine or M0RuleEngine(db)

    def submit_application(
        self,
        data: KYCApplicationCreate,
        document_bytes: bytes,
        document_mime_type: str,
        selfie_bytes: bytes,
        selfie_mime_type: str,
        applicant_id: str | None = None,
        request_id: str = "unknown",
    ) -> KYCApplicationReceipt:
        """
        Processes a multipart KYC application submission.
        """
        # 1. Determine or validate applicant_id
        app_id = applicant_id or f"app_{uuid.uuid4().hex[:12]}"
        
        # Check for duplicate submission
        existing = self.db.execute(
            select(KYCApplication).where(KYCApplication.applicant_id == app_id)
        ).scalar_one_or_none()
        if existing:
            raise DuplicateApplicationError(app_id)

        # 2. Validate binary file constraints
        if not document_bytes or len(document_bytes) == 0:
            raise ArtifactStorageError("Document artifact cannot be empty.", code="empty_document")
        if not selfie_bytes or len(selfie_bytes) == 0:
            raise ArtifactStorageError("Selfie artifact cannot be empty.", code="empty_selfie")

        if len(document_bytes) > settings.artifact_max_bytes:
            raise ArtifactStorageError(
                f"Document size ({len(document_bytes)} bytes) exceeds maximum allowed limit ({settings.artifact_max_bytes} bytes).",
                code="document_too_large",
            )
        if len(selfie_bytes) > settings.artifact_max_bytes:
            raise ArtifactStorageError(
                f"Selfie size ({len(selfie_bytes)} bytes) exceeds maximum allowed limit ({settings.artifact_max_bytes} bytes).",
                code="selfie_too_large",
            )

        # 3. Store binary artifacts in ArtifactStore (verifies MIME and magic bytes)
        doc_uri = self.artifact_store.save_artifact(
            applicant_id=app_id,
            artifact_type="identity_document",
            data=document_bytes,
            mime_type=document_mime_type,
        )
        selfie_uri = self.artifact_store.save_artifact(
            applicant_id=app_id,
            artifact_type="applicant_selfie",
            data=selfie_bytes,
            mime_type=selfie_mime_type,
        )

        # 4. Cryptographic PII pseudonymization (salted HMAC-SHA256)
        identity_hash = hash_pii(data.id_number, is_numeric_id=True)
        phone_hash = hash_pii(data.phone, is_numeric_id=True)
        email_hash = hash_pii(data.email)
        device_hash = hash_pii(data.device_fingerprint)
        ip_hash = hash_pii(data.ip_address) if data.ip_address else None
        id_last4 = data.id_number.strip()[-4:]

        # 5. Populate Relational Identity Graph (Node & Typed Edges)
        applicant_node = ApplicantNode(
            applicant_id=app_id,
            node_type=NodeType.APPLICANT.value,
            identity_hash=identity_hash,
            phone_hash=phone_hash,
            email_hash=email_hash,
            device_fingerprint_hash=device_hash,
            ip_address_hash=ip_hash,
            status="pending",
            risk_score=None,
            is_fraud=None,
            properties_json={
                "account_type": data.account_type,
                "id_type": data.id_type,
            },
        )
        self.db.add(applicant_node)

        # Add graph edges
        edges = [
            ApplicantEdge(
                source_hash=app_id,
                target_hash=identity_hash,
                edge_type=EdgeType.HAS_IDENTITY.value,
            ),
            ApplicantEdge(
                source_hash=app_id,
                target_hash=phone_hash,
                edge_type=EdgeType.HAS_PHONE.value,
            ),
            ApplicantEdge(
                source_hash=app_id,
                target_hash=email_hash,
                edge_type=EdgeType.HAS_EMAIL.value,
            ),
            ApplicantEdge(
                source_hash=app_id,
                target_hash=device_hash,
                edge_type=EdgeType.HAS_DEVICE.value,
            ),
        ]
        if ip_hash:
            edges.append(
                ApplicantEdge(
                    source_hash=app_id,
                    target_hash=ip_hash,
                    edge_type=EdgeType.HAS_IP.value,
                )
            )

        for edge in edges:
            self.db.add(edge)

        # Flush so the new edges are visible to M0 queries in the same transaction
        self.db.flush()

        # 6. Evaluate Fast-Path 1-Hop Rules (M0 Screening)
        rule_eval = self.rule_engine.evaluate_applicant(
            applicant_id=app_id,
            identifiers={
                "identity_number": data.id_number,
                "phone": data.phone,
                "email": data.email,
                "device_fingerprint": data.device_fingerprint,
            },
        )

        risk_tier = rule_eval.risk_tier
        risk_score = rule_eval.risk_score
        reason_codes = rule_eval.reason_codes

        if risk_tier == "REJECT":
            app_status = "rejected"
            message = "Application rejected by automated screening rules."
        elif risk_tier == "REVIEW":
            app_status = "under_review"
            message = "Application flagged for manual fraud review."
        else:
            app_status = "screening"
            message = "Application received and passed initial screening; awaiting document forensics."

        applicant_node.status = app_status
        applicant_node.risk_score = risk_score

        # 7. Persist KYC Application Record
        application = KYCApplication(
            applicant_id=app_id,
            account_type=data.account_type,
            id_type=data.id_type,
            claimed_name=data.legal_name.strip(),
            claimed_dob=data.dob.strip(),
            id_number_last4=id_last4,
            identity_hash=identity_hash,
            phone_hash=phone_hash,
            email_hash=email_hash,
            device_fingerprint_hash=device_hash,
            ip_address_hash=ip_hash,
            document_artifact_uri=doc_uri,
            selfie_artifact_uri=selfie_uri,
            document_mime_type=document_mime_type,
            selfie_mime_type=selfie_mime_type,
            status=app_status,
            risk_tier=risk_tier,
            risk_score=risk_score,
            reasons_json=reason_codes,
            metadata_json=data.client_metadata or {},
        )
        self.db.add(application)

        # 8. Asynchronous Dual-Plane Enqueueing (PostgreSQL Lease Queue)
        try:
            if "kyc.document_forensics" in default_registry._handlers:
                enqueue_job(
                    db=self.db,
                    job_type="kyc.document_forensics",
                    payload={
                        "applicant_id": app_id,
                        "document_artifact_uri": doc_uri,
                        "selfie_artifact_uri": selfie_uri,
                        "claimed_name": data.legal_name.strip(),
                        "claimed_dob": data.dob.strip(),
                        "id_type": data.id_type,
                    },
                    request_id=request_id,
                )
        except Exception as e:
            # Enqueueing failure should not abort the application intake transaction
            logger.warning(
                "Failed to enqueue async document forensics job",
                extra={"applicant_id": app_id, "error": str(e)},
            )

        self.db.commit()
        self.db.refresh(application)

        return KYCApplicationReceipt(
            applicant_id=app_id,
            account_type=application.account_type,
            status=application.status,
            risk_tier=application.risk_tier,
            risk_score=application.risk_score,
            reason_codes=application.reasons_json,
            document_artifact_uri=application.document_artifact_uri,
            selfie_artifact_uri=application.selfie_artifact_uri,
            created_at=application.created_at.isoformat(),
            message=message,
        )

    def get_application(self, applicant_id: str) -> KYCApplication:
        """Retrieves a single KYC application record by applicant_id."""
        app = self.db.execute(
            select(KYCApplication).where(KYCApplication.applicant_id == applicant_id)
        ).scalar_one_or_none()
        if not app:
            raise ApplicantNotFoundError(applicant_id)
        return app

    def list_applications(self, skip: int = 0, limit: int = 50) -> list[KYCApplication]:
        """Lists KYC applications paginated."""
        return list(
            self.db.execute(
                select(KYCApplication)
                .order_by(KYCApplication.created_at.desc())
                .offset(skip)
                .limit(min(limit, 100))
            )
            .scalars()
            .all()
        )
