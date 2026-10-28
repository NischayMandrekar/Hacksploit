import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.orm import Session

from app.core.errors import KYCValidationError
from app.infrastructure.database import get_db
from app.infrastructure.storage.artifact_store import ArtifactStore, LocalDiskArtifactStore
from app.modules.kyc.schemas import (
    KYCApplicationCreate,
    KYCApplicationDetail,
    KYCApplicationReceipt,
)
from app.modules.kyc.service import KYCApplicationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kyc", tags=["KYC Onboarding & Verification"])


def get_artifact_store() -> ArtifactStore:
    """Dependency provider for binary artifact storage."""
    return LocalDiskArtifactStore()


def get_kyc_service(
    db: Session = Depends(get_db),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
) -> KYCApplicationService:
    return KYCApplicationService(db=db, artifact_store=artifact_store)


@router.post(
    "/applications",
    response_model=KYCApplicationReceipt,
    status_code=status.HTTP_201_CREATED,
    summary="Submit KYC Onboarding Application",
    description="Accepts applicant demographic details, government ID, live selfie, and device fingerprint via multipart form.",
)
async def submit_application(
    request: Request,
    document_file: UploadFile = File(..., description="Government ID document scan or photo"),
    selfie_file: UploadFile = File(..., description="Applicant live selfie or webcam capture"),
    legal_name: str = Form(..., description="Full legal name"),
    dob: str = Form(..., description="Date of birth in YYYY-MM-DD"),
    phone: str = Form(..., description="Mobile phone number"),
    email: str = Form(..., description="Email address"),
    id_number: str = Form(..., description="Government identity document number"),
    device_fingerprint: str = Form(..., description="Client device / browser canvas fingerprint"),
    account_type: str = Form("savings", description="Requested account type"),
    id_type: str = Form("aadhaar", description="ID document type (e.g. aadhaar, pan, passport)"),
    ip_address: str | None = Form(None, description="Client IP address"),
    client_metadata: str | None = Form(None, description="Optional JSON telemetry string"),
    applicant_id: str | None = Form(None, description="Optional custom or idempotent applicant ID"),
    service: KYCApplicationService = Depends(get_kyc_service),
) -> KYCApplicationReceipt:
    request_id = getattr(request.state, "request_id", "unknown")

    # Parse optional client metadata JSON
    parsed_metadata: dict[str, Any] = {}
    if client_metadata:
        try:
            parsed_metadata = json.loads(client_metadata)
            if not isinstance(parsed_metadata, dict):
                raise ValueError("client_metadata must be a JSON object.")
        except Exception as e:
            raise KYCValidationError(
                f"Invalid client_metadata format: {e}",
                code="invalid_client_metadata",
            ) from e

    # Capture IP address from request headers if not explicitly supplied
    if not ip_address:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            ip_address = forwarded_for.split(",")[0].strip()
        elif request.client:
            ip_address = request.client.host

    # Validate input form schema
    try:
        payload = KYCApplicationCreate(
            account_type=account_type,
            legal_name=legal_name,
            dob=dob,
            phone=phone,
            email=email,
            id_type=id_type,
            id_number=id_number,
            device_fingerprint=device_fingerprint,
            ip_address=ip_address,
            client_metadata=parsed_metadata,
        )
    except PydanticValidationError as e:
        first_error = e.errors()[0]
        field_name = ".".join(str(loc) for loc in first_error["loc"])
        msg = f"Validation failed for field '{field_name}': {first_error['msg']}"
        sanitized_errors = [
            {
                "field": ".".join(str(loc) for loc in err["loc"]),
                "message": err["msg"],
                "type": err["type"],
            }
            for err in e.errors()
        ]
        raise KYCValidationError(
            message=msg,
            code="schema_validation_error",
            details={"errors": sanitized_errors},
        ) from e

    # Read binary files
    doc_bytes = await document_file.read()
    selfie_bytes = await selfie_file.read()

    doc_mime = document_file.content_type or "application/octet-stream"
    selfie_mime = selfie_file.content_type or "application/octet-stream"

    return service.submit_application(
        data=payload,
        document_bytes=doc_bytes,
        document_mime_type=doc_mime,
        selfie_bytes=selfie_bytes,
        selfie_mime_type=selfie_mime,
        applicant_id=applicant_id,
        request_id=request_id,
    )


@router.get(
    "/applications/{applicant_id}",
    response_model=KYCApplicationDetail,
    summary="Get KYC Application Status & Screening Details",
)
def get_application_detail(
    applicant_id: str,
    service: KYCApplicationService = Depends(get_kyc_service),
) -> KYCApplicationDetail:
    app = service.get_application(applicant_id)
    return KYCApplicationDetail(**app.to_dict())


@router.get(
    "/applications",
    response_model=list[KYCApplicationDetail],
    summary="List KYC Applications (Paginated)",
)
def list_applications(
    skip: int = 0,
    limit: int = 50,
    service: KYCApplicationService = Depends(get_kyc_service),
) -> list[KYCApplicationDetail]:
    apps = service.list_applications(skip=skip, limit=limit)
    return [KYCApplicationDetail(**app.to_dict()) for app in apps]
