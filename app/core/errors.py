import logging
from typing import Any
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class PlatformError(Exception):
    """Base domain exception with HTTP status, machine error code, and message."""
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ArtifactStorageError(PlatformError):
    """Raised when an uploaded document or selfie fails validation or storage."""
    def __init__(self, message: str, code: str = "artifact_storage_error", status_code: int = 400):
        super().__init__(status_code=status_code, code=code, message=message)


class KYCValidationError(PlatformError):
    """Raised when submitted KYC application data fails business format or sanity rules."""
    def __init__(
        self,
        message: str,
        code: str = "kyc_validation_error",
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(status_code=status_code, code=code, message=message, details=details)


class ApplicantNotFoundError(PlatformError):
    """Raised when an applicant_id is not found in the platform."""
    def __init__(self, applicant_id: str):
        super().__init__(
            status_code=404,
            code="applicant_not_found",
            message=f"KYC application '{applicant_id}' was not found.",
            details={"applicant_id": applicant_id},
        )


class DuplicateApplicationError(PlatformError):
    """Raised when an application with the specified applicant_id already exists."""
    def __init__(self, applicant_id: str):
        super().__init__(
            status_code=409,
            code="duplicate_application",
            message=f"KYC application '{applicant_id}' already exists.",
            details={"applicant_id": applicant_id},
        )


async def platform_error_handler(request: Request, exc: PlatformError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
                "request_id": request_id,
            }
        },
    )


async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("unexpected_server_error", extra={"request_id": request_id})
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_server_error",
                "message": "An unexpected error occurred while processing the request.",
                "request_id": request_id,
            }
        },
    )
