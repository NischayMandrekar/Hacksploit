from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
import logging
logger = logging.getLogger(__name__)

class PlatformError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        self.status_code, self.code, self.message = status_code, code, message

async def platform_error_handler(request: Request, exc: PlatformError):
    return JSONResponse(status_code=exc.status_code, content={"error": {"code": exc.code, "message": exc.message, "request_id": getattr(request.state, "request_id", "unknown")}})

async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("unhandled_request_error", extra={"request_id": getattr(request.state, "request_id", "unknown")})
    return JSONResponse(status_code=500, content={"error": {"code": "internal_error", "message": "An unexpected error occurred.", "request_id": getattr(request.state, "request_id", "unknown")}})


async def request_validation_error_handler(request: Request, exc: RequestValidationError):
    # Do not echo rejected input values; they may contain secrets or large payloads.
    fields = [
        {"field": ".".join(str(part) for part in item.get("loc", ())), "reason": item.get("type", "invalid")}
        for item in exc.errors()[:20]
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "invalid_request",
                "message": "Request validation failed.",
                "request_id": getattr(request.state, "request_id", "unknown"),
                "fields": fields,
            }
        },
    )
