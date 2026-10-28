import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.core.errors import PlatformError, platform_error_handler, unexpected_error_handler
from app.infrastructure.database import check_database_connection
from app.modules.identity.router import router as identity_router
from app.modules.kyc.router import router as kyc_router
from app.modules.products.router import router as products_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="ARGUS: Synthetic Identity Fraud Detection Modular Monolith Platform",
)

# Exception handlers
app.add_exception_handler(PlatformError, platform_error_handler)
app.add_exception_handler(Exception, unexpected_error_handler)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "")[:100] or uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.monotonic()

    response = await call_next(request)

    duration_ms = round((time.monotonic() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "http_request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
            "request_id": request_id,
        },
    )
    return response


# Route composition
app.include_router(identity_router, prefix="/api/v1")
app.include_router(kyc_router, prefix="/api/v1")
app.include_router(products_router)


@app.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
def readiness() -> dict[str, str]:
    try:
        check_database_connection()
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        ) from exc

    return {"status": "ready"}