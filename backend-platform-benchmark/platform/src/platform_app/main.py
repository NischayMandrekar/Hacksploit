import logging, time, uuid
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from platform_app.api import router
from platform_app.identity import router as identity_router
from platform_app.config import settings
from platform_app.db import SessionLocal
from platform_app.errors import (
    PlatformError, platform_error_handler, request_validation_error_handler,
    unexpected_error_handler,
)
from platform_app.obs import REQUESTS, REQUEST_LATENCY, configure_logging, metrics_payload

configure_logging(settings.app_log_level)
logger = logging.getLogger(__name__)
app = FastAPI(title="Local Backend Platform", version="0.1.0", description="Durable local-first job API with an independent worker.")
app.include_router(router, prefix="/api/v1")
app.include_router(identity_router, prefix="/api/v1")
app.add_exception_handler(PlatformError, platform_error_handler)
app.add_exception_handler(RequestValidationError, request_validation_error_handler)
app.add_exception_handler(Exception, unexpected_error_handler)

@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "")[:100] or uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        raise
    response.headers["X-Request-ID"] = request_id
    route = request.scope.get("route")
    route_name = getattr(route, "path", "unmatched")
    status = response.status_code
    REQUESTS.labels(request.method, route_name, f"{status // 100}xx").inc()
    REQUEST_LATENCY.labels(request.method, route_name).observe(time.monotonic() - started)
    logger.info("http_request", extra={"service": "api", "request_id": request_id, "duration_ms": round((time.monotonic() - started) * 1000, 2)})
    return response

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    try:
        with SessionLocal() as db: db.execute(text("SELECT 1"))
        return {"status": "ready", "checks": {"postgres": "ok"}}
    except SQLAlchemyError:
        return Response(content='{"status":"not_ready","checks":{"postgres":"failed"}}', status_code=503, media_type="application/json")

@app.get("/metrics", include_in_schema=False)
def metrics():
    if not settings.prometheus_enabled: return Response(status_code=404)
    payload, content_type = metrics_payload()
    return Response(content=payload, media_type=content_type)
