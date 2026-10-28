import json, logging, secrets, uuid
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, Header, Query, Request, Response
from sqlalchemy import select, update, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from platform_app.config import settings
from platform_app.cache import rate_limiter
from platform_app.db import get_db
from platform_app.errors import PlatformError
from platform_app.db_errors import integrity_constraint_name
from platform_app.jobs import backend, canonical_hash
from platform_app.messaging import WakeupPublisher, EventPublisher
from platform_app.models import APIKey, Job, utcnow
from platform_app.obs import JOBS_CREATED
from platform_app.schemas import KeyCreate, KeyIssued, JobCreate, JobView, CancelView
from platform_app.security import current_api_key, hash_key, new_key, require_master_key, require_scope
from platform_app.auth import Principal, require_permission
from platform_app.execution import PermanentJobError

logger = logging.getLogger(__name__)
router = APIRouter()
wakeups = WakeupPublisher(); events = EventPublisher()

def to_view(job: Job) -> JobView:
    return JobView(id=job.id, job_type=job.job_type, status=job.status, attempt_count=job.attempt_count, max_attempts=job.max_attempts, available_at=job.available_at, created_at=job.created_at, updated_at=job.updated_at, started_at=job.started_at, finished_at=job.finished_at, result=job.result_json, error_code=job.error_code, last_error=job.last_error)

@router.post("/api-keys", response_model=KeyIssued, status_code=201, dependencies=[Depends(require_master_key)])
def create_api_key(body: KeyCreate, db: Session = Depends(get_db)):
    raw = new_key(); now = utcnow()
    if body.expires_at and body.expires_at <= now: raise PlatformError(422, "invalid_expiry", "expires_at must be in the future.")
    key = APIKey(name=body.name, key_prefix=raw[:12], key_hash=hash_key(raw), scopes=body.scopes, created_at=now, expires_at=body.expires_at)
    db.add(key); db.commit(); db.refresh(key)
    return KeyIssued(id=key.id, name=key.name, key_prefix=key.key_prefix, raw_key=raw, scopes=key.scopes, created_at=key.created_at, expires_at=key.expires_at)

@router.delete("/api-keys/{key_id}", status_code=204, dependencies=[Depends(require_master_key)])
def revoke_api_key(key_id: uuid.UUID, db: Session = Depends(get_db)):
    key = db.get(APIKey, key_id)
    if not key: raise PlatformError(404, "not_found", "API key not found.")
    key.is_active = False; key.revoked_at = utcnow(); db.commit()
    return Response(status_code=204)

@router.post("/jobs", response_model=JobView, status_code=202)
def create_job(body: JobCreate, request: Request, response: Response, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"), principal: Principal = Depends(require_permission("jobs:create")), db: Session = Depends(get_db)):
    if idempotency_key is not None and (len(idempotency_key) > 200 or not idempotency_key.strip()): raise PlatformError(422, "invalid_idempotency_key", "Idempotency-Key must contain 1-200 non-whitespace characters.")
    if not rate_limiter.allowed(str(principal.api_key_id or principal.user_id)): raise PlatformError(429, "rate_limit_exceeded", "Too many job submissions; retry after the rate-limit window.")
    try: backend.validate_type(body.job_type)
    except PermanentJobError as exc: raise PlatformError(422, exc.code, exc.message)
    input_bytes = len(json.dumps(body.input, ensure_ascii=False, separators=(",", ":")).encode())
    if input_bytes > settings.job_max_input_bytes: raise PlatformError(413, "input_too_large", "Job input exceeds the configured byte limit.")
    timeout = body.timeout_seconds or settings.job_default_timeout_seconds
    if timeout > settings.job_max_timeout_seconds: raise PlatformError(422, "timeout_too_large", "timeout_seconds exceeds the configured maximum.")
    max_attempts = body.max_attempts or settings.worker_max_attempts
    request_payload = {"job_type": body.job_type, "input": body.input, "timeout_seconds": timeout, "max_attempts": max_attempts}
    digest = canonical_hash(request_payload)
    if idempotency_key:
        existing_query = select(Job).where(Job.idempotency_key == idempotency_key)
        if principal.auth_kind == "api_key":
            existing_query = existing_query.where(Job.api_key_id == principal.api_key_id)
        else:
            existing_query = existing_query.where(Job.user_id == principal.user_id, Job.api_key_id.is_(None))
        existing = db.execute(existing_query).scalar_one_or_none()
        if existing:
            if existing.request_hash != digest: raise PlatformError(409, "idempotency_conflict", "This Idempotency-Key was already used with a different request.")
            response.status_code = 200
            return to_view(existing)
    now = utcnow()
    job = Job(api_key_id=principal.api_key_id, user_id=principal.user_id, job_type=body.job_type, input_json=body.input, status="queued", attempt_count=0, max_attempts=max_attempts, available_at=now, idempotency_key=idempotency_key, request_hash=digest, deadline_at=now + timedelta(seconds=timeout), request_id=request.state.request_id, created_at=now, updated_at=now)
    db.add(job)
    try:
        db.commit(); db.refresh(job)
    except IntegrityError as exc:
        db.rollback()
        # Only translate the expected partial-unique-index race. Hiding unrelated
        # integrity failures would mask schema/data bugs as successful replays.
        constraint = integrity_constraint_name(exc)
        if constraint not in {"uq_jobs_idempotency", "uq_jobs_user_idempotency"}:
            raise
        if idempotency_key:
            existing_query = select(Job).where(Job.idempotency_key == idempotency_key)
            if principal.auth_kind == "api_key":
                existing_query = existing_query.where(Job.api_key_id == principal.api_key_id)
            else:
                existing_query = existing_query.where(Job.user_id == principal.user_id, Job.api_key_id.is_(None))
            existing = db.execute(existing_query).scalar_one_or_none()
            if existing and existing.request_hash == digest:
                response.status_code = 200
                return to_view(existing)
            if existing:
                raise PlatformError(409, "idempotency_conflict", "This Idempotency-Key was already used with a different request.")
        raise
    JOBS_CREATED.inc(); events.publish("job.queued", str(job.id), job.request_id, 0); wakeups.publish(str(job.id))
    return to_view(job)

@router.get("/jobs/{job_id}", response_model=JobView)
def get_job(job_id: uuid.UUID, principal: Principal = Depends(require_permission("jobs:read")), db: Session = Depends(get_db)):
    owner_clause = Job.api_key_id == principal.api_key_id if principal.auth_kind == "api_key" else Job.user_id == principal.user_id
    job = db.execute(select(Job).where(Job.id == job_id, owner_clause)).scalar_one_or_none()
    if not job: raise PlatformError(404, "not_found", "Job not found.")
    return to_view(job)

@router.post("/jobs/{job_id}/cancel", response_model=CancelView)
def cancel_job(job_id: uuid.UUID, principal: Principal = Depends(require_permission("jobs:cancel")), db: Session = Depends(get_db)):
    owner_clause = Job.api_key_id == principal.api_key_id if principal.auth_kind == "api_key" else Job.user_id == principal.user_id
    job = db.execute(select(Job).where(Job.id == job_id, owner_clause).with_for_update()).scalar_one_or_none()
    if not job: raise PlatformError(404, "not_found", "Job not found.")
    now = utcnow(); requested = False; transitioned_to_cancelled = False
    if job.status == "queued":
        job.status = "cancelled"; job.finished_at = now; job.updated_at = now; job.error_code = "cancelled"; job.last_error = "Cancelled before execution."; requested = True
        transitioned_to_cancelled = True
    elif job.status == "running" and job.cancel_requested_at is None:
        job.cancel_requested_at = now; job.updated_at = now; requested = True
    db.commit()
    if transitioned_to_cancelled:
        from platform_app.obs import JOBS_COMPLETED
        JOBS_COMPLETED.labels("cancelled").inc()
        events.publish("job.cancelled", str(job.id), job.request_id, job.attempt_count)
    return CancelView(id=job.id, status=job.status, cancel_requested=requested)


@router.get("/jobs", response_model=list[JobView])
def list_jobs(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=100000),
    principal: Principal = Depends(require_permission("jobs:read")),
    db: Session = Depends(get_db),
):
    owner_clause = Job.api_key_id == principal.api_key_id if principal.auth_kind == "api_key" else Job.user_id == principal.user_id
    rows = db.execute(
        select(Job).where(owner_clause).order_by(Job.created_at.desc()).limit(limit).offset(offset)
    ).scalars().all()
    return [to_view(job) for job in rows]
