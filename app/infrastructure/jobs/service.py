import logging
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.core.config import settings
from app.infrastructure.database import SessionLocal
from app.infrastructure.jobs.models import Job
from app.infrastructure.jobs.registry import (
    ExecutionContext,
    HandlerRegistry,
    JobCancelled,
    PermanentJobError,
    RetryableJobError,
    default_registry,
)
from app.modules.identity.models import ensure_utc, utcnow

logger = logging.getLogger(__name__)


def safe_error_message(msg: str, limit: int = 500) -> str:
    return " ".join(str(msg).split())[:limit]


def enqueue_job(
    db: Session,
    job_type: str,
    payload: dict,
    user_id: uuid.UUID | None = None,
    timeout_seconds: int | None = None,
    max_attempts: int | None = None,
    request_id: str = "unknown",
    registry: HandlerRegistry | None = None,
) -> Job:
    """Enqueues a durable background job into the PostgreSQL queue."""
    active_registry = registry or default_registry
    active_registry.validate_type(job_type)

    timeout = timeout_seconds or settings.job_default_timeout_seconds if hasattr(settings, "job_default_timeout_seconds") else 30
    attempts = max_attempts or settings.worker_max_attempts
    now = utcnow()

    job = Job(
        user_id=user_id,
        job_type=job_type,
        input_json=payload,
        status="queued",
        attempt_count=0,
        max_attempts=attempts,
        available_at=now,
        deadline_at=now + timedelta(seconds=timeout),
        request_id=request_id,
        created_at=now,
        updated_at=now,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def claim_one(
    db: Session,
    worker_id: str,
    lease_seconds: int = 45,
) -> Job | None:
    """
    Claims the next available job using PostgreSQL FOR UPDATE SKIP LOCKED.
    Falls back gracefully to standard selection on SQLite in unit tests.
    Performs lease recovery and deadline expiration before claim.
    """
    now = utcnow()
    is_postgres = make_url(settings.database_url).get_backend_name() != "sqlite"

    # 1. Expire queued jobs whose deadlines passed before execution started
    overdue_query = select(Job).where(Job.status == "queued", Job.deadline_at <= now)
    if is_postgres:
        overdue_query = overdue_query.with_for_update(skip_locked=True)
    overdue_jobs = db.execute(overdue_query.limit(20)).scalars().all()

    for job in overdue_jobs:
        job.status = "failed"
        job.error_code = "deadline_exceeded"
        job.last_error = "Job deadline elapsed before a worker could claim it."
        job.finished_at = now
        job.updated_at = now

    # 2. Recover expired leases from dead/stalled workers
    expired_query = select(Job).where(Job.status == "running", Job.lease_expires_at < now)
    if is_postgres:
        expired_query = expired_query.with_for_update(skip_locked=True)
    expired_jobs = db.execute(expired_query.limit(20)).scalars().all()

    for job in expired_jobs:
        if job.cancel_requested_at is not None:
            job.status = "cancelled"
            job.error_code = "cancelled"
            job.last_error = "Cancellation requested; worker lease expired."
            job.finished_at = now
        elif job.attempt_count >= job.max_attempts or ensure_utc(job.deadline_at) <= now:
            job.status = "failed"
            job.error_code = "deadline_exceeded" if ensure_utc(job.deadline_at) <= now else "lease_expired"
            job.last_error = "Worker lease expired; max attempts exhausted or deadline passed."
            job.finished_at = now
        else:
            job.status = "queued"
            job.available_at = now
            job.worker_id = None
            job.lease_token = None
            job.lease_expires_at = None

        job.updated_at = now

    db.flush()

    # 3. Claim the next ready job
    candidate_query = (
        select(Job)
        .where(Job.status == "queued", Job.available_at <= now, Job.deadline_at > now)
        .order_by(Job.available_at, Job.created_at)
    )
    if is_postgres:
        candidate_query = candidate_query.with_for_update(skip_locked=True)

    candidate = db.execute(candidate_query.limit(1)).scalars().first()
    if candidate is None:
        db.commit()
        return None

    candidate.status = "running"
    candidate.attempt_count += 1
    candidate.worker_id = worker_id
    candidate.lease_token = uuid.uuid4().hex
    candidate.lease_expires_at = now + timedelta(seconds=lease_seconds)
    candidate.started_at = candidate.started_at or now
    candidate.updated_at = now

    db.commit()
    db.refresh(candidate)
    return candidate


def is_cancel_requested(
    db: Session,
    job_id: uuid.UUID,
    lease_token: str,
) -> bool:
    """Checks whether the client requested cancellation or the lease was fenced."""
    job = db.execute(
        select(
            Job.status,
            Job.lease_token,
            Job.lease_expires_at,
            Job.cancel_requested_at,
        ).where(Job.id == job_id)
    ).first()

    if not job:
        return True
    if job.status != "running":
        return True
    if job.lease_token != lease_token:
        return True
    if job.cancel_requested_at is not None:
        return True
    if job.lease_expires_at is None or ensure_utc(job.lease_expires_at) <= utcnow():
        return True
    return False


def finish_job(
    db: Session,
    job_id: uuid.UUID,
    lease_token: str,
    status: str,
    result: dict | None = None,
    error_code: str | None = None,
    error: str | None = None,
    retry_at: datetime | None = None,
) -> bool:
    """
    Completes or requeues a job. Enforces lease fencing: if the lease_token no
    longer matches (e.g. lease recovered by another worker), update is rejected.
    """
    now = utcnow()
    job = db.execute(
        select(Job).where(Job.id == job_id, Job.lease_token == lease_token)
    ).scalar_one_or_none()

    if not job:
        logger.warning(
            "lease_fencing_rejected_finish",
            extra={"job_id": str(job_id), "lease_token": lease_token},
        )
        return False

    job.updated_at = now
    if status == "queued":
        job.status = "queued"
        job.available_at = retry_at or now
        job.worker_id = None
        job.lease_token = None
        job.lease_expires_at = None
        job.error_code = error_code
        job.last_error = safe_error_message(error or "Retry scheduled.")
    else:
        job.status = status
        job.finished_at = now
        job.result_json = result
        job.error_code = error_code
        job.last_error = safe_error_message(error) if error else None
        job.lease_expires_at = None

    db.commit()
    return True


def execute_job_lifecycle(
    job: Job,
    registry: HandlerRegistry | None = None,
    db_factory: Callable[[], Session] | None = None,
    execution_timeout_seconds: int = 30,
) -> None:
    """Executes a claimed job with cooperative cancellation and lease fencing."""
    active_registry = registry or default_registry
    get_session = db_factory or SessionLocal

    job_id = job.id
    lease_token = job.lease_token or ""
    deadline_monotonic = time.monotonic() + execution_timeout_seconds

    def check_cancelled() -> bool:
        with get_session() as db:
            return is_cancel_requested(db, job_id, lease_token)

    context = ExecutionContext(
        job_id=str(job_id),
        request_id=job.request_id,
        attempt=job.attempt_count,
        deadline_monotonic=deadline_monotonic,
        cancellation_check=check_cancelled,
    )

    try:
        result = active_registry.execute(job.job_type, job.input_json, context)
        with get_session() as db:
            finish_job(db, job_id, lease_token, status="succeeded", result=result)

    except JobCancelled as exc:
        with get_session() as db:
            finish_job(
                db,
                job_id,
                lease_token,
                status="cancelled",
                error_code="cancelled",
                error=str(exc),
            )

    except PermanentJobError as exc:
        with get_session() as db:
            finish_job(
                db,
                job_id,
                lease_token,
                status="failed",
                error_code=exc.code,
                error=exc.message,
            )

    except RetryableJobError as exc:
        _handle_retry_or_fail(get_session, job, lease_token, exc.code, exc.message)

    except Exception as exc:
        logger.exception("job_execution_unexpected_failure", extra={"job_id": str(job_id)})
        _handle_retry_or_fail(get_session, job, lease_token, "unexpected_error", str(exc))


def _handle_retry_or_fail(
    db_factory: Callable[[], Session],
    job: Job,
    lease_token: str,
    code: str,
    message: str,
) -> None:
    now = utcnow()
    with db_factory() as db:
        if job.attempt_count < job.max_attempts and ensure_utc(job.deadline_at) > now:
            # Exponential backoff: 2s, 4s, 8s
            backoff_seconds = 2 ** job.attempt_count
            retry_at = now + timedelta(seconds=backoff_seconds)
            finish_job(
                db,
                job.id,
                lease_token,
                status="queued",
                error_code=code,
                error=message,
                retry_at=retry_at,
            )
        else:
            finish_job(
                db,
                job.id,
                lease_token,
                status="failed",
                error_code=code if job.attempt_count < job.max_attempts else "max_attempts_exceeded",
                error=message,
            )
