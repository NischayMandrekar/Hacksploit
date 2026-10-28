import hashlib, json, logging, random, time, uuid
from datetime import datetime, timedelta, timezone
from sqlalchemy import and_, or_, select, update, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from platform_app.config import settings
from platform_app.execution import CPUExecutionBackend, ExecutionContext, JobCancelled, PermanentJobError, RetryableJobError
from platform_app.models import Job, utcnow
from platform_app.obs import JOBS_COMPLETED, JOBS_RETRIED, JOBS_ACTIVE, QUEUE_DEPTH, CLAIM_FAILURES
from platform_app.messaging import EventPublisher

logger = logging.getLogger(__name__)
backend = CPUExecutionBackend()
events = EventPublisher()

def canonical_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def safe_error(value: str, limit: int = 500) -> str:
    return " ".join(str(value).split())[:limit]

def claim_one(db: Session, worker_id: str) -> Job | None:
    now = utcnow()
    pending_events: list[tuple[str, str, str, int]] = []
    # Expire queued jobs whose end-to-end deadline elapsed before they could be claimed.
    overdue = db.execute(select(Job).where(Job.status == "queued", Job.deadline_at <= now).with_for_update(skip_locked=True).limit(50)).scalars().all()
    for job in overdue:
        job.status = "failed"; job.error_code = "deadline_exceeded"; job.last_error = "Job deadline elapsed before execution."; job.finished_at = now; job.updated_at = now
        JOBS_COMPLETED.labels("failed").inc()
        pending_events.append(("job.failed", str(job.id), job.request_id, job.attempt_count))
    # Recover expired leases before selecting work. A worker with an old lease token can no longer commit.
    expired = db.execute(select(Job).where(Job.status == "running", Job.lease_expires_at < now).with_for_update(skip_locked=True).limit(20)).scalars().all()
    for job in expired:
        # A cancellation request wins over lease recovery: never requeue cancelled work.
        if job.cancel_requested_at is not None:
            job.status = "cancelled"
            job.error_code = "cancelled"
            job.last_error = "Cancellation requested; worker lease expired."
            job.finished_at = now
            job.worker_id = None
            job.lease_token = None
            job.lease_expires_at = None
            JOBS_COMPLETED.labels("cancelled").inc()
            pending_events.append(("job.cancelled", str(job.id), job.request_id, job.attempt_count))
        elif job.attempt_count >= job.max_attempts or job.deadline_at <= now:
            job.status = "failed"
            job.error_code = "deadline_exceeded" if job.deadline_at <= now else "lease_expired"
            job.last_error = "Worker lease expired; retry budget exhausted or deadline elapsed."
            job.finished_at = now
            job.worker_id = None
            job.lease_token = None
            job.lease_expires_at = None
            JOBS_COMPLETED.labels("failed").inc()
            pending_events.append(("job.failed", str(job.id), job.request_id, job.attempt_count))
        else:
            job.status = "queued"
            job.available_at = now
            job.worker_id = None
            job.lease_token = None
            job.lease_expires_at = None
            JOBS_RETRIED.inc()
            pending_events.append(("job.queued", str(job.id), job.request_id, job.attempt_count))
        job.updated_at = now
    db.flush()
    candidate = db.execute(select(Job).where(Job.status == "queued", Job.available_at <= now, Job.deadline_at > now).order_by(Job.available_at, Job.created_at).with_for_update(skip_locked=True).limit(1)).scalars().first()
    if candidate is None:
        db.commit()
        for event_type, job_id, request_id, attempt in pending_events:
            events.publish(event_type, job_id, request_id, attempt)
        return None
    candidate.status = "running"; candidate.attempt_count += 1; candidate.worker_id = worker_id
    candidate.lease_token = uuid.uuid4().hex; candidate.lease_expires_at = now + timedelta(seconds=settings.worker_lease_seconds)
    candidate.started_at = candidate.started_at or now; candidate.updated_at = now
    pending_events.append(("job.running", str(candidate.id), candidate.request_id, candidate.attempt_count))
    db.commit(); db.refresh(candidate)
    for event_type, job_id, request_id, attempt in pending_events:
        events.publish(event_type, job_id, request_id, attempt)
    return candidate

def is_cancel_requested(job_id, lease_token) -> bool:
    from platform_app.db import SessionLocal
    with SessionLocal() as db:
        job = db.execute(select(Job.cancel_requested_at, Job.status, Job.lease_token, Job.lease_expires_at).where(Job.id == job_id)).first()
        return (
            not job
            or job.status != "running"
            or job.lease_token != lease_token
            or job.cancel_requested_at is not None
            or job.lease_expires_at is None
            or job.lease_expires_at <= utcnow()
        )

def finish_job(job_id, lease_token, status, result=None, error_code=None, error=None, retry_at=None, attempt=0, request_id="") -> bool:
    from platform_app.db import SessionLocal
    now = utcnow()
    values = {"updated_at": now}
    if status == "queued":
        values.update(status="queued", available_at=retry_at, worker_id=None, lease_token=None, lease_expires_at=None, error_code=error_code, last_error=safe_error(error or "Retry scheduled"))
    else:
        values.update(status=status, finished_at=now, result_json=result, error_code=error_code, last_error=safe_error(error) if error else None, lease_expires_at=None)
    with SessionLocal() as db:
        changed = db.execute(
            update(Job)
            .where(
                Job.id == job_id,
                Job.status == "running",
                Job.lease_token == lease_token,
                Job.lease_expires_at > now,
            )
            .values(**values)
        )
        db.commit()
        if changed.rowcount != 1:
            logger.warning("stale_worker_result_discarded", extra={"job_id": str(job_id), "worker_id": "worker", "request_id": request_id})
            return False
    if status in ("succeeded", "failed", "cancelled"):
        JOBS_COMPLETED.labels(status).inc()
        events.publish(f"job.{status}", str(job_id), request_id, attempt)
    elif status == "queued": JOBS_RETRIED.inc()
    return True

def run_claimed(job: Job):
    started = time.monotonic(); JOBS_ACTIVE.inc()
    try:
        max_seconds = min(settings.worker_execution_timeout_seconds, max(1, int((job.deadline_at - utcnow()).total_seconds())))
        context = ExecutionContext(str(job.id), job.request_id, job.attempt_count, started + max_seconds, lambda: is_cancel_requested(job.id, job.lease_token))
        result = backend.execute(job.job_type, job.input_json, context)
        if len(json.dumps(result, ensure_ascii=False).encode()) > settings.job_max_output_bytes:
            raise PermanentJobError("output_too_large", "Handler output exceeded the configured limit.")
        if is_cancel_requested(job.id, job.lease_token):
            finish_job(job.id, job.lease_token, "cancelled", error_code="cancelled", error="Cancellation requested", attempt=job.attempt_count, request_id=job.request_id)
        else:
            finish_job(job.id, job.lease_token, "succeeded", result=result, attempt=job.attempt_count, request_id=job.request_id)
    except JobCancelled:
        finish_job(job.id, job.lease_token, "cancelled", error_code="cancelled", error="Cancellation requested", attempt=job.attempt_count, request_id=job.request_id)
    except PermanentJobError as exc:
        finish_job(job.id, job.lease_token, "failed", error_code=exc.code, error=exc.message, attempt=job.attempt_count, request_id=job.request_id)
    except Exception as exc:
        now = utcnow()
        can_retry = job.attempt_count < job.max_attempts and job.deadline_at > now
        if can_retry:
            delay = min(30, 2 ** max(0, job.attempt_count - 1)) * random.uniform(0.8, 1.2)
            finish_job(job.id, job.lease_token, "queued", error_code="execution_error", error=type(exc).__name__, retry_at=now + timedelta(seconds=delay), attempt=job.attempt_count, request_id=job.request_id)
        else:
            finish_job(job.id, job.lease_token, "failed", error_code="execution_error", error=type(exc).__name__, attempt=job.attempt_count, request_id=job.request_id)
        logger.warning("job_execution_failed", extra={"job_id": str(job.id), "request_id": job.request_id, "error_code": "execution_error"}, exc_info=True)
    finally:
        JOBS_ACTIVE.dec()

def refresh_queue_depth(db: Session):
    try: QUEUE_DEPTH.set(db.query(func.count(Job.id)).filter(Job.status == "queued").scalar() or 0)
    except Exception: logger.debug("queue_depth_refresh_failed", exc_info=True)
