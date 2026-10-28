import time

import pytest
from datetime import timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.infrastructure.database import Base
from app.infrastructure.jobs.models import Job
from app.infrastructure.jobs.registry import (
    ExecutionContext,
    HandlerRegistry,
    PermanentJobError,
    RetryableJobError,
)
from app.infrastructure.jobs.service import (
    claim_one,
    enqueue_job,
    execute_job_lifecycle,
    finish_job,
)
from app.worker.worker import WorkerService
from app.modules.identity.models import utcnow

# In-memory SQLite engine with StaticPool for test isolation
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(
    bind=test_engine, autoflush=False, expire_on_commit=False
)


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def registry():
    reg = HandlerRegistry()

    def dummy_success(payload: dict, context: ExecutionContext) -> dict:
        context.check()
        return {"processed": True, "value": payload.get("x", 0) * 2}

    def dummy_permanent_failure(payload: dict, context: ExecutionContext) -> dict:
        context.check()
        raise PermanentJobError("Invalid input parameters", code="bad_input")

    def dummy_transient_failure(payload: dict, context: ExecutionContext) -> dict:
        context.check()
        raise RetryableJobError("Connection timeout", code="transient_network_error")

    def dummy_long_task(payload: dict, context: ExecutionContext) -> dict:
        for _ in range(10):
            context.check()
            time.sleep(0.01)
        return {"done": True}

    reg.register("test.compute", dummy_success)
    reg.register("test.fail_permanent", dummy_permanent_failure)
    reg.register("test.fail_transient", dummy_transient_failure)
    reg.register("test.long", dummy_long_task)
    return reg


def test_handler_registry_validation(registry):
    assert registry.get("test.compute") is not None
    assert registry.get("nonexistent.task") is None

    with pytest.raises(PermanentJobError, match="Unsupported or unregistered"):
        registry.validate_type("nonexistent.task")

    with pytest.raises(ValueError, match="already registered"):
        registry.register("test.compute", lambda p, c: {})


def test_enqueue_and_claim_lifecycle(registry):
    with TestingSessionLocal() as db:
        job = enqueue_job(
            db=db,
            job_type="test.compute",
            payload={"x": 21},
            registry=registry,
        )
        assert job.status == "queued"
        assert job.attempt_count == 0
        job_id = job.id

    with TestingSessionLocal() as db:
        claimed = claim_one(db, worker_id="worker-test-1", lease_seconds=10)
        assert claimed is not None
        assert claimed.id == job_id
        assert claimed.status == "running"
        assert claimed.attempt_count == 1
        assert claimed.worker_id == "worker-test-1"
        assert claimed.lease_token is not None

        # Double claim should return None because job is running
        double_claim = claim_one(db, worker_id="worker-test-2")
        assert double_claim is None


def test_finish_job_and_lease_fencing():
    with TestingSessionLocal() as db:
        job = Job(
            job_type="test.compute",
            input_json={"x": 5},
            status="running",
            worker_id="w-1",
            lease_token="valid_token_123",
            deadline_at=utcnow() + timedelta(minutes=5),
        )
        db.add(job)
        db.commit()
        job_id = job.id

    # Attempt to finish with invalid/stale lease token -> rejected
    with TestingSessionLocal() as db:
        fenced = finish_job(
            db=db,
            job_id=job_id,
            lease_token="stale_token_xyz",
            status="succeeded",
            result={"res": 10},
        )
        assert fenced is False

    # Attempt to finish with correct lease token -> succeeds
    with TestingSessionLocal() as db:
        success = finish_job(
            db=db,
            job_id=job_id,
            lease_token="valid_token_123",
            status="succeeded",
            result={"res": 10},
        )
        assert success is True

        verified = db.get(Job, job_id)
        assert verified.status == "succeeded"
        assert verified.result_json == {"res": 10}
        assert verified.lease_expires_at is None


def test_execute_job_lifecycle_success(registry):
    with TestingSessionLocal() as db:
        enqueue_job(
            db=db,
            job_type="test.compute",
            payload={"x": 25},
            registry=registry,
        )
        claimed = claim_one(db, worker_id="worker-test-1")

    execute_job_lifecycle(
        job=claimed,
        registry=registry,
        db_factory=TestingSessionLocal,
    )

    with TestingSessionLocal() as db:
        final_job = db.get(Job, claimed.id)
        assert final_job.status == "succeeded"
        assert final_job.result_json == {"processed": True, "value": 50}


def test_execute_job_lifecycle_permanent_failure(registry):
    with TestingSessionLocal() as db:
        enqueue_job(
            db=db,
            job_type="test.fail_permanent",
            payload={},
            registry=registry,
        )
        claimed = claim_one(db, worker_id="worker-test-1")

    execute_job_lifecycle(
        job=claimed,
        registry=registry,
        db_factory=TestingSessionLocal,
    )

    with TestingSessionLocal() as db:
        final_job = db.get(Job, claimed.id)
        assert final_job.status == "failed"
        assert final_job.error_code == "bad_input"
        assert "Invalid input" in final_job.last_error


def test_execute_job_lifecycle_retryable_backoff(registry):
    with TestingSessionLocal() as db:
        enqueue_job(
            db=db,
            job_type="test.fail_transient",
            payload={},
            max_attempts=3,
            registry=registry,
        )
        claimed = claim_one(db, worker_id="worker-test-1")

    # First attempt fails -> should requeue for retry
    execute_job_lifecycle(
        job=claimed,
        registry=registry,
        db_factory=TestingSessionLocal,
    )

    with TestingSessionLocal() as db:
        requeued_job = db.get(Job, claimed.id)
        assert requeued_job.status == "queued"
        assert requeued_job.attempt_count == 1
        assert requeued_job.error_code == "transient_network_error"
        assert requeued_job.worker_id is None


def test_cooperative_cancellation(registry):
    with TestingSessionLocal() as db:
        enqueue_job(
            db=db,
            job_type="test.long",
            payload={},
            registry=registry,
        )
        claimed = claim_one(db, worker_id="worker-test-1")
        # Request cancellation while job is claimed
        claimed.cancel_requested_at = utcnow()
        db.commit()

    execute_job_lifecycle(
        job=claimed,
        registry=registry,
        db_factory=TestingSessionLocal,
    )

    with TestingSessionLocal() as db:
        final_job = db.get(Job, claimed.id)
        assert final_job.status == "cancelled"
        assert final_job.error_code == "cancelled"


def test_worker_service_run_once(registry):
    with TestingSessionLocal() as db:
        enqueue_job(
            db=db,
            job_type="test.compute",
            payload={"x": 50},
            registry=registry,
        )

    worker = WorkerService(
        worker_id="worker-single-pass",
        registry=registry,
        db_factory=TestingSessionLocal,
    )

    executed = worker.run_once()
    assert executed is True

    # Next pass should find nothing
    executed_empty = worker.run_once()
    assert executed_empty is False

    with TestingSessionLocal() as db:
        jobs = db.query(Job).all()
        assert len(jobs) == 1
        assert jobs[0].status == "succeeded"
        assert jobs[0].result_json == {"processed": True, "value": 100}
