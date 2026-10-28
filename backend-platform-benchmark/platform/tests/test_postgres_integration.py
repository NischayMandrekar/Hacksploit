"""Integration tests intentionally require PostgreSQL (locking/partial indexes are PostgreSQL-specific)."""
import uuid, hashlib
from datetime import timedelta
from sqlalchemy import select
from platform_app.models import APIKey, Job, utcnow


def seed_key(db):
    key = APIKey(name="integration", key_prefix="plk_test", key_hash=hashlib.sha256(uuid.uuid4().hex.encode()).hexdigest(), scopes=["jobs:create", "jobs:read", "jobs:cancel"], created_at=utcnow(), is_active=True)
    db.add(key); db.commit(); return key

def test_job_claim_is_durable_and_lease_token_is_unique(db_session):
    from platform_app.jobs import claim_one, canonical_hash
    key = seed_key(db_session); now = utcnow()
    job = Job(api_key_id=key.id, job_type="demo.echo", input_json={"message": "hello"}, status="queued", attempt_count=0, max_attempts=3, available_at=now, request_hash=canonical_hash({"message":"hello"}), deadline_at=now+timedelta(seconds=30), request_id="test-request", created_at=now, updated_at=now)
    db_session.add(job); db_session.commit(); job_id = job.id
    claimed = claim_one(db_session, "integration-worker")
    assert claimed is not None and claimed.id == job_id
    assert claimed.status == "running" and claimed.attempt_count == 1
    assert claimed.lease_token and claimed.lease_expires_at > utcnow()
    persisted = db_session.execute(select(Job).where(Job.id == job_id)).scalar_one()
    assert persisted.status == "running" and persisted.lease_token == claimed.lease_token

def test_partial_idempotency_index_allows_null_keys(db_session):
    from platform_app.jobs import canonical_hash
    key = seed_key(db_session); now = utcnow()
    for _ in range(2):
        db_session.add(Job(api_key_id=key.id, job_type="demo.echo", input_json={}, status="queued", attempt_count=0, max_attempts=3, available_at=now, idempotency_key=None, request_hash=canonical_hash({}), deadline_at=now+timedelta(seconds=30), request_id=uuid.uuid4().hex, created_at=now, updated_at=now))
    db_session.commit()
    assert db_session.query(Job).count() == 2

def test_expired_lease_is_recovered_with_a_new_token(db_session):
    from platform_app.jobs import claim_one, canonical_hash
    key = seed_key(db_session); now = utcnow(); old_token = uuid.uuid4().hex
    job = Job(api_key_id=key.id, job_type="demo.echo", input_json={"message": "retry"}, status="running", attempt_count=1, max_attempts=3, available_at=now, request_hash=canonical_hash({"message":"retry"}), worker_id="dead-worker", lease_token=old_token, lease_expires_at=now-timedelta(seconds=5), deadline_at=now+timedelta(seconds=60), request_id="recover-request", created_at=now, updated_at=now)
    db_session.add(job); db_session.commit(); job_id = job.id
    claimed = claim_one(db_session, "replacement-worker")
    assert claimed is not None and claimed.id == job_id
    assert claimed.status == "running" and claimed.attempt_count == 2
    assert claimed.lease_token != old_token

def test_queued_job_past_deadline_is_failed(db_session):
    from platform_app.jobs import claim_one, canonical_hash
    key = seed_key(db_session); now = utcnow()
    job = Job(api_key_id=key.id, job_type="demo.echo", input_json={}, status="queued", attempt_count=0, max_attempts=3, available_at=now-timedelta(seconds=3), request_hash=canonical_hash({}), deadline_at=now-timedelta(seconds=1), request_id="expired-request", created_at=now-timedelta(seconds=5), updated_at=now-timedelta(seconds=5))
    db_session.add(job); db_session.commit(); job_id = job.id
    assert claim_one(db_session, "worker") is None
    db_session.refresh(job)
    assert job.id == job_id and job.status == "failed" and job.error_code == "deadline_exceeded"

def test_idempotency_key_is_unique_per_api_key(db_session):
    from sqlalchemy.exc import IntegrityError
    from platform_app.jobs import canonical_hash
    key = seed_key(db_session); now = utcnow()
    common = dict(api_key_id=key.id, job_type="demo.echo", input_json={}, status="queued", attempt_count=0, max_attempts=3, available_at=now, idempotency_key="same-key", request_hash=canonical_hash({}), deadline_at=now+timedelta(seconds=30), request_id="idempotency-request", created_at=now, updated_at=now)
    db_session.add(Job(**common)); db_session.commit()
    db_session.add(Job(**{**common, "request_id": "second-request"}))
    try:
        db_session.commit()
        assert False, "PostgreSQL should reject duplicate non-null idempotency key for the same API key"
    except IntegrityError:
        db_session.rollback()

def test_stale_worker_cannot_commit_after_lease_recovery(db_session, engine, monkeypatch):
    from sqlalchemy.orm import sessionmaker
    from platform_app.jobs import claim_one, canonical_hash, finish_job
    import platform_app.db as db_module
    key = seed_key(db_session); now = utcnow()
    job = Job(api_key_id=key.id, job_type="demo.echo", input_json={}, status="queued", attempt_count=0, max_attempts=3, available_at=now, request_hash=canonical_hash({}), deadline_at=now+timedelta(seconds=90), request_id="stale-worker-test", created_at=now, updated_at=now)
    db_session.add(job); db_session.commit(); job_id = job.id
    first_claim = claim_one(db_session, "worker-old")
    old_token = first_claim.lease_token
    first_claim.lease_expires_at = utcnow()-timedelta(seconds=1); db_session.commit()
    replacement = claim_one(db_session, "worker-new")
    assert replacement is not None and replacement.id == job_id and replacement.lease_token != old_token
    monkeypatch.setattr(db_module, "SessionLocal", sessionmaker(bind=engine, expire_on_commit=False))
    changed = finish_job(job_id, old_token, "succeeded", result={"incorrect": True}, request_id="stale-worker-test")
    assert changed is False
    db_session.refresh(replacement)
    assert replacement.status == "running" and replacement.result_json is None


def test_expired_lease_cannot_finish_before_another_worker_reclaims_it(db_session, engine, monkeypatch):
    """Lease expiry itself fences a worker; correctness must not depend on reclaim timing."""
    from sqlalchemy.orm import sessionmaker
    from platform_app.jobs import claim_one, canonical_hash, finish_job
    import platform_app.db as db_module

    key = seed_key(db_session)
    now = utcnow()
    job = Job(
        api_key_id=key.id, job_type="demo.echo", input_json={}, status="queued",
        attempt_count=0, max_attempts=3, available_at=now, request_hash=canonical_hash({}),
        deadline_at=now + timedelta(seconds=90), request_id="expired-lease-fence",
        created_at=now, updated_at=now,
    )
    db_session.add(job); db_session.commit(); job_id = job.id
    claimed = claim_one(db_session, "worker-old")
    token = claimed.lease_token
    claimed.lease_expires_at = utcnow() - timedelta(seconds=1)
    db_session.commit()
    monkeypatch.setattr(db_module, "SessionLocal", sessionmaker(bind=engine, expire_on_commit=False))

    changed = finish_job(job_id, token, "succeeded", result={"stale": True}, request_id="expired-lease-fence")
    assert changed is False
    db_session.refresh(claimed)
    assert claimed.status == "running"
    assert claimed.result_json is None


def test_concurrent_workers_never_claim_the_same_job(db_session, engine):
    """Two independent PostgreSQL sessions cannot own the same queued row."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from sqlalchemy.orm import sessionmaker
    from platform_app.jobs import claim_one, canonical_hash

    key = seed_key(db_session)
    now = utcnow()
    job = Job(
        api_key_id=key.id, job_type="demo.echo", input_json={"message": "once"},
        status="queued", attempt_count=0, max_attempts=3, available_at=now,
        request_hash=canonical_hash({"message": "once"}),
        deadline_at=now + timedelta(seconds=60), request_id="concurrent-claim",
        created_at=now, updated_at=now,
    )
    db_session.add(job); db_session.commit(); expected_id = job.id
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    barrier = Barrier(2)

    def claim(worker_id):
        with Session() as session:
            barrier.wait(timeout=5)
            claimed = claim_one(session, worker_id)
            return claimed.id if claimed else None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ("worker-a", "worker-b")))

    assert results.count(expected_id) == 1
    assert results.count(None) == 1


def test_cancelled_job_is_terminal_against_worker_completion(db_session, engine, monkeypatch):
    from sqlalchemy.orm import sessionmaker
    from platform_app.jobs import claim_one, canonical_hash, finish_job
    import platform_app.db as db_module

    key = seed_key(db_session)
    now = utcnow()
    job = Job(
        api_key_id=key.id, job_type="demo.echo", input_json={}, status="queued",
        attempt_count=0, max_attempts=3, available_at=now, request_hash=canonical_hash({}),
        deadline_at=now + timedelta(seconds=60), request_id="cancel-terminal",
        created_at=now, updated_at=now,
    )
    db_session.add(job); db_session.commit(); job_id = job.id
    claimed = claim_one(db_session, "worker-cancel-race")
    token = claimed.lease_token
    claimed.status = "cancelled"; claimed.finished_at = utcnow(); db_session.commit()
    monkeypatch.setattr(db_module, "SessionLocal", sessionmaker(bind=engine, expire_on_commit=False))

    changed = finish_job(job_id, token, "succeeded", result={"late": True}, request_id="cancel-terminal")
    assert changed is False
    db_session.refresh(claimed)
    assert claimed.status == "cancelled"
    assert claimed.result_json is None


def test_expired_lease_with_cancellation_request_is_not_requeued(db_session):
    from platform_app.jobs import claim_one, canonical_hash

    key = seed_key(db_session)
    now = utcnow()
    job = Job(
        api_key_id=key.id, job_type="demo.echo", input_json={}, status="queued",
        attempt_count=0, max_attempts=3, available_at=now, request_hash=canonical_hash({}),
        deadline_at=now + timedelta(seconds=60), request_id="cancel-lease-recovery",
        created_at=now, updated_at=now,
    )
    db_session.add(job)
    db_session.commit()
    claimed = claim_one(db_session, "worker-being-cancelled")
    claimed.cancel_requested_at = utcnow()
    claimed.lease_expires_at = utcnow() - timedelta(seconds=1)
    db_session.commit()

    assert claim_one(db_session, "replacement-worker") is None
    db_session.refresh(claimed)
    assert claimed.status == "cancelled"
    assert claimed.lease_token is None
    assert claimed.worker_id is None
