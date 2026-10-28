"""HTTP-level integration tests; run only against a dedicated PostgreSQL database."""
from datetime import timedelta
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
MASTER = "integration-test-master-key-" + "x" * 40


@pytest.fixture
def client(db_session, engine, monkeypatch):
    """Use a fresh SQLAlchemy session per HTTP request, as production does."""
    from sqlalchemy.orm import sessionmaker
    from platform_app.config import settings
    from platform_app.db import get_db
    from platform_app.main import app

    monkeypatch.setattr(settings, "app_master_key", MASTER)
    monkeypatch.setattr(settings, "auth_jwt_secret", MASTER + "-jwt-secret")
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    def override_get_db():
        with Session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def issue_key(client, scopes=None, expires_at=None):
    body = {"name": "integration", "scopes": scopes or ["jobs:create", "jobs:read", "jobs:cancel"]}
    if expires_at is not None:
        body["expires_at"] = expires_at
    response = client.post("/api/v1/api-keys", headers={"X-Master-Key": MASTER}, json=body)
    assert response.status_code == 201, response.text
    return response.json()


def submit(client, raw_key, payload=None, idempotency_key=None):
    headers = {"X-API-Key": raw_key}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    return client.post(
        "/api/v1/jobs",
        headers=headers,
        json=payload or {"job_type": "demo.echo", "input": {"message": "hello"}},
    )


def test_master_key_and_scope_enforcement(client):
    denied = client.post(
        "/api/v1/api-keys",
        headers={"X-Master-Key": "wrong"},
        json={"name": "nope", "scopes": ["jobs:read"]},
    )
    assert denied.status_code == 401

    key = issue_key(client, ["jobs:read"])
    response = submit(client, key["raw_key"])
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "insufficient_scope"


def test_job_submission_idempotency_and_payload_conflict(client, db_session):
    from sqlalchemy import select
    from platform_app.models import Job

    key = issue_key(client)
    first = submit(client, key["raw_key"], idempotency_key="stable-request")
    replay = submit(client, key["raw_key"], idempotency_key="stable-request")
    conflict = submit(
        client,
        key["raw_key"],
        payload={"job_type": "demo.echo", "input": {"message": "different"}},
        idempotency_key="stable-request",
    )
    assert first.status_code == 202
    assert replay.status_code == 200
    assert first.json()["id"] == replay.json()["id"]
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_conflict"
    count = db_session.execute(select(Job).where(Job.api_key_id == UUID(key["id"]))).scalars().all()
    assert len(count) == 1


def test_job_ownership_is_hidden_and_key_revocation_takes_effect(client):
    owner = issue_key(client)
    other = issue_key(client)
    created = submit(client, owner["raw_key"])
    assert created.status_code == 202
    job_id = created.json()["id"]

    hidden = client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": other["raw_key"]})
    assert hidden.status_code == 404

    revoked = client.delete(
        f"/api/v1/api-keys/{owner['id']}", headers={"X-Master-Key": MASTER}
    )
    assert revoked.status_code == 204
    after_revoke = client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": owner["raw_key"]})
    assert after_revoke.status_code == 401


def test_queued_job_can_be_cancelled_and_remains_terminal(client, db_session):
    from platform_app.models import Job

    key = issue_key(client)
    created = submit(client, key["raw_key"])
    job_id = created.json()["id"]

    cancelled = client.post(
        f"/api/v1/jobs/{job_id}/cancel", headers={"X-API-Key": key["raw_key"]}
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["cancel_requested"] is True

    stored = db_session.get(Job, UUID(job_id))
    assert stored.status == "cancelled"
    assert stored.finished_at is not None

    repeated = client.post(
        f"/api/v1/jobs/{job_id}/cancel", headers={"X-API-Key": key["raw_key"]}
    )
    assert repeated.status_code == 200
    assert repeated.json()["status"] == "cancelled"
    assert repeated.json()["cancel_requested"] is False


def test_expired_api_key_is_rejected(client, db_session):
    from platform_app.models import APIKey, utcnow

    key = issue_key(client)
    stored_key = db_session.get(APIKey, UUID(key["id"]))
    stored_key.expires_at = utcnow() - timedelta(seconds=1)
    db_session.commit()

    response = client.get("/api/v1/jobs/00000000-0000-0000-0000-000000000001", headers={"X-API-Key": key["raw_key"]})
    assert response.status_code == 401


def test_concurrent_http_idempotency_returns_one_durable_job(client, engine):
    """Concurrent retries race through independent HTTP sessions and DB transactions."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from sqlalchemy import select
    from platform_app.models import Job
    from sqlalchemy.orm import sessionmaker

    key = issue_key(client)
    barrier = Barrier(2)

    def submit_concurrently():
        barrier.wait(timeout=5)
        return submit(client, key["raw_key"], idempotency_key="concurrent-http-idempotency")

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: submit_concurrently(), range(2)))

    assert sorted(response.status_code for response in responses) == [200, 202]
    assert responses[0].json()["id"] == responses[1].json()["id"]
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as session:
        rows = session.execute(
            select(Job).where(
                Job.api_key_id == UUID(key["id"]),
                Job.idempotency_key == "concurrent-http-idempotency",
            )
        ).scalars().all()
    assert len(rows) == 1


def test_rate_limiter_rejection_uses_standard_error_envelope(client, monkeypatch):
    from platform_app.api import rate_limiter

    key = issue_key(client)
    monkeypatch.setattr(rate_limiter, "allowed", lambda _key_id: False)
    response = submit(client, key["raw_key"])

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limit_exceeded"
    assert response.json()["error"]["request_id"]


def register(client, email="person@example.com", password="a-long-test-password-123"):
    return client.post("/api/v1/auth/register", json={"email": email, "password": password})


def login(client, email="person@example.com", password="a-long-test-password-123"):
    return client.post("/api/v1/auth/login", json={"email": email, "password": password})


def test_registration_login_profile_and_case_insensitive_email(client):
    created = register(client)
    assert created.status_code == 201, created.text
    assert created.json()["role"] == "user"
    assert "password_hash" not in created.json()

    duplicate = register(client, email="PERSON@EXAMPLE.COM")
    assert duplicate.status_code == 409

    bad_login = login(client, password="incorrect-password")
    assert bad_login.status_code == 401

    authenticated = login(client)
    assert authenticated.status_code == 200, authenticated.text
    tokens = authenticated.json()
    assert tokens["token_type"] == "Bearer"
    assert tokens["access_token"]
    assert tokens["refresh_token"].startswith("plr_")

    profile = client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert profile.status_code == 200
    assert profile.json()["email"] == "person@example.com"


def test_refresh_rotation_reuse_revokes_token_family(client):
    assert register(client).status_code == 201
    logged_in = login(client)
    old_refresh = logged_in.json()["refresh_token"]

    rotated = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert rotated.status_code == 200, rotated.text
    new_refresh = rotated.json()["refresh_token"]
    assert new_refresh != old_refresh

    reuse = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert reuse.status_code == 401
    assert reuse.json()["error"]["code"] == "refresh_token_reused"

    after_reuse = client.post("/api/v1/auth/refresh", json={"refresh_token": new_refresh})
    assert after_reuse.status_code == 401


def test_user_job_ownership_and_user_owned_api_key(client):
    first = register(client, "first@example.com")
    second = register(client, "second@example.com")
    assert first.status_code == second.status_code == 201
    first_tokens = login(client, "first@example.com").json()
    second_tokens = login(client, "second@example.com").json()
    first_auth = {"Authorization": f"Bearer {first_tokens['access_token']}"}
    second_auth = {"Authorization": f"Bearer {second_tokens['access_token']}"}

    submitted = client.post("/api/v1/jobs", headers=first_auth, json={"job_type": "demo.echo", "input": {"message": "owned"}})
    assert submitted.status_code == 202, submitted.text
    job_id = submitted.json()["id"]
    assert client.get(f"/api/v1/jobs/{job_id}", headers=first_auth).status_code == 200
    assert client.get(f"/api/v1/jobs/{job_id}", headers=second_auth).status_code == 404

    issued = client.post("/api/v1/users/me/api-keys", headers=first_auth, json={"name": "script", "scopes": ["jobs:create", "jobs:read"]})
    assert issued.status_code == 201, issued.text
    raw_key = issued.json()["raw_key"]
    via_key = client.post("/api/v1/jobs", headers={"X-API-Key": raw_key}, json={"job_type": "demo.echo", "input": {"message": "key-owned"}})
    assert via_key.status_code == 202, via_key.text
    assert client.get(f"/api/v1/jobs/{via_key.json()['id']}", headers=first_auth).status_code == 200
    assert client.get("/api/v1/users/me/api-keys", headers=first_auth).status_code == 200


def test_password_change_revokes_refresh_sessions_and_requires_new_password(client):
    assert register(client, "password-change@example.com").status_code == 201
    tokens = login(client, "password-change@example.com").json()
    auth = {"Authorization": f"Bearer {tokens['access_token']}"}

    wrong = client.post("/api/v1/auth/change-password", headers=auth, json={
        "current_password": "not-the-current-password",
        "new_password": "a-different-long-password-456",
    })
    assert wrong.status_code == 401

    changed = client.post("/api/v1/auth/change-password", headers=auth, json={
        "current_password": "a-long-test-password-123",
        "new_password": "a-different-long-password-456",
    })
    assert changed.status_code == 204

    old_refresh = client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert old_refresh.status_code == 401
    assert login(client, "password-change@example.com").status_code == 401
    assert login(client, "password-change@example.com", "a-different-long-password-456").status_code == 200


def test_user_can_list_and_revoke_only_their_session_family(client):
    assert register(client, "sessions@example.com").status_code == 201
    tokens = login(client, "sessions@example.com").json()
    auth = {"Authorization": f"Bearer {tokens['access_token']}"}

    sessions = client.get("/api/v1/users/me/sessions", headers=auth)
    assert sessions.status_code == 200, sessions.text
    assert len(sessions.json()) == 1
    family_id = sessions.json()[0]["family_id"]

    revoked = client.delete(f"/api/v1/users/me/sessions/{family_id}", headers=auth)
    assert revoked.status_code == 204
    refresh = client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert refresh.status_code == 401

    missing = client.delete(
        "/api/v1/users/me/sessions/00000000-0000-0000-0000-000000000001", headers=auth
    )
    assert missing.status_code == 404


def test_last_admin_cannot_be_demoted_even_by_bootstrap_key(client):
    from platform_app.models import User
    from sqlalchemy import select

    assert register(client, "only-admin@example.com").status_code == 201
    tokens = login(client, "only-admin@example.com").json()
    user_id = tokens["user"]["id"]
    promoted = client.post(
        f"/api/v1/admin/bootstrap/{user_id}/role",
        headers={"X-Master-Key": MASTER},
        json={"role": "admin"},
    )
    assert promoted.status_code == 200, promoted.text

    demoted = client.post(
        f"/api/v1/admin/bootstrap/{user_id}/role",
        headers={"X-Master-Key": MASTER},
        json={"role": "user"},
    )
    assert demoted.status_code == 409
    assert demoted.json()["error"]["code"] == "last_admin_required"


def test_admin_can_demote_another_admin_while_one_admin_remains(client):
    assert register(client, "admin-one@example.com").status_code == 201
    first_tokens = login(client, "admin-one@example.com").json()
    first_id = first_tokens["user"]["id"]
    first_auth = {"Authorization": f"Bearer {first_tokens['access_token']}"}
    assert client.post(
        f"/api/v1/admin/bootstrap/{first_id}/role",
        headers={"X-Master-Key": MASTER}, json={"role": "admin"},
    ).status_code == 200

    assert register(client, "admin-two@example.com").status_code == 201
    second_tokens = login(client, "admin-two@example.com").json()
    second_id = second_tokens["user"]["id"]
    assert client.post(
        f"/api/v1/admin/bootstrap/{second_id}/role",
        headers={"X-Master-Key": MASTER}, json={"role": "admin"},
    ).status_code == 200

    demoted = client.patch(
        f"/api/v1/admin/users/{second_id}/role",
        headers=first_auth, json={"role": "user"},
    )
    assert demoted.status_code == 200, demoted.text
    assert demoted.json()["role"] == "user"
