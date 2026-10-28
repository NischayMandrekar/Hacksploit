import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.infrastructure.database import Base, get_db
from app.main import app

# In-memory SQLite engine with StaticPool so all connections share the same memory database
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(
    bind=test_engine, autoflush=False, expire_on_commit=False
)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_test_db():
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_user_registration_and_bootstrap_admin(client):
    # First user should be auto-promoted to admin
    res1 = client.post(
        "/api/v1/auth/register",
        json={"email": "first_admin@argus.ai", "password": "supersecretpassword123"},
    )
    assert res1.status_code == 201
    user1 = res1.json()
    assert user1["email"] == "first_admin@argus.ai"
    assert user1["role"] == "admin"

    # Second user should have standard "user" role
    res2 = client.post(
        "/api/v1/auth/register",
        json={"email": "regular_applicant@argus.ai", "password": "supersecretpassword123"},
    )
    assert res2.status_code == 201
    user2 = res2.json()
    assert user2["email"] == "regular_applicant@argus.ai"
    assert user2["role"] == "user"

    # Duplicate email should be rejected
    res_dup = client.post(
        "/api/v1/auth/register",
        json={"email": "first_admin@argus.ai", "password": "differentpassword123"},
    )
    assert res_dup.status_code == 409
    assert res_dup.json()["error"]["code"] == "email_taken"


def test_login_and_token_issue(client):
    client.post(
        "/api/v1/auth/register",
        json={"email": "analyst@argus.ai", "password": "strongpassword123"},
    )

    # Valid login
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "analyst@argus.ai", "password": "strongpassword123"},
    )
    assert res.status_code == 200
    tokens = res.json()
    assert "access_token" in tokens
    assert "refresh_token" in tokens
    assert tokens["token_type"] == "Bearer"

    # Invalid login
    res_bad = client.post(
        "/api/v1/auth/login",
        json={"email": "analyst@argus.ai", "password": "wrongpassword!"},
    )
    assert res_bad.status_code == 401
    assert res_bad.json()["error"]["code"] == "invalid_credentials"


def test_authenticated_profile_endpoint(client):
    client.post(
        "/api/v1/auth/register",
        json={"email": "user@argus.ai", "password": "strongpassword123"},
    )
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": "user@argus.ai", "password": "strongpassword123"},
    )
    access_token = login_res.json()["access_token"]

    # Valid profile fetch
    res = client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert res.status_code == 200
    assert res.json()["email"] == "user@argus.ai"

    # Missing or invalid token
    res_no_auth = client.get("/api/v1/users/me")
    assert res_no_auth.status_code == 401


def test_refresh_token_rotation(client):
    client.post(
        "/api/v1/auth/register",
        json={"email": "rotation_test@argus.ai", "password": "strongpassword123"},
    )
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": "rotation_test@argus.ai", "password": "strongpassword123"},
    )
    rfr_token_1 = login_res.json()["refresh_token"]

    # First rotation
    res_rot = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": rfr_token_1},
    )
    assert res_rot.status_code == 200
    rfr_token_2 = res_rot.json()["refresh_token"]
    assert rfr_token_2 != rfr_token_1

    # Immediate replay within 30-second grace window should return valid tokens without revoking
    res_grace = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": rfr_token_1},
    )
    assert res_grace.status_code == 200


def test_role_based_access_control(client):
    # Register first user -> admin
    admin_login = client.post(
        "/api/v1/auth/register",
        json={"email": "admin@argus.ai", "password": "adminpassword123"},
    )
    assert "id" in admin_login.json()
    admin_token = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@argus.ai", "password": "adminpassword123"},
    ).json()["access_token"]

    # Register second user -> user
    user_res = client.post(
        "/api/v1/auth/register",
        json={"email": "target@argus.ai", "password": "userpassword123"},
    )
    user_id = user_res.json()["id"]
    user_token = client.post(
        "/api/v1/auth/login",
        json={"email": "target@argus.ai", "password": "userpassword123"},
    ).json()["access_token"]

    # Standard user attempting to modify role should receive 403
    forbidden_res = client.post(
        f"/api/v1/admin/users/{user_id}/role",
        headers={"Authorization": f"Bearer {user_token}"},
        json={"role": "analyst"},
    )
    assert forbidden_res.status_code == 403
    assert forbidden_res.json()["error"]["code"] == "admin_required"

    # Admin updating target user to analyst should succeed
    promote_res = client.post(
        f"/api/v1/admin/users/{user_id}/role",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"role": "analyst"},
    )
    assert promote_res.status_code == 200
    assert promote_res.json()["role"] == "analyst"


def test_api_key_management(client):
    client.post(
        "/api/v1/auth/register",
        json={"email": "service_owner@argus.ai", "password": "ownerpassword123"},
    )
    token = client.post(
        "/api/v1/auth/login",
        json={"email": "service_owner@argus.ai", "password": "ownerpassword123"},
    ).json()["access_token"]

    # Create API key
    res = client.post(
        "/api/v1/api-keys",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "test-service-key", "scopes": ["kyc:submit", "kyc:read"]},
    )
    assert res.status_code == 201
    issued = res.json()
    assert "raw_key" in issued
    assert issued["name"] == "test-service-key"
    key_id = issued["id"]

    # Revoke API key
    del_res = client.delete(
        f"/api/v1/api-keys/{key_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert del_res.status_code == 204
