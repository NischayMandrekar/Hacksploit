import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.infrastructure.database import Base, get_db
from app.infrastructure.storage.artifact_store import LocalDiskArtifactStore
from app.main import app
from app.modules.graph.models import HEX_64_PATTERN, ApplicantEdge, ApplicantNode
from app.modules.kyc.models import KYCApplication
from app.modules.kyc.router import get_artifact_store

# In-memory SQLite engine with StaticPool
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
def setup_test_db(tmp_path):
    artifact_dir = tmp_path / "test_artifacts"
    store = LocalDiskArtifactStore(base_dir=str(artifact_dir))

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_artifact_store] = lambda: store

    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


# Sample valid headers
VALID_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00" + b"\x00" * 128
VALID_PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00" + b"\x00" * 128


def test_kyc_application_ingestion_success(client):
    """
    Verifies full synchronous ingestion flow:
    - Multipart form upload with document & selfie.
    - Magic-byte verification and artifact persistence.
    - PII pseudonymization (salted HMAC-SHA256).
    - Database record persistence for KYCApplication, ApplicantNode, and 1-hop edges.
    """
    form_data = {
        "legal_name": "Arjun Sharma",
        "dob": "1992-07-24",
        "phone": "+919876543210",
        "email": "arjun.sharma@example.com",
        "id_type": "aadhaar",
        "id_number": "123456789012",
        "device_fingerprint": "dev_canvas_fp_9876543210_abc",
        "account_type": "savings",
        "applicant_id": "app_arjun_01",
    }

    files = {
        "document_file": ("aadhaar_front.jpg", io.BytesIO(VALID_JPEG), "image/jpeg"),
        "selfie_file": ("webcam_selfie.png", io.BytesIO(VALID_PNG), "image/png"),
    }

    response = client.post("/api/v1/kyc/applications", data=form_data, files=files)
    assert response.status_code == 201, response.text
    data = response.json()

    assert data["applicant_id"] == "app_arjun_01"
    assert data["account_type"] == "savings"
    assert data["status"] in ("screening", "approved")
    assert data["risk_tier"] == "APPROVE"
    assert data["risk_score"] is not None
    assert data["document_artifact_uri"].startswith("file://")
    assert data["selfie_artifact_uri"].startswith("file://")

    # Verify Database state
    db = TestingSessionLocal()
    try:
        # 1. KYCApplication record
        app_rec = db.execute(
            select(KYCApplication).where(KYCApplication.applicant_id == "app_arjun_01")
        ).scalar_one()

        assert app_rec.claimed_name == "Arjun Sharma"
        assert app_rec.claimed_dob == "1992-07-24"
        assert app_rec.id_number_last4 == "9012"
        # Verify hashes are 64-char hex digests, never raw values
        assert HEX_64_PATTERN.match(app_rec.identity_hash)
        assert app_rec.identity_hash != "123456789012"
        assert HEX_64_PATTERN.match(app_rec.phone_hash)
        assert app_rec.phone_hash != "+919876543210"
        assert HEX_64_PATTERN.match(app_rec.email_hash)
        assert app_rec.email_hash != "arjun.sharma@example.com"
        assert HEX_64_PATTERN.match(app_rec.device_fingerprint_hash)

        # 2. ApplicantNode in identity graph
        node = db.execute(
            select(ApplicantNode).where(ApplicantNode.applicant_id == "app_arjun_01")
        ).scalar_one()
        assert node.identity_hash == app_rec.identity_hash
        assert node.phone_hash == app_rec.phone_hash
        assert node.email_hash == app_rec.email_hash
        assert node.device_fingerprint_hash == app_rec.device_fingerprint_hash

        # 3. ApplicantEdges in identity graph
        edges = db.execute(
            select(ApplicantEdge).where(ApplicantEdge.source_hash == "app_arjun_01")
        ).scalars().all()
        edge_types = {e.edge_type for e in edges}
        assert "HAS_IDENTITY" in edge_types
        assert "HAS_PHONE" in edge_types
        assert "HAS_EMAIL" in edge_types
        assert "HAS_DEVICE" in edge_types
        assert len(edges) >= 4
    finally:
        db.close()


def test_magic_bytes_mismatch_rejection(client):
    """
    Verifies that spoofed files (e.g. text file pretending to be image/jpeg)
    are rejected with HTTP 400 and magic_bytes_mismatch code.
    """
    form_data = {
        "legal_name": "Spoofed Attacker",
        "dob": "1995-01-01",
        "phone": "+919111111111",
        "email": "attacker@darkweb.org",
        "id_type": "aadhaar",
        "id_number": "999988887777",
        "device_fingerprint": "spoofed_browser_profile_01",
    }

    # Text content disguised as image/jpeg
    fake_jpeg = b"<!DOCTYPE html><html><body>Spoofed attack payload</body></html>"
    files = {
        "document_file": ("malicious_doc.jpg", io.BytesIO(fake_jpeg), "image/jpeg"),
        "selfie_file": ("legit_selfie.png", io.BytesIO(VALID_PNG), "image/png"),
    }

    response = client.post("/api/v1/kyc/applications", data=form_data, files=files)
    assert response.status_code == 400
    err = response.json()["error"]
    assert err["code"] == "magic_bytes_mismatch"
    assert "File contents do not match declared MIME type" in err["message"]

    # Verify nothing was saved to DB
    db = TestingSessionLocal()
    try:
        count = db.execute(select(KYCApplication)).scalars().all()
        assert len(count) == 0
    finally:
        db.close()


def test_empty_file_rejection(client):
    """Verifies that 0-byte uploads are rejected with HTTP 400."""
    form_data = {
        "legal_name": "Zero Byte User",
        "dob": "1990-10-10",
        "phone": "+919222222222",
        "email": "zerobyte@example.com",
        "id_type": "pan",
        "id_number": "ABCDE1234F",
        "device_fingerprint": "clean_device_profile_02",
    }

    files = {
        "document_file": ("empty.jpg", io.BytesIO(b""), "image/jpeg"),
        "selfie_file": ("selfie.png", io.BytesIO(VALID_PNG), "image/png"),
    }

    response = client.post("/api/v1/kyc/applications", data=form_data, files=files)
    assert response.status_code == 400
    err = response.json()["error"]
    assert err["code"] == "empty_document"


def test_oversized_file_rejection(client, monkeypatch):
    """Verifies that uploads exceeding the max size limit are rejected with HTTP 400."""
    # Temporarily set max bytes to 500 bytes for test
    monkeypatch.setattr(settings, "artifact_max_bytes", 500)

    form_data = {
        "legal_name": "Oversized User",
        "dob": "1990-10-10",
        "phone": "+919333333333",
        "email": "oversized@example.com",
        "id_type": "passport",
        "id_number": "Z1234567",
        "device_fingerprint": "clean_device_profile_03",
    }

    large_jpeg = VALID_JPEG + b"\x00" * 1000
    files = {
        "document_file": ("large.jpg", io.BytesIO(large_jpeg), "image/jpeg"),
        "selfie_file": ("selfie.png", io.BytesIO(VALID_PNG), "image/png"),
    }

    response = client.post("/api/v1/kyc/applications", data=form_data, files=files)
    assert response.status_code == 400
    err = response.json()["error"]
    assert err["code"] == "document_too_large"


def test_unsupported_mime_type_rejection(client):
    """Verifies that disallowed MIME types (e.g. text/plain or exe) are rejected."""
    form_data = {
        "legal_name": "Unsupported MIME User",
        "dob": "1990-10-10",
        "phone": "+919444444444",
        "email": "mime@example.com",
        "id_type": "aadhaar",
        "id_number": "111122223333",
        "device_fingerprint": "clean_device_profile_04",
    }

    files = {
        "document_file": ("notes.txt", io.BytesIO(b"just text file"), "text/plain"),
        "selfie_file": ("selfie.png", io.BytesIO(VALID_PNG), "image/png"),
    }

    response = client.post("/api/v1/kyc/applications", data=form_data, files=files)
    assert response.status_code == 400
    err = response.json()["error"]
    assert err["code"] == "unsupported_mime_type"


def test_pydantic_schema_validation_rejections(client):
    """Verifies that invalid demographic fields (malformed email, phone, DOB) fail validation."""
    # 1. Invalid email
    bad_email = {
        "legal_name": "Test User",
        "dob": "1990-10-10",
        "phone": "+919876543210",
        "email": "not-an-email",
        "id_type": "aadhaar",
        "id_number": "123456789012",
        "device_fingerprint": "device_fp_01",
    }
    files = {
        "document_file": ("doc.jpg", io.BytesIO(VALID_JPEG), "image/jpeg"),
        "selfie_file": ("selfie.png", io.BytesIO(VALID_PNG), "image/png"),
    }
    res = client.post("/api/v1/kyc/applications", data=bad_email, files=files)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "schema_validation_error"

    # 2. Invalid phone (too short)
    bad_phone = dict(bad_email)
    bad_phone["email"] = "valid@example.com"
    bad_phone["phone"] = "123"
    files = {
        "document_file": ("doc.jpg", io.BytesIO(VALID_JPEG), "image/jpeg"),
        "selfie_file": ("selfie.png", io.BytesIO(VALID_PNG), "image/png"),
    }
    res2 = client.post("/api/v1/kyc/applications", data=bad_phone, files=files)
    assert res2.status_code == 400
    assert res2.json()["error"]["code"] == "schema_validation_error"

    # 3. Future DOB
    bad_dob = dict(bad_email)
    bad_dob["email"] = "valid@example.com"
    bad_dob["dob"] = "2099-01-01"
    files = {
        "document_file": ("doc.jpg", io.BytesIO(VALID_JPEG), "image/jpeg"),
        "selfie_file": ("selfie.png", io.BytesIO(VALID_PNG), "image/png"),
    }
    res3 = client.post("/api/v1/kyc/applications", data=bad_dob, files=files)
    assert res3.status_code == 400
    assert res3.json()["error"]["code"] == "schema_validation_error"


def test_duplicate_applicant_id_rejection(client):
    """Verifies that submitting the same applicant_id twice returns HTTP 409."""
    form_data = {
        "legal_name": "Original Applicant",
        "dob": "1988-03-14",
        "phone": "+919555555555",
        "email": "original@example.com",
        "id_type": "aadhaar",
        "id_number": "555566667777",
        "device_fingerprint": "unique_device_profile_55",
        "applicant_id": "duplicate_test_id_100",
    }
    files = {
        "document_file": ("doc.jpg", io.BytesIO(VALID_JPEG), "image/jpeg"),
        "selfie_file": ("selfie.png", io.BytesIO(VALID_PNG), "image/png"),
    }
    res1 = client.post("/api/v1/kyc/applications", data=form_data, files=files)
    assert res1.status_code == 201

    # Second submission with same applicant_id
    files2 = {
        "document_file": ("doc.jpg", io.BytesIO(VALID_JPEG), "image/jpeg"),
        "selfie_file": ("selfie.png", io.BytesIO(VALID_PNG), "image/png"),
    }
    res2 = client.post("/api/v1/kyc/applications", data=form_data, files=files2)
    assert res2.status_code == 409
    assert res2.json()["error"]["code"] == "duplicate_application"


def test_m0_rule_screening_trigger_on_ingestion(client):
    """
    Verifies that M0 fast-path screening runs synchronously:
    - App 1 submits cleanly.
    - App 2 submits with App 1's Aadhaar -> triggers IDENTITY_REUSE_COLLISION -> REJECT.
    - App 3 submits with App 1's phone -> triggers PHONE_REUSE_MODERATE -> REVIEW.
    """
    shared_aadhaar = "888899990000"
    shared_phone = "+919777777777"

    # 1. Clean App 1
    app1_data = {
        "legal_name": "Person One",
        "dob": "1990-01-01",
        "phone": shared_phone,
        "email": "person1@example.com",
        "id_type": "aadhaar",
        "id_number": shared_aadhaar,
        "device_fingerprint": "device_fp_person1",
        "applicant_id": "app_person_1",
    }
    files1 = {
        "document_file": ("doc.jpg", io.BytesIO(VALID_JPEG), "image/jpeg"),
        "selfie_file": ("selfie.png", io.BytesIO(VALID_PNG), "image/png"),
    }
    res1 = client.post("/api/v1/kyc/applications", data=app1_data, files=files1)
    assert res1.status_code == 201
    assert res1.json()["risk_tier"] == "APPROVE"

    # 2. App 2 reuses App 1's Aadhaar (Identity Collision -> REJECT)
    app2_data = {
        "legal_name": "Person Two Impersonator",
        "dob": "1991-02-02",
        "phone": "+919888888888",
        "email": "person2@example.com",
        "id_type": "aadhaar",
        "id_number": shared_aadhaar,  # COLLISION
        "device_fingerprint": "device_fp_person2",
        "applicant_id": "app_person_2",
    }
    files2 = {
        "document_file": ("doc.jpg", io.BytesIO(VALID_JPEG), "image/jpeg"),
        "selfie_file": ("selfie.png", io.BytesIO(VALID_PNG), "image/png"),
    }
    res2 = client.post("/api/v1/kyc/applications", data=app2_data, files=files2)
    assert res2.status_code == 201
    rec2 = res2.json()
    assert rec2["risk_tier"] == "REJECT"
    assert rec2["status"] == "rejected"
    assert "IDENTITY_REUSE_COLLISION" in rec2["reason_codes"]

    # 3. App 3 reuses shared_phone (Moderate degree 2 -> REVIEW)
    app3_data = {
        "legal_name": "Person Three Family Member",
        "dob": "1993-03-03",
        "phone": shared_phone,  # Shared with person 1
        "email": "person3@example.com",
        "id_type": "aadhaar",
        "id_number": "111100001111",
        "device_fingerprint": "device_fp_person3",
        "applicant_id": "app_person_3",
    }
    files3 = {
        "document_file": ("doc.jpg", io.BytesIO(VALID_JPEG), "image/jpeg"),
        "selfie_file": ("selfie.png", io.BytesIO(VALID_PNG), "image/png"),
    }
    res3 = client.post("/api/v1/kyc/applications", data=app3_data, files=files3)
    assert res3.status_code == 201
    rec3 = res3.json()
    assert rec3["risk_tier"] == "REVIEW"
    assert rec3["status"] == "under_review"
    assert "PHONE_REUSE_MODERATE" in rec3["reason_codes"]


def test_get_application_detail_and_list(client):
    """Verifies GET /api/v1/kyc/applications/{id} and GET /api/v1/kyc/applications."""
    form_data = {
        "legal_name": "Siddharth Verma",
        "dob": "1985-06-20",
        "phone": "+919666666666",
        "email": "siddharth@example.com",
        "id_type": "pan",
        "id_number": "ABCDE9876K",
        "device_fingerprint": "dev_siddharth_profile",
        "applicant_id": "app_sid_verma",
    }
    files = {
        "document_file": ("pan.jpg", io.BytesIO(VALID_JPEG), "image/jpeg"),
        "selfie_file": ("selfie.png", io.BytesIO(VALID_PNG), "image/png"),
    }
    client.post("/api/v1/kyc/applications", data=form_data, files=files)

    # 1. Get detail
    res_get = client.get("/api/v1/kyc/applications/app_sid_verma")
    assert res_get.status_code == 200
    detail = res_get.json()
    assert detail["applicant_id"] == "app_sid_verma"
    assert detail["claimed_name"] == "Siddharth Verma"
    assert detail["id_number_last4"] == "876K"
    assert detail["id_type"] == "pan"

    # 2. Get non-existent
    res_404 = client.get("/api/v1/kyc/applications/non_existent_app_id")
    assert res_404.status_code == 404
    assert res_404.json()["error"]["code"] == "applicant_not_found"

    # 3. List applications
    res_list = client.get("/api/v1/kyc/applications")
    assert res_list.status_code == 200
    items = res_list.json()
    assert len(items) >= 1
    assert any(item["applicant_id"] == "app_sid_verma" for item in items)
