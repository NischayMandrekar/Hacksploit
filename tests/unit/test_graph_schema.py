import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import hash_pii
from app.infrastructure.database import Base
from app.modules.graph.models import (
    ApplicantEdge,
    ApplicantNode,
    EdgeType,
    NodeType,
)

# In-memory SQLite engine with StaticPool for unit test isolation
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
def db_session():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_table_creation_and_indexes():
    """Verify that applicant_nodes and applicant_edges tables and indices are created."""
    tables = Base.metadata.tables
    assert "applicant_nodes" in tables
    assert "applicant_edges" in tables

    edge_table = tables["applicant_edges"]
    index_names = {idx.name for idx in edge_table.indexes}
    assert "ix_applicant_edges_source_edge_type" in index_names
    assert "ix_applicant_edges_target_edge_type" in index_names
    assert "ix_applicant_edges_source_target" in index_names

    node_table = tables["applicant_nodes"]
    node_index_names = {idx.name for idx in node_table.indexes}
    assert "ix_applicant_nodes_applicant_id" in node_index_names
    assert "ix_applicant_nodes_identity_hash" in node_index_names
    assert "ix_applicant_nodes_device_hash" in node_index_names


def test_applicant_node_persistence_with_valid_hashes(db_session):
    """Verify adding an ApplicantNode with salted HMAC-SHA256 digests persists correctly."""
    phone_hash = hash_pii("+919876543210", is_numeric_id=True)
    identity_hash = hash_pii("123456789012", is_numeric_id=True)
    device_hash = hash_pii("device-fingerprint-random-seed-xyz")

    node = ApplicantNode(
        applicant_id="app_1001",
        node_type=NodeType.APPLICANT.value,
        identity_hash=identity_hash,
        phone_hash=phone_hash,
        device_fingerprint_hash=device_hash,
        status="pending",
        risk_score=0.15,
        properties_json={"application_channel": "mobile_web", "income_declared": 75000},
    )
    db_session.add(node)
    db_session.commit()

    retrieved = db_session.scalar(
        select(ApplicantNode).where(ApplicantNode.applicant_id == "app_1001")
    )
    assert retrieved is not None
    assert retrieved.applicant_id == "app_1001"
    assert retrieved.phone_hash == phone_hash
    assert retrieved.identity_hash == identity_hash
    assert retrieved.device_fingerprint_hash == device_hash
    assert retrieved.status == "pending"
    assert retrieved.risk_score == 0.15
    assert retrieved.properties_json["application_channel"] == "mobile_web"

    # Verify serialization
    data = retrieved.to_dict()
    assert data["applicant_id"] == "app_1001"
    assert data["phone_hash"] == phone_hash
    assert "created_at" in data
    assert data["created_at"] is not None


def test_applicant_node_rejects_raw_pii():
    """Verify that attempting to assign raw PII to identifier fields raises ValueError."""
    # Raw email containing @ and not 64 hex chars
    with pytest.raises(ValueError, match="HMAC-SHA256 digest; storing raw PII is strictly prohibited"):
        ApplicantNode(
            applicant_id="app_1002",
            email_hash="user@example.com",
        )

    # Raw 12-digit Aadhaar number
    with pytest.raises(ValueError, match="HMAC-SHA256 digest; storing raw PII is strictly prohibited"):
        ApplicantNode(
            applicant_id="app_1003",
            identity_hash="123456789012",
        )

    # Raw phone number with plus sign
    with pytest.raises(ValueError, match="HMAC-SHA256 digest; storing raw PII is strictly prohibited"):
        ApplicantNode(
            applicant_id="app_1004",
            phone_hash="+919876543210",
        )


def test_applicant_edge_persistence_and_indexed_lookup(db_session):
    """
    Verify adding an ApplicantEdge persists and can be queried via composite index
    (source_hash, edge_type) and (target_hash, edge_type).
    """
    applicant_id = "app_2001"
    phone_hash = hash_pii("+919999988888", is_numeric_id=True)
    device_hash = hash_pii("device-canvas-fingerprint-abc")

    # Add edges: Applicant -> HAS_PHONE -> PhoneHash, Applicant -> HAS_DEVICE -> DeviceHash
    edge_phone = ApplicantEdge(
        source_hash=applicant_id,
        target_hash=phone_hash,
        edge_type=EdgeType.HAS_PHONE.value,
        source_type=NodeType.APPLICANT.value,
        target_type=NodeType.PHONE.value,
        weight=1.0,
        properties_json={"carrier": "Jio", "verified_via_otp": True},
    )
    edge_device = ApplicantEdge(
        source_hash=applicant_id,
        target_hash=device_hash,
        edge_type=EdgeType.HAS_DEVICE.value,
        source_type=NodeType.APPLICANT.value,
        target_type=NodeType.DEVICE.value,
        weight=1.0,
    )

    db_session.add_all([edge_phone, edge_device])
    db_session.commit()

    # 1. 1-hop lookup by source_hash and edge_type (forward hop)
    phone_edges = db_session.scalars(
        select(ApplicantEdge).where(
            ApplicantEdge.source_hash == applicant_id,
            ApplicantEdge.edge_type == EdgeType.HAS_PHONE.value,
        )
    ).all()
    assert len(phone_edges) == 1
    assert phone_edges[0].target_hash == phone_hash
    assert phone_edges[0].properties_json["carrier"] == "Jio"

    # 2. 1-hop reverse lookup by target_hash and edge_type (finding shared phone across applicants)
    second_applicant_id = "app_2002"
    edge_shared = ApplicantEdge(
        source_hash=second_applicant_id,
        target_hash=phone_hash,
        edge_type=EdgeType.HAS_PHONE.value,
        source_type=NodeType.APPLICANT.value,
        target_type=NodeType.PHONE.value,
    )
    db_session.add(edge_shared)
    db_session.commit()

    sharing_applicants = db_session.scalars(
        select(ApplicantEdge.source_hash).where(
            ApplicantEdge.target_hash == phone_hash,
            ApplicantEdge.edge_type == EdgeType.HAS_PHONE.value,
        )
    ).all()
    assert set(sharing_applicants) == {"app_2001", "app_2002"}


def test_applicant_edge_unique_constraint(db_session):
    """Verify that duplicate (source_hash, target_hash, edge_type) rows are rejected."""
    edge1 = ApplicantEdge(
        source_hash="app_3001",
        target_hash="identifier_hash_1",
        edge_type=EdgeType.HAS_EMAIL.value,
    )
    edge2 = ApplicantEdge(
        source_hash="app_3001",
        target_hash="identifier_hash_1",
        edge_type=EdgeType.HAS_EMAIL.value,
    )
    db_session.add(edge1)
    db_session.commit()

    db_session.add(edge2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_applicant_edge_rejects_raw_pii():
    """Verify that edge endpoints reject obvious raw PII strings (emails, plus phones)."""
    with pytest.raises(ValueError, match="raw unhashed PII"):
        ApplicantEdge(
            source_hash="app_4001",
            target_hash="raw_email@domain.com",
            edge_type=EdgeType.HAS_EMAIL.value,
        )

    with pytest.raises(ValueError, match="raw unhashed PII"):
        ApplicantEdge(
            source_hash="+91 98765 43210",
            target_hash="some_hash",
            edge_type=EdgeType.HAS_PHONE.value,
        )
