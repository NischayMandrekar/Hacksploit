import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.infrastructure.database import Base
from app.modules.graph.community import CommunityDetector
from app.modules.graph.engine import RelationalIdentityGraphEngine

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
def db_session():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def graph_engine(db_session):
    return RelationalIdentityGraphEngine(db=db_session)


@pytest.fixture
def community_detector(graph_engine):
    return CommunityDetector(engine=graph_engine, min_ring_size=2, ring_density_threshold=0.30)


def test_isolated_clean_applicant(graph_engine, community_detector):
    """Verify that an isolated applicant with no shared links is identified as ISOLATED_CLEAN."""
    graph_engine.add_identity_node(
        "app_single_001",
        attributes={"phone": "+91 91111 00000", "status": "approved", "risk_score": 0.04},
    )

    summary = community_detector.detect_community("app_single_001")
    assert summary.is_ring is False
    assert len(summary.applicant_members) == 1
    assert summary.applicant_members[0].applicant_id == "app_single_001"
    assert summary.fraud_prevalence == 0.0
    assert summary.cycle_count == 0
    assert summary.reason_codes == ["ISOLATED_CLEAN"]


def test_dense_synthetic_fraud_ring(graph_engine, community_detector):
    """
    Verify detection of a coordinated 3-applicant fraud ring with circular attribute swapping
    (App1-Phone1-App2-Device2-App3-Email3-App1).
    """
    phone_1 = "+91 92222 11111"
    device_2 = "emulator-canvas-ring-002"
    email_3 = "shared_ring_mule@domain.com"

    # App1 shares Phone1 with App2, and Email3 with App3
    graph_engine.add_identity_node(
        "app_ring_A",
        attributes={"phone": phone_1, "email": email_3, "risk_score": 0.60},
    )
    # App2 shares Phone1 with App1, and Device2 with App3
    graph_engine.add_identity_node(
        "app_ring_B",
        attributes={"phone": phone_1, "device_fingerprint": device_2, "risk_score": 0.70},
    )
    # App3 shares Device2 with App2, and Email3 with App1 (Completes the circular ring)
    graph_engine.add_identity_node(
        "app_ring_C",
        attributes={"device_fingerprint": device_2, "email": email_3, "risk_score": 0.65},
    )

    summary = community_detector.detect_community("app_ring_A", depth=3)

    assert summary.is_ring is True
    assert summary.cycle_count >= 1
    assert len(summary.applicant_members) == 3
    member_ids = {m.applicant_id for m in summary.applicant_members}
    assert member_ids == {"app_ring_A", "app_ring_B", "app_ring_C"}
    assert "HIGH_DENSITY_RING" in summary.reason_codes
    assert "CIRCULAR_ATTRIBUTE_SWAP" in summary.reason_codes


def test_fraud_contagion_detection(graph_engine, community_detector):
    """
    Verify that an applicant sharing identifiers with a confirmed fraudster
    is flagged for fraud contagion risk.
    """
    shared_device = "burner-phone-canvas-999"

    # Known fraudster
    graph_engine.add_identity_node(
        "app_bad_actor",
        attributes={"device_fingerprint": shared_device, "is_fraud": True, "risk_score": 0.98},
    )
    # New applicant sharing the device
    graph_engine.add_identity_node(
        "app_new_victim_or_mule",
        attributes={"device_fingerprint": shared_device, "risk_score": 0.10},
    )

    summary = community_detector.detect_community("app_new_victim_or_mule")

    assert summary.is_ring is True
    assert summary.fraud_prevalence == 0.50
    assert "FRAUD_CONTAGION_RISK" in summary.reason_codes


def test_benign_community_isolation(graph_engine, community_detector):
    """
    Verify that benign family members sharing only 1 phone number
    are classified as BENIGN_COMMUNITY and not falsely branded as a fraud ring.
    """
    home_phone = "+91 93333 44444"

    graph_engine.add_identity_node(
        "app_parent",
        attributes={"phone": home_phone, "device_fingerprint": "parent_ipad", "risk_score": 0.05},
    )
    graph_engine.add_identity_node(
        "app_child",
        attributes={"phone": home_phone, "device_fingerprint": "child_laptop", "risk_score": 0.05},
    )

    summary = community_detector.detect_community("app_parent")

    assert summary.is_ring is False
    assert len(summary.applicant_members) == 2
    assert summary.cycle_count == 0
    assert summary.fraud_prevalence == 0.0
    assert summary.reason_codes == ["BENIGN_COMMUNITY"]


def test_batch_community_detection(graph_engine, community_detector, db_session):
    """
    Verify that detect_all_communities groups multi-applicant clusters across the database
    while ignoring single isolated applicants.
    """
    # Cluster 1: Ring of 2 applicants
    graph_engine.add_identity_node("app_c1_a", attributes={"phone": "+91 94444 11111"})
    graph_engine.add_identity_node("app_c1_b", attributes={"phone": "+91 94444 11111"})

    # Cluster 2: Ring of 2 applicants
    graph_engine.add_identity_node("app_c2_a", attributes={"phone": "+91 94444 22222"})
    graph_engine.add_identity_node("app_c2_b", attributes={"phone": "+91 94444 22222"})

    # Isolated clean applicants
    for i in range(1, 4):
        graph_engine.add_identity_node(f"app_clean_{i}", attributes={"phone": f"+91 95555 0000{i}"})

    all_clusters = community_detector.detect_all_communities(db=db_session)

    # Should detect exactly the 2 multi-applicant communities
    assert len(all_clusters) == 2
    cluster_member_sets = [{m.applicant_id for m in c.applicant_members} for c in all_clusters]
    assert {"app_c1_a", "app_c1_b"} in cluster_member_sets
    assert {"app_c2_a", "app_c2_b"} in cluster_member_sets
