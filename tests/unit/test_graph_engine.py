import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import hash_pii
from app.infrastructure.database import Base
from app.modules.graph.engine import RelationalIdentityGraphEngine
from app.modules.graph.models import EdgeType

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


def test_add_identity_node_and_edges(graph_engine, db_session):
    """Verify adding an applicant node automatically hashes PII and creates typed edges."""
    applicant_id = "app_test_001"
    raw_phone = "+91 98765 43210"
    raw_email = "applicant@testdomain.com"
    raw_aadhaar = "1234-5678-9012"
    raw_device = "canvas-fingerprint-xyz-123"

    graph_engine.add_identity_node(
        applicant_id=applicant_id,
        attributes={
            "phone": raw_phone,
            "email": raw_email,
            "aadhaar": raw_aadhaar,
            "device_fingerprint": raw_device,
            "status": "pending",
            "risk_score": 0.25,
            "properties": {"city": "Bengaluru"},
        },
    )

    subnetwork = graph_engine.find_linked_clusters(applicant_id, depth=1)
    assert subnetwork.root_id == applicant_id
    assert len(subnetwork.nodes) == 5  # Applicant + 4 identifiers (phone, email, identity, device)
    assert len(subnetwork.edges) == 4

    expected_phone_hash = hash_pii(raw_phone, is_numeric_id=True)
    assert any(n["id"] == expected_phone_hash for n in subnetwork.nodes)


def test_isolated_applicant_ego_network(graph_engine):
    """Verify that an applicant with no neighbors returns a clean bounded subnetwork."""
    applicant_id = "app_isolated_001"
    graph_engine.add_identity_node(
        applicant_id=applicant_id,
        attributes={"status": "approved", "risk_score": 0.05},
    )

    features = graph_engine.compute_graph_features(applicant_id)
    assert features.shared_phone_degree == 0.0
    assert features.shared_device_degree == 0.0
    assert features.shared_ssn_degree == 0.0
    assert features.ego_node_count == 1.0
    assert features.ego_edge_count == 0.0
    assert features.ego_density == 0.0
    assert features.max_neighbor_risk == 0.0
    assert features.cycle_count == 0.0


def test_2hop_cluster_extraction_isolation(graph_engine):
    """
    Verify that 2-hop extraction boundedly pulls only connected cluster members
    and does NOT leak or pull unrelated clusters.
    """
    # Cluster A: App_A1 and App_A2 share phone_A. App_A2 also has device_A.
    graph_engine.add_identity_node(
        "app_A1",
        attributes={"phone": "+91 99999 11111", "risk_score": 0.1},
    )
    graph_engine.add_identity_node(
        "app_A2",
        attributes={
            "phone": "+91 99999 11111",
            "device_fingerprint": "device-cluster-a",
            "risk_score": 0.3,
        },
    )

    # Cluster B: App_B1 and App_B2 share phone_B (completely unrelated)
    graph_engine.add_identity_node(
        "app_B1",
        attributes={"phone": "+91 88888 22222", "risk_score": 0.2},
    )
    graph_engine.add_identity_node(
        "app_B2",
        attributes={"phone": "+91 88888 22222", "risk_score": 0.4},
    )

    # 1. Query 2-hop subnetwork from app_A1 (contains shared phone and app_A2)
    subnetwork_2hop = graph_engine.find_linked_clusters("app_A1", depth=2)
    node_ids_2hop = {n["id"] for n in subnetwork_2hop.nodes}

    assert "app_A1" in node_ids_2hop
    assert "app_A2" in node_ids_2hop
    assert hash_pii("+91 99999 11111", is_numeric_id=True) in node_ids_2hop
    assert "app_B1" not in node_ids_2hop
    assert "app_B2" not in node_ids_2hop
    assert hash_pii("+91 88888 22222", is_numeric_id=True) not in node_ids_2hop

    # 2. Query 3-hop subnetwork from app_A1 (expands to app_A2's other device)
    subnetwork_3hop = graph_engine.find_linked_clusters("app_A1", depth=3)
    node_ids_3hop = {n["id"] for n in subnetwork_3hop.nodes}

    assert hash_pii("device-cluster-a") in node_ids_3hop
    # Must STILL not contain Cluster B members (Strict Boundary Isolation)
    assert "app_B1" not in node_ids_3hop
    assert "app_B2" not in node_ids_3hop


def test_topological_feature_computation(graph_engine):
    """
    Verify the 8 topological feature values on an interconnected synthetic fraud ring.
    Max neighbor risk must exclude the target applicant's own risk score to prevent leakage.
    """
    shared_phone = "+91 77777 00000"
    shared_device = "shared-hardware-device-999"

    # App_Target: Clean score (0.05), shares phone with App_Fraud, shares device with App_Mule
    graph_engine.add_identity_node(
        "app_target",
        attributes={"phone": shared_phone, "device_fingerprint": shared_device, "risk_score": 0.05},
    )

    # App_Fraud: Known fraudster (0.95), shares phone with App_Target
    graph_engine.add_identity_node(
        "app_fraud",
        attributes={"phone": shared_phone, "risk_score": 0.95, "is_fraud": True},
    )

    # App_Mule: Suspicious (0.60), shares device with App_Target
    graph_engine.add_identity_node(
        "app_mule",
        attributes={"device_fingerprint": shared_device, "risk_score": 0.60},
    )

    features = graph_engine.compute_graph_features("app_target")

    assert features.shared_phone_degree == 1.0  # Connected to app_fraud via shared phone
    assert features.shared_device_degree == 1.0  # Connected to app_mule via shared device
    assert features.shared_ssn_degree == 0.0
    assert features.max_neighbor_risk == 0.95  # Highest risk among neighbors (excludes target's 0.05)
    assert features.ego_node_count == 5.0  # target, fraud, mule, phone_hash, device_hash
    assert features.ego_edge_count == 4.0  # target-phone, fraud-phone, target-device, mule-device
    assert features.ego_density > 0.0
    assert features.cycle_count == 0.0


def test_cycle_preservation_in_bipartite_ring(graph_engine):
    """
    Verify that 4-cycles formed by circular attribute swapping (App1-Phone-App2-Device-App1)
    are preserved and counted in the extracted ego-network.
    """
    phone = "+91 66666 55555"
    device = "emulator-canvas-id-888"

    # App1 and App2 share BOTH the phone and the device
    graph_engine.add_identity_node(
        "app_ring_1",
        attributes={"phone": phone, "device_fingerprint": device, "risk_score": 0.7},
    )
    graph_engine.add_identity_node(
        "app_ring_2",
        attributes={"phone": phone, "device_fingerprint": device, "risk_score": 0.8},
    )

    features = graph_engine.compute_graph_features("app_ring_1")
    assert features.cycle_count == 1.0
    assert features.shared_phone_degree == 1.0
    assert features.shared_device_degree == 1.0


def test_fast_1hop_identifier_degree(graph_engine):
    """Verify the O(log n) fast 1-hop degree query against indexed applicant_edges."""
    phone = "+91 55555 44444"
    phone_hash = hash_pii(phone, is_numeric_id=True)

    # 3 applicants sharing the same phone
    for i in range(1, 4):
        graph_engine.add_identity_node(f"app_batch_{i}", attributes={"phone": phone})

    degree = graph_engine.get_1hop_identifier_degree(phone_hash, EdgeType.HAS_PHONE.value)
    assert degree == 3
