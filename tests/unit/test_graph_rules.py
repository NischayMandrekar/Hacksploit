import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.infrastructure.database import Base
from app.modules.graph.engine import RelationalIdentityGraphEngine
from app.modules.graph.rules import M0RuleEngine

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
def rule_engine(db_session):
    return M0RuleEngine(
        db=db_session,
        phone_review_threshold=2,
        phone_reject_threshold=3,
        device_review_threshold=2,
        device_reject_threshold=3,
    )


def test_clean_applicant_approval(rule_engine):
    """Verify that a brand new applicant with fresh identifiers passes M0 cleanly in < 20ms."""
    result = rule_engine.evaluate_applicant(
        applicant_id="app_fresh_001",
        identifiers={
            "phone": "+91 91234 56789",
            "email": "fresh@domain.com",
            "aadhaar": "9999 8888 7777",
            "device_fingerprint": "clean-device-canvas-001",
        },
        application_context={"ocr_name_similarity": 0.95, "id_checksum_valid": True},
    )

    assert result.risk_tier == "APPROVE"
    assert result.risk_score <= 0.10
    assert result.reason_codes == ["CLEAN"]
    assert len(result.violations) == 0
    assert result.execution_time_ms < 20.0


def test_identity_reuse_collision(graph_engine, rule_engine):
    """
    Verify that an attempt to register with an Aadhaar number already linked
    to an existing customer triggers an instant IDENTITY_REUSE_COLLISION REJECT.
    """
    shared_aadhaar = "1111-2222-3333"

    # Existing customer
    graph_engine.add_identity_node(
        "app_legitimate_customer",
        attributes={"aadhaar": shared_aadhaar, "phone": "+91 90000 00001"},
    )

    # New applicant reusing the same Aadhaar
    result = rule_engine.evaluate_applicant(
        applicant_id="app_impostor_002",
        identifiers={
            "aadhaar": shared_aadhaar,
            "phone": "+91 90000 00002",
            "device_fingerprint": "impostor-device-xyz",
        },
    )

    assert result.risk_tier == "REJECT"
    assert "IDENTITY_REUSE_COLLISION" in result.reason_codes
    assert result.risk_score >= 0.70
    assert result.shared_counts["identity"] == 1


def test_phone_reuse_moderate_vs_ring(graph_engine, rule_engine):
    """
    Verify differentiation between benign family sharing (degree 2 -> REVIEW)
    and organized burner SIM ring (degree >= 3 -> REJECT).
    """
    shared_phone = "+91 98888 77777"

    # 1 existing applicant sharing phone (family sharing scenario)
    graph_engine.add_identity_node(
        "app_spouse_1",
        attributes={"phone": shared_phone},
    )

    # Evaluate spouse 2 (degree = 2)
    result_moderate = rule_engine.evaluate_applicant(
        applicant_id="app_spouse_2",
        identifiers={"phone": shared_phone, "device_fingerprint": "device_spouse_2"},
    )
    assert result_moderate.risk_tier == "REVIEW"
    assert "PHONE_REUSE_MODERATE" in result_moderate.reason_codes
    assert result_moderate.shared_counts["phone"] == 1

    # Add 2 more applicants on the same phone (syndicate ring scenario)
    graph_engine.add_identity_node("app_spouse_2", attributes={"phone": shared_phone})
    graph_engine.add_identity_node("app_syndicate_3", attributes={"phone": shared_phone})

    # Evaluate 4th applicant (shared_count = 3 >= reject threshold)
    result_ring = rule_engine.evaluate_applicant(
        applicant_id="app_syndicate_4",
        identifiers={"phone": shared_phone},
    )
    assert result_ring.risk_tier == "REJECT"
    assert "R1_HIGH_PHONE_REUSE" in result_ring.reason_codes
    assert result_ring.shared_counts["phone"] == 3


def test_device_reuse_ring(graph_engine, rule_engine):
    """Verify that multiple applications from the same emulator device trigger R2_HIGH_DEVICE_REUSE."""
    farm_device = "emulator-device-farm-rack-42"

    for i in range(1, 4):
        graph_engine.add_identity_node(
            f"app_bot_{i}",
            attributes={"device_fingerprint": farm_device, "phone": f"+91 90000 000{i}"},
        )

    result = rule_engine.evaluate_applicant(
        applicant_id="app_bot_4",
        identifiers={"device_fingerprint": farm_device},
    )

    assert result.risk_tier == "REJECT"
    assert "R2_HIGH_DEVICE_REUSE" in result.reason_codes
    assert result.shared_counts["device"] == 3


def test_confirmed_fraud_neighbor_link(graph_engine, rule_engine):
    """
    Verify that an applicant sharing an identifier with a known confirmed fraudster
    is immediately marked with CONFIRMED_FRAUD_LINK and maximum risk score.
    """
    burner_phone = "+91 97777 66666"

    # Known fraudster in bank history
    graph_engine.add_identity_node(
        "app_confirmed_fraudster",
        attributes={"phone": burner_phone, "is_fraud": True, "status": "fraud", "risk_score": 0.99},
    )

    result = rule_engine.evaluate_applicant(
        applicant_id="app_mule_account",
        identifiers={"phone": burner_phone},
    )

    assert result.risk_tier == "REJECT"
    assert "CONFIRMED_FRAUD_LINK" in result.reason_codes
    assert result.risk_score >= 0.95


def test_application_context_violations(rule_engine):
    """Verify that OCR contradictions, invalid checksums, and velocity bursts trigger correct reason codes."""
    result = rule_engine.evaluate_applicant(
        applicant_id="app_contradiction",
        identifiers={"phone": "+91 95555 11111"},
        application_context={
            "ocr_name_similarity": 0.55,  # Contradiction (< 0.75)
            "dob_age_mismatch_years": 5.0,  # Contradiction (> 2)
            "id_checksum_valid": False,  # Checksum failed
            "velocity_1h_applications": 6,  # Burst (>= 4)
        },
    )

    assert result.risk_tier == "REJECT"
    assert "R3_NAME_CONTRADICTION" in result.reason_codes
    assert "R4_DOB_CONTRADICTION" in result.reason_codes
    assert "R5_INVALID_ID_CHECKSUM" in result.reason_codes
    assert "R6_VELOCITY_BURST" in result.reason_codes
