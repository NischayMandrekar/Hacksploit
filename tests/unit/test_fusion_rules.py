import time
from app.core.forensics import EvidenceStatus
from app.core.fusion import ApplicationEvidence, RiskTier
from app.modules.fusion.rules import DeterministicM0Model, evaluate_m0_rules_dict


def test_m0_clean_applicant_approved():
    """A clean applicant with genuine documents and no shared attributes is approved with 0.04 base rate."""
    model = DeterministicM0Model()
    evidence = ApplicationEvidence(
        applicant_id="app_clean_001",
        claimed_name="Aarav Sharma",
        claimed_dob="1992-05-15",
        id_type="aadhaar",
        shared_phone_degree=1,
        shared_device_degree=1,
        shared_identity_degree=0,
        ocr_name_similarity=0.98,
        dob_mismatch_years=0.0,
        id_checksum_valid=True,
        secure_qr_status="VERIFIED",
        secure_qr_valid=True,
        image_dimension_status=EvidenceStatus.PRESENT,
        exif_status=EvidenceStatus.PRESENT,
        jpeg_quality_status=EvidenceStatus.PRESENT,
        face_similarity_status=EvidenceStatus.PRESENT,
        face_cosine_similarity=0.88,
        velocity_1h_applications=1,
    )

    assessment = model.predict_risk(evidence)

    assert assessment.applicant_id == "app_clean_001"
    assert assessment.model_name == "M0_Rules"
    assert assessment.risk_tier == RiskTier.APPROVE
    assert assessment.risk_score == 0.04
    assert assessment.reason_codes == ["CLEAN"]
    assert "passed" in assessment.summary.lower()
    assert assessment.execution_time_ms < 5.0  # Must be fast


def test_m0_confirmed_fraud_link_immediate_reject():
    """Direct links to confirmed fraud accounts immediately reject the applicant."""
    model = DeterministicM0Model()
    evidence = ApplicationEvidence(
        applicant_id="app_fraud_link",
        has_confirmed_fraud_neighbor=True,
        fraud_neighbor_count=2,
    )

    assessment = model.predict_risk(evidence)

    assert assessment.risk_tier == RiskTier.REJECT
    assert assessment.risk_score >= 0.75
    assert "CONFIRMED_FRAUD_LINK" in assessment.reason_codes


def test_m0_identity_reuse_collision_reject():
    """National ID / Aadhaar already registered to another applicant is immediately rejected."""
    model = DeterministicM0Model()
    evidence = ApplicationEvidence(
        applicant_id="app_id_collision",
        shared_identity_degree=1,
    )

    assessment = model.predict_risk(evidence)

    assert assessment.risk_tier == RiskTier.REJECT
    assert assessment.risk_score >= 0.75
    assert "IDENTITY_REUSE_COLLISION" in assessment.reason_codes


def test_m0_phone_reuse_tiers():
    """Phone reuse is tiered: >= 3 triggers REJECT, degree 2 triggers REVIEW."""
    model = DeterministicM0Model()

    # Degree 3: Rejection
    high_phone = ApplicationEvidence(
        applicant_id="app_phone_3",
        shared_phone_degree=3,
    )
    res_high = model.predict_risk(high_phone)
    assert res_high.risk_tier == RiskTier.REJECT
    assert "R1_HIGH_PHONE_REUSE" in res_high.reason_codes

    # Degree 2: Review (household sharing)
    mod_phone = ApplicationEvidence(
        applicant_id="app_phone_2",
        shared_phone_degree=2,
    )
    res_mod = model.predict_risk(mod_phone)
    assert res_mod.risk_tier == RiskTier.REVIEW
    assert "PHONE_REUSE_MODERATE" in res_mod.reason_codes


def test_m0_device_reuse_tiers():
    """Device reuse is tiered: >= 3 triggers REJECT, degree 2 triggers REVIEW."""
    model = DeterministicM0Model()

    # Degree 4: Rejection
    high_dev = ApplicationEvidence(
        applicant_id="app_dev_4",
        shared_device_degree=4,
    )
    res_high = model.predict_risk(high_dev)
    assert res_high.risk_tier == RiskTier.REJECT
    assert "R2_HIGH_DEVICE_REUSE" in res_high.reason_codes

    # Degree 2: Review
    mod_dev = ApplicationEvidence(
        applicant_id="app_dev_2",
        shared_device_degree=2,
    )
    res_mod = model.predict_risk(mod_dev)
    assert res_mod.risk_tier == RiskTier.REVIEW
    assert "DEVICE_REUSE_MODERATE" in res_mod.reason_codes


def test_m0_secure_qr_tampered_reject():
    """Tampered Aadhaar Secure QR triggers immediate rejection."""
    model = DeterministicM0Model()
    evidence = ApplicationEvidence(
        applicant_id="app_qr_tampered",
        secure_qr_status="TAMPERED",
        secure_qr_valid=False,
    )

    assessment = model.predict_risk(evidence)

    assert assessment.risk_tier == RiskTier.REJECT
    assert assessment.risk_score >= 0.75
    assert "R7_SECURE_QR_TAMPERED" in assessment.reason_codes


def test_m0_invalid_id_checksum_reject():
    """Failed Verhoeff/Luhn/PAN checksum triggers immediate rejection."""
    model = DeterministicM0Model()
    evidence = ApplicationEvidence(
        applicant_id="app_bad_checksum",
        id_checksum_valid=False,
    )

    assessment = model.predict_risk(evidence)

    assert assessment.risk_tier == RiskTier.REJECT
    assert assessment.risk_score >= 0.75
    assert "R5_INVALID_ID_CHECKSUM" in assessment.reason_codes


def test_m0_demographic_contradictions_review():
    """OCR name and DOB mismatches trigger REVIEW tier."""
    model = DeterministicM0Model()
    evidence = ApplicationEvidence(
        applicant_id="app_contradictions",
        ocr_name_similarity=0.60,
        dob_mismatch_years=5.0,
    )

    assessment = model.predict_risk(evidence)

    assert assessment.risk_tier == RiskTier.REVIEW
    assert "R3_NAME_CONTRADICTION" in assessment.reason_codes
    assert "R4_DOB_CONTRADICTION" in assessment.reason_codes
    assert 0.30 <= assessment.risk_score <= 0.74


def test_m0_biometric_face_mismatch_review():
    """Document face crop vs selfie cosine similarity < 0.40 triggers REVIEW."""
    model = DeterministicM0Model()
    evidence = ApplicationEvidence(
        applicant_id="app_face_mismatch",
        face_similarity_status=EvidenceStatus.ANOMALOUS,
        face_cosine_similarity=0.22,
    )

    assessment = model.predict_risk(evidence)

    assert assessment.risk_tier == RiskTier.REVIEW
    assert "DOCUMENT_FACE_MISMATCH" in assessment.reason_codes


def test_m0_media_forensic_anomalies_review():
    """AI generator dimensions, EXIF tags, and ELA tampering trigger REVIEW."""
    model = DeterministicM0Model()
    evidence = ApplicationEvidence(
        applicant_id="app_tampered_img",
        image_dimension_status=EvidenceStatus.ANOMALOUS,
        exif_status=EvidenceStatus.ANOMALOUS,
        jpeg_quality_status=EvidenceStatus.ANOMALOUS,
        ela_status=EvidenceStatus.ANOMALOUS,
        fft_status=EvidenceStatus.ANOMALOUS,
    )

    assessment = model.predict_risk(evidence)

    assert assessment.risk_tier == RiskTier.REVIEW
    assert "IMAGE_AI_SYNTHETIC" in assessment.reason_codes
    assert "JPEG_QUALITY_SYNTHETIC" in assessment.reason_codes
    assert "IMAGE_ELA_TAMPERED" in assessment.reason_codes
    assert "IMAGE_FFT_GRID_ANOMALY" in assessment.reason_codes


def test_m0_velocity_burst():
    """Velocity burst >= 4 triggers REJECT; 2-3 triggers REVIEW."""
    model = DeterministicM0Model()

    res_burst = model.predict_risk(ApplicationEvidence(
        applicant_id="app_burst",
        velocity_1h_applications=5,
    ))
    assert res_burst.risk_tier == RiskTier.REJECT
    assert "R6_VELOCITY_BURST" in res_burst.reason_codes

    res_mod = model.predict_risk(ApplicationEvidence(
        applicant_id="app_vel_mod",
        velocity_1h_applications=2,
    ))
    assert res_mod.risk_tier == RiskTier.REVIEW
    assert "VELOCITY_MODERATE" in res_mod.reason_codes


def test_m0_missing_signals_handling():
    """
    ARCHITECTURE.md §3 Invariant Rule 4:
    Missing signals are explicitly tracked as UNAVAILABLE, never defaulted to clean/passed,
    and do NOT cause false positive rejections.
    """
    model = DeterministicM0Model()
    # Applicant with no document or facial signals yet (e.g. preliminary screening)
    evidence = ApplicationEvidence(
        applicant_id="app_partial_evidence",
        shared_phone_degree=1,
        shared_device_degree=1,
        secure_qr_status="UNAVAILABLE",
        id_checksum_valid=None,
        ocr_name_similarity=None,
        dob_mismatch_years=None,
        face_similarity_status=EvidenceStatus.UNAVAILABLE,
        face_cosine_similarity=None,
    )

    assessment = model.predict_risk(evidence)

    assert assessment.risk_tier == RiskTier.APPROVE
    assert assessment.risk_score == 0.04
    assert "secure_qr" in assessment.missing_signals
    assert "id_checksum" in assessment.missing_signals
    assert "ocr_name_similarity" in assessment.missing_signals
    assert "face_similarity" in assessment.missing_signals

    # Attributions include informational missing signals
    missing_attributions = [a for a in assessment.attributions if a.severity == "INFO"]
    assert len(missing_attributions) >= 4
    for a in missing_attributions:
        assert a.evidence_status == EvidenceStatus.UNAVAILABLE
        assert a.score_impact == 0.0


def test_m0_submillisecond_performance():
    """M0 must evaluate synchronously in < 1ms on average."""
    model = DeterministicM0Model()
    evidence = ApplicationEvidence(
        applicant_id="app_bench",
        shared_phone_degree=2,
        ocr_name_similarity=0.80,
        dob_mismatch_years=1.0,
        id_checksum_valid=True,
    )

    # Warmup
    for _ in range(10):
        model.predict_risk(evidence)

    runs = 100
    t0 = time.perf_counter()
    for _ in range(runs):
        model.predict_risk(evidence)
    total_time_ms = (time.perf_counter() - t0) * 1000.0
    avg_ms = total_time_ms / runs

    assert avg_ms < 1.0, f"Average execution time {avg_ms:.3f}ms exceeded 1.0ms limit"


def test_evaluate_m0_rules_dict_helper():
    """Verifies the dict helper conforms to MODELS_TRAINING_SPEC.md §3.3."""
    clean_dict = {
        "applicant_id": "app_spec_clean",
        "shared_phone_count": 1,
        "shared_device_count": 1,
        "ocr_name_similarity": 0.95,
        "dob_age_mismatch_years": 0.0,
        "id_checksum_valid": True,
        "velocity_1h_applications": 1,
    }
    score, reasons = evaluate_m0_rules_dict(clean_dict)
    assert score == 0.04
    assert reasons == ["CLEAN"]

    fraud_dict = {
        "applicant_id": "app_spec_ring",
        "shared_phone_count": 4,
        "shared_device_count": 3,
        "ocr_name_similarity": 0.50,
        "dob_age_mismatch_years": 10.0,
        "id_checksum_valid": False,
        "velocity_1h_applications": 6,
    }
    f_score, f_reasons = evaluate_m0_rules_dict(fraud_dict)
    assert f_score >= 0.75
    assert "R1_HIGH_PHONE_REUSE" in f_reasons
    assert "R2_HIGH_DEVICE_REUSE" in f_reasons
    assert "R3_NAME_CONTRADICTION" in f_reasons
    assert "R4_DOB_CONTRADICTION" in f_reasons
    assert "R5_INVALID_ID_CHECKSUM" in f_reasons
    assert "R6_VELOCITY_BURST" in f_reasons
