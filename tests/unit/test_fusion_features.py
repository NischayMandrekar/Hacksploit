"""
Unit tests for app.modules.fusion.features (Task 4.2).

Verification criteria (PROJECT_PLAN.md Task 4.2):
  - Exact vector dimension of 10.
  - All values normalised to [0.0, 1.0].
  - Correct missingness flags across clean, tampered, and missing-evidence scenarios.
  - extract_with_m0 / extract_m3_features replace dim 2 with real M0 score.
  - to_numpy() returns a (10,) float32 array.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.forensics import EvidenceStatus
from app.core.fusion import ApplicationEvidence
from app.modules.fusion.features import (
    FEATURE_DIM,
    KYCFeatureVectorizer,
    M3FeatureVector,
    _clamp,
    extract_m3_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_evidence(**kwargs) -> ApplicationEvidence:
    """Return a minimal ApplicationEvidence with all forensics UNAVAILABLE."""
    defaults: dict = dict(
        applicant_id="test-001",
        velocity_1h_applications=1,
        velocity_24h_applications=1,
        device_velocity_1h=1,
        shared_phone_degree=1,
        shared_device_degree=1,
        shared_identity_degree=0,
        shared_email_degree=0,
        has_confirmed_fraud_neighbor=False,
        fraud_neighbor_count=0,
        ego_density=0.0,
        cycle_count=0.0,
        image_dimension_status=EvidenceStatus.UNAVAILABLE,
        exif_status=EvidenceStatus.UNAVAILABLE,
        jpeg_quality_status=EvidenceStatus.UNAVAILABLE,
        ela_status=EvidenceStatus.UNAVAILABLE,
        ela_max_diff=None,
        fft_status=EvidenceStatus.UNAVAILABLE,
        fft_grid_ratio=None,
        face_similarity_status=EvidenceStatus.UNAVAILABLE,
        face_cosine_similarity=None,
    )
    defaults.update(kwargs)
    return ApplicationEvidence(**defaults)


def _clean_evidence() -> ApplicationEvidence:
    """Clean applicant: valid checksum, good name match, fresh face, low velocity."""
    return _minimal_evidence(
        ocr_name_similarity=0.97,
        id_checksum_valid=True,
        ela_status=EvidenceStatus.PRESENT,
        ela_max_diff=2.0,
        face_similarity_status=EvidenceStatus.PRESENT,
        face_cosine_similarity=0.92,
    )


def _tampered_evidence() -> ApplicationEvidence:
    """Tampered document: checksum fails, ELA max, name mismatch."""
    return _minimal_evidence(
        ocr_name_similarity=0.20,
        id_checksum_valid=False,
        ela_status=EvidenceStatus.ANOMALOUS,
        ela_max_diff=48.0,
        face_similarity_status=EvidenceStatus.PRESENT,
        face_cosine_similarity=0.30,
    )


def _fraud_ring_evidence() -> ApplicationEvidence:
    """Fraud ring: confirmed neighbor, phone/device reuse, identity collision."""
    return _minimal_evidence(
        has_confirmed_fraud_neighbor=True,
        fraud_neighbor_count=3,
        shared_phone_degree=4,
        shared_device_degree=3,
        shared_identity_degree=1,
        velocity_1h_applications=5,
        device_velocity_1h=4,
        cycle_count=8.0,
    )


# ---------------------------------------------------------------------------
# Dimension and type invariants
# ---------------------------------------------------------------------------


class TestDimension:
    def test_feature_dim_constant_is_10(self):
        assert FEATURE_DIM == 10

    def test_extract_returns_namedtuple_of_length_10(self):
        v = KYCFeatureVectorizer().extract(_minimal_evidence())
        assert isinstance(v, M3FeatureVector)
        assert len(v) == FEATURE_DIM

    def test_to_numpy_shape_and_dtype(self):
        v = KYCFeatureVectorizer().extract(_minimal_evidence())
        arr = v.to_numpy()
        assert arr.shape == (FEATURE_DIM,)
        assert arr.dtype == np.float32

    def test_extract_m3_features_helper_dim(self):
        v = extract_m3_features(_minimal_evidence())
        assert len(v) == FEATURE_DIM


# ---------------------------------------------------------------------------
# Normalisation invariants — all values must be in [0.0, 1.0]
# ---------------------------------------------------------------------------


class TestNormalisation:
    @pytest.mark.parametrize(
        "evidence_fn",
        [_minimal_evidence, _clean_evidence, _tampered_evidence, _fraud_ring_evidence],
    )
    def test_all_values_in_unit_interval(self, evidence_fn):
        v = KYCFeatureVectorizer().extract(evidence_fn())
        for name, val in v._asdict().items():
            assert 0.0 <= val <= 1.0, f"Feature '{name}' = {val} is outside [0, 1]"

    def test_clamp_nan(self):
        assert _clamp(float("nan")) == 0.5

    def test_clamp_inf(self):
        assert _clamp(float("inf")) == 0.5

    def test_clamp_negative_inf(self):
        assert _clamp(float("-inf")) == 0.5

    def test_clamp_over_one(self):
        assert _clamp(1.5) == 1.0

    def test_clamp_below_zero(self):
        assert _clamp(-0.1) == 0.0


# ---------------------------------------------------------------------------
# Missingness flags — binary, correct across all scenarios
# ---------------------------------------------------------------------------


class TestMissingnessFlags:
    def test_all_missing_when_no_evidence(self):
        v = KYCFeatureVectorizer().extract(_minimal_evidence())
        assert v.is_doc_missing == 1.0
        assert v.is_selfie_missing == 1.0
        assert v.is_graph_isolated == 1.0

    def test_doc_not_missing_when_ocr_present(self):
        ev = _minimal_evidence(ocr_name_similarity=0.90)
        v = KYCFeatureVectorizer().extract(ev)
        assert v.is_doc_missing == 0.0

    def test_doc_not_missing_when_checksum_present(self):
        ev = _minimal_evidence(id_checksum_valid=True)
        v = KYCFeatureVectorizer().extract(ev)
        assert v.is_doc_missing == 0.0

    def test_doc_not_missing_when_ela_present(self):
        ev = _minimal_evidence(ela_status=EvidenceStatus.PRESENT, ela_max_diff=5.0)
        v = KYCFeatureVectorizer().extract(ev)
        assert v.is_doc_missing == 0.0

    def test_selfie_not_missing_when_face_present(self):
        ev = _minimal_evidence(
            face_similarity_status=EvidenceStatus.PRESENT,
            face_cosine_similarity=0.85,
        )
        v = KYCFeatureVectorizer().extract(ev)
        assert v.is_selfie_missing == 0.0

    def test_graph_not_isolated_when_phone_shared(self):
        ev = _minimal_evidence(shared_phone_degree=2)
        v = KYCFeatureVectorizer().extract(ev)
        assert v.is_graph_isolated == 0.0

    def test_graph_not_isolated_when_fraud_neighbor(self):
        ev = _minimal_evidence(has_confirmed_fraud_neighbor=True, fraud_neighbor_count=1)
        v = KYCFeatureVectorizer().extract(ev)
        assert v.is_graph_isolated == 0.0


# ---------------------------------------------------------------------------
# Feature semantics — clean vs. tampered
# ---------------------------------------------------------------------------


class TestCleanApplicant:
    def setup_method(self):
        self.v = KYCFeatureVectorizer().extract(_clean_evidence())

    def test_ocr_contradiction_low(self):
        # similarity = 0.97 → contradiction ≈ 0.03
        assert self.v.ocr_contradiction == pytest.approx(0.03, abs=0.01)

    def test_id_checksum_clean(self):
        assert self.v.id_checksum_valid == 0.0

    def test_ela_score_low(self):
        # max_diff=2.0, saturation=50.0 → 0.04
        assert self.v.tampering_ela_score == pytest.approx(0.04, abs=0.01)

    def test_face_dist_low(self):
        # similarity=0.92 → dist=0.08
        assert self.v.face_cosine_dist == pytest.approx(0.08, abs=0.01)

    def test_flags_all_zero(self):
        assert self.v.is_doc_missing == 0.0
        assert self.v.is_selfie_missing == 0.0
        assert self.v.is_graph_isolated == 1.0  # no extra graph connections


class TestTamperedApplicant:
    def setup_method(self):
        self.v = KYCFeatureVectorizer().extract(_tampered_evidence())

    def test_ocr_contradiction_high(self):
        assert self.v.ocr_contradiction == pytest.approx(0.80, abs=0.01)

    def test_id_checksum_invalid(self):
        assert self.v.id_checksum_valid == 1.0

    def test_ela_score_near_saturation(self):
        # max_diff=48.0, saturation=50.0 → 0.96
        assert self.v.tampering_ela_score == pytest.approx(0.96, abs=0.01)

    def test_face_dist_high(self):
        # similarity=0.30 → dist=0.70
        assert self.v.face_cosine_dist == pytest.approx(0.70, abs=0.01)


class TestFraudRing:
    def setup_method(self):
        self.v = KYCFeatureVectorizer().extract(_fraud_ring_evidence())

    def test_m2_graph_risk_elevated(self):
        # degree_score = 8/40 = 0.20, contagion_score = 3/5 = 0.60, cycle_score = 0.80
        # raw = 0.4*0.20 + 0.5*0.60 + 0.1*0.80 = 0.08 + 0.30 + 0.08 = 0.46
        assert self.v.m2_prob == pytest.approx(0.46, abs=0.01)

    def test_m0_proxy_near_one(self):
        # confirmed fraud neighbor → proxy hits 1.0
        assert self.v.m0_rule_score == 1.0

    def test_m1_velocity_elevated(self):
        # velocity_1h=5, device_v=4 → (5+4)/(2*10) = 0.45
        assert self.v.m1_prob == pytest.approx(0.45, abs=0.01)

    def test_is_graph_isolated_zero(self):
        assert self.v.is_graph_isolated == 0.0


# ---------------------------------------------------------------------------
# Missing-evidence neutral mid-point (0.5)
# ---------------------------------------------------------------------------


class TestMissingEvidenceNeutral:
    def test_ocr_contradiction_is_05_when_doc_missing(self):
        v = KYCFeatureVectorizer().extract(_minimal_evidence())
        assert v.ocr_contradiction == 0.5

    def test_id_checksum_is_05_when_doc_missing(self):
        v = KYCFeatureVectorizer().extract(_minimal_evidence())
        assert v.id_checksum_valid == 0.5

    def test_ela_score_is_05_when_unavailable(self):
        v = KYCFeatureVectorizer().extract(_minimal_evidence())
        assert v.tampering_ela_score == 0.5

    def test_face_dist_is_05_when_selfie_missing(self):
        v = KYCFeatureVectorizer().extract(_minimal_evidence())
        assert v.face_cosine_dist == 0.5


# ---------------------------------------------------------------------------
# extract_with_m0 — real M0 score substitution
# ---------------------------------------------------------------------------


class TestExtractWithM0:
    def test_dim_2_is_replaced(self):
        ev = _clean_evidence()
        vz = KYCFeatureVectorizer()
        base = vz.extract(ev)
        with_m0 = vz.extract_with_m0(ev, m0_risk_score=0.77)
        assert with_m0.m0_rule_score == pytest.approx(0.77, abs=1e-9)
        # Other dimensions unchanged
        assert with_m0.m1_prob == base.m1_prob
        assert with_m0.face_cosine_dist == base.face_cosine_dist

    def test_m0_score_is_clamped(self):
        ev = _minimal_evidence()
        vz = KYCFeatureVectorizer()
        v = vz.extract_with_m0(ev, m0_risk_score=1.5)
        assert v.m0_rule_score == 1.0

    def test_extract_m3_features_uses_m0_when_provided(self):
        ev = _clean_evidence()
        v = extract_m3_features(ev, m0_risk_score=0.33)
        assert v.m0_rule_score == pytest.approx(0.33, abs=1e-9)

    def test_extract_m3_features_uses_proxy_when_m0_absent(self):
        # Confirm proxy behavior: clean applicant → low score from proxy
        ev = _clean_evidence()
        v = extract_m3_features(ev)
        # Clean evidence has no hard signals → proxy should be 0 or very low
        assert v.m0_rule_score == 0.0


# ---------------------------------------------------------------------------
# Edge cases and boundary conditions
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_velocity_saturation(self):
        ev = _minimal_evidence(velocity_1h_applications=100, device_velocity_1h=100)
        v = KYCFeatureVectorizer().extract(ev)
        assert v.m1_prob == 1.0

    def test_ela_above_saturation_clamped(self):
        ev = _minimal_evidence(
            ela_status=EvidenceStatus.ANOMALOUS,
            ela_max_diff=999.0,
        )
        v = KYCFeatureVectorizer().extract(ev)
        assert v.tampering_ela_score == 1.0

    def test_face_similarity_above_one_clamped(self):
        # Pathological input: similarity > 1.0 (should clamp dist to 0.0)
        ev = _minimal_evidence(
            face_similarity_status=EvidenceStatus.PRESENT,
            face_cosine_similarity=1.05,
        )
        v = KYCFeatureVectorizer().extract(ev)
        assert v.face_cosine_dist == 0.0

    def test_to_dict_keys_match_namedtuple_fields(self):
        v = KYCFeatureVectorizer().extract(_minimal_evidence())
        d = v.to_dict()
        assert set(d.keys()) == set(M3FeatureVector._fields)

    def test_vectorizer_is_reusable(self):
        vz = KYCFeatureVectorizer()
        v1 = vz.extract(_clean_evidence())
        v2 = vz.extract(_tampered_evidence())
        # Verify different inputs produce different outputs
        assert v1 != v2
