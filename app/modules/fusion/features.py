"""
KYC Feature Vectorizer (Task 4.2).

Extracts the authoritative 10-dimensional M3 feature vector (``X_M3``) from a
live :class:`~app.core.fusion.ApplicationEvidence` instance.

Vector layout (index → name → source):
  0  m1_prob              Behavioral velocity / age-profile risk proxy
  1  m2_prob              Relational ego-network graph risk proxy
  2  m0_rule_score        M0 deterministic combined rule score (already in [0,1])
  3  ocr_contradiction    1 − ocr_name_similarity (0 = perfect match, 1 = total mismatch)
  4  id_checksum_valid    0.0 if Verhoeff/PAN valid, 1.0 if invalid, 0.5 if unknown
  5  tampering_ela_score  Normalized ELA max-diff (0 = clean, 1 = heavy tampering)
  6  face_cosine_dist     1 − face_cosine_similarity (0 = identical, 1 = strangers)
  7  is_doc_missing       1.0 when no document was submitted, else 0.0
  8  is_selfie_missing    1.0 when no selfie was submitted, else 0.0
  9  is_graph_isolated    1.0 when applicant has no graph neighbours, else 0.0

Design constraints (ARCHITECTURE.md §2, AGENTS.md §5):
  - Zero reliance on e-commerce transaction features (card1, TransactionAmt, …).
  - All dimensions are floats in [0.0, 1.0].
  - Missing evidence is represented as explicit missingness flags; feature
    slots that depend on missing evidence fall back to a neutral mid-point
    (0.5) so the MLP can distinguish "unknown" from "clean" (0.0) and
    "bad" (1.0) without imputation hacks.
  - This module is pure Python / dataclass only; no ML library imported here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from app.core.forensics import EvidenceStatus
from app.core.fusion import ApplicationEvidence

# ---------------------------------------------------------------------------
# Public vector dimension constant — checked in unit tests
# ---------------------------------------------------------------------------
FEATURE_DIM: int = 10

# ELA max-diff saturation point: diffs above this are treated as 1.0 tampering.
_ELA_SATURATION: float = 50.0

# Velocity saturation: >= this many applications in 1h is treated as max risk.
_VELOCITY_1H_SATURATION: int = 10

# Graph degree saturation: >= this many shared connections saturates graph risk.
_DEGREE_SATURATION: int = 10


class M3FeatureVector(NamedTuple):
    """
    Typed, named 10-dimensional feature vector for M3 fusion.

    All fields are floats in [0.0, 1.0].  Missing evidence is signalled by
    the three explicit binary flags at indices 7-9; dependent continuous
    features fall back to 0.5 to avoid imputation bias.
    """

    m1_prob: float
    m2_prob: float
    m0_rule_score: float
    ocr_contradiction: float
    id_checksum_valid: float
    tampering_ela_score: float
    face_cosine_dist: float
    is_doc_missing: float
    is_selfie_missing: float
    is_graph_isolated: float

    def to_numpy(self) -> np.ndarray:
        """Return a (10,) float32 array compatible with PyTorch and Scikit-learn."""
        return np.array(self, dtype=np.float32)

    def to_dict(self) -> dict[str, float]:
        return self._asdict()  # type: ignore[return-value]


@dataclass(frozen=True)
class KYCFeatureVectorizer:
    """
    Stateless vectorizer that maps :class:`~app.core.fusion.ApplicationEvidence`
    to a typed :class:`M3FeatureVector`.

    All scoring logic is deterministic and side-effect-free.  One instance may
    be shared across threads.
    """

    # Velocity proxy calibration
    velocity_1h_saturation: int = _VELOCITY_1H_SATURATION

    # Graph proxy calibration
    degree_saturation: int = _DEGREE_SATURATION

    # ELA saturation
    ela_saturation: float = _ELA_SATURATION

    def extract(self, evidence: ApplicationEvidence) -> M3FeatureVector:
        """
        Extract a normalised 10-dimensional feature vector from *evidence*.

        Returns a :class:`M3FeatureVector` with all values in [0.0, 1.0].
        """
        doc_missing = self._doc_missing(evidence)
        selfie_missing = self._selfie_missing(evidence)
        graph_isolated = self._graph_isolated(evidence)

        return M3FeatureVector(
            m1_prob=self._m1_behavioral_velocity(evidence),
            m2_prob=self._m2_graph_risk(evidence, graph_isolated),
            m0_rule_score=self._m0_rule_score(evidence),
            ocr_contradiction=self._ocr_contradiction(evidence, doc_missing),
            id_checksum_valid=self._id_checksum(evidence, doc_missing),
            tampering_ela_score=self._ela_score(evidence),
            face_cosine_dist=self._face_cosine_dist(evidence, selfie_missing),
            is_doc_missing=float(doc_missing),
            is_selfie_missing=float(selfie_missing),
            is_graph_isolated=float(graph_isolated),
        )

    def extract_with_m0(
        self,
        evidence: ApplicationEvidence,
        m0_risk_score: float,
    ) -> M3FeatureVector:
        """
        Like :meth:`extract` but substitutes the real M0 *risk_score* for
        dimension 2 instead of the internal proxy.

        Callers in the full pipeline MUST use this variant so that M3 receives
        the calibrated M0 output rather than the proxy reconstruction.
        """
        base = self.extract(evidence)
        return M3FeatureVector(
            m1_prob=base.m1_prob,
            m2_prob=base.m2_prob,
            m0_rule_score=_clamp(m0_risk_score),
            ocr_contradiction=base.ocr_contradiction,
            id_checksum_valid=base.id_checksum_valid,
            tampering_ela_score=base.tampering_ela_score,
            face_cosine_dist=base.face_cosine_dist,
            is_doc_missing=base.is_doc_missing,
            is_selfie_missing=base.is_selfie_missing,
            is_graph_isolated=base.is_graph_isolated,
        )

    # ------------------------------------------------------------------
    # Internal helpers — each maps to exactly one feature dimension
    # ------------------------------------------------------------------

    def _m1_behavioral_velocity(self, ev: ApplicationEvidence) -> float:
        """
        Velocity / age-profile risk proxy for M1.

        Combines 1-hour application velocity and device velocity into a single
        [0, 1] score.  This is a domain-grounded proxy; the real M1 model will
        be retrained on labeled KYC data and replace this proxy (ARCHITECTURE.md §4).
        """
        v1h = min(ev.velocity_1h_applications, self.velocity_1h_saturation)
        dv = min(ev.device_velocity_1h, self.velocity_1h_saturation)
        raw = (v1h + dv) / (2.0 * self.velocity_1h_saturation)
        return _clamp(raw)

    def _m2_graph_risk(self, ev: ApplicationEvidence, graph_isolated: bool) -> float:
        """
        Relational ego-network risk proxy for M2.

        Combines shared-identifier degree, confirmed fraud contagion, and
        ego-network cycle density.  An isolated applicant returns 0.0; a
        saturated fraud hub returns 1.0.
        """
        if graph_isolated:
            return 0.0

        total_degree = (
            ev.shared_phone_degree
            + ev.shared_device_degree
            + ev.shared_identity_degree
            + ev.shared_email_degree
        )
        degree_score = min(total_degree, self.degree_saturation * 4) / (
            self.degree_saturation * 4
        )

        contagion_score = (
            min(ev.fraud_neighbor_count, 5) / 5.0
            if ev.has_confirmed_fraud_neighbor
            else 0.0
        )

        cycle_score = _clamp(ev.cycle_count / 10.0) if ev.cycle_count > 0 else 0.0

        raw = 0.4 * degree_score + 0.5 * contagion_score + 0.1 * cycle_score
        return _clamp(raw)

    def _m0_rule_score(self, ev: ApplicationEvidence) -> float:
        """
        Deterministic M0 aggregate proxy for dimension 2.

        When the real M0 risk_score is available, callers should use
        :meth:`extract_with_m0` instead.
        """
        score = 0.0

        if ev.has_confirmed_fraud_neighbor:
            score = max(score, 1.0)
        if ev.shared_identity_degree >= 1:
            score = max(score, 0.85)
        if ev.id_checksum_valid is False:
            score = max(score, 0.80)
        if ev.secure_qr_status == "TAMPERED":
            score = max(score, 0.80)

        if ev.shared_phone_degree >= 3:
            score = max(score, 0.75)
        elif ev.shared_phone_degree == 2:
            score = max(score, 0.40)
        if ev.shared_device_degree >= 3:
            score = max(score, 0.70)
        elif ev.shared_device_degree == 2:
            score = max(score, 0.35)

        if ev.velocity_1h_applications >= 4:
            score = max(score, 0.70)
        elif ev.velocity_1h_applications >= 2:
            score = max(score, 0.35)

        if ev.ela_status == EvidenceStatus.ANOMALOUS:
            score = max(score, 0.50)
        if ev.image_dimension_status == EvidenceStatus.ANOMALOUS:
            score = max(score, 0.35)
        if ev.exif_status == EvidenceStatus.ANOMALOUS:
            score = max(score, 0.30)

        return _clamp(score)

    def _ocr_contradiction(self, ev: ApplicationEvidence, doc_missing: bool) -> float:
        """
        OCR name-contradiction score.

        Returns 0.5 when no document is available (unknown, not clean).
        Returns 1 - similarity when available (0 = perfect match, 1 = total mismatch).
        """
        if doc_missing or ev.ocr_name_similarity is None:
            return 0.5
        return _clamp(1.0 - ev.ocr_name_similarity)

    def _id_checksum(self, ev: ApplicationEvidence, doc_missing: bool) -> float:
        """
        Identity number checksum validity.

        * 0.0 — checksum passed (Verhoeff for Aadhaar, format check for PAN)
        * 1.0 — checksum failed (explicit forgery signal)
        * 0.5 — unknown (document missing or check not performed)
        """
        if doc_missing or ev.id_checksum_valid is None:
            return 0.5
        return 0.0 if ev.id_checksum_valid else 1.0

    def _ela_score(self, ev: ApplicationEvidence) -> float:
        """
        ELA tampering score normalised to [0, 1] against saturation constant.

        Returns 0.5 when the ELA signal is unavailable (unknown, not clean).
        """
        if ev.ela_status == EvidenceStatus.UNAVAILABLE:
            return 0.5
        if ev.ela_status == EvidenceStatus.PRESENT:
            ela_val = ev.ela_max_diff if ev.ela_max_diff is not None else 0.0
        else:
            # ANOMALOUS — use the recorded diff or saturate
            ela_val = (
                ev.ela_max_diff
                if ev.ela_max_diff is not None
                else self.ela_saturation
            )
        return _clamp(ela_val / self.ela_saturation)

    def _face_cosine_dist(
        self, ev: ApplicationEvidence, selfie_missing: bool
    ) -> float:
        """
        Face distance = 1 - cosine similarity.

        Returns 0.5 when selfie or face embedding is unavailable.
        """
        if selfie_missing or ev.face_similarity_status == EvidenceStatus.UNAVAILABLE:
            return 0.5
        if ev.face_cosine_similarity is None:
            return 0.5
        return _clamp(1.0 - ev.face_cosine_similarity)

    # ------------------------------------------------------------------
    # Missingness detectors
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_missing(ev: ApplicationEvidence) -> bool:
        """True when no document forensic evidence is present."""
        has_ocr = (
            ev.ocr_name_similarity is not None
            or ev.document_extracted_name is not None
        )
        has_checksum = ev.id_checksum_valid is not None
        has_ela = ev.ela_status != EvidenceStatus.UNAVAILABLE
        return not (has_ocr or has_checksum or has_ela)

    @staticmethod
    def _selfie_missing(ev: ApplicationEvidence) -> bool:
        """True when no selfie / face embedding evidence is present."""
        return ev.face_similarity_status == EvidenceStatus.UNAVAILABLE

    @staticmethod
    def _graph_isolated(ev: ApplicationEvidence) -> bool:
        """True when the applicant has no recorded graph edges."""
        return (
            ev.shared_phone_degree <= 1
            and ev.shared_device_degree <= 1
            and ev.shared_identity_degree == 0
            and ev.shared_email_degree == 0
            and not ev.has_confirmed_fraud_neighbor
        )


# ---------------------------------------------------------------------------
# Module-level convenience helper
# ---------------------------------------------------------------------------


def extract_m3_features(
    evidence: ApplicationEvidence,
    m0_risk_score: float | None = None,
) -> M3FeatureVector:
    """
    Convenience function: vectorize *evidence* and (optionally) substitute the
    real *m0_risk_score* for dimension 2.

    Args:
        evidence: Populated :class:`~app.core.fusion.ApplicationEvidence`.
        m0_risk_score: If provided, replaces the proxy M0 dimension with the
            actual calibrated M0 output (preferred in pipeline use).

    Returns:
        :class:`M3FeatureVector` with all values in [0.0, 1.0].
    """
    vectorizer = KYCFeatureVectorizer()
    if m0_risk_score is not None:
        return vectorizer.extract_with_m0(evidence, m0_risk_score)
    return vectorizer.extract(evidence)


# ---------------------------------------------------------------------------
# Private utility
# ---------------------------------------------------------------------------


def _clamp(value: float) -> float:
    """Clamp *value* to [0.0, 1.0] and guard against NaN/Inf."""
    if math.isnan(value) or math.isinf(value):
        return 0.5
    return max(0.0, min(1.0, value))
