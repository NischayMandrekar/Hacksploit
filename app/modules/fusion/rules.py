import time
from typing import Any

from app.core.forensics import EvidenceStatus
from app.core.fusion import (
    ApplicationEvidence,
    EvidenceAttribution,
    FraudModel,
    RiskAssessment,
    RiskTier,
)


class DeterministicM0Model(FraudModel):
    """
    Baseline M0: Deterministic Multi-Source Screening Engine.
    Evaluates identity graph collisions, document contradictions, biometric
    mismatches, image tampering, and submission velocity in sub-millisecond time.
    Provides explainable reason codes, structured attributions, and calibrated risk tiers.
    """

    def __init__(
        self,
        phone_reject_threshold: int = 3,
        phone_review_threshold: int = 2,
        device_reject_threshold: int = 3,
        device_review_threshold: int = 2,
        name_similarity_threshold: float = 0.75,
        dob_mismatch_years_threshold: float = 2.0,
        velocity_reject_threshold: int = 4,
        velocity_review_threshold: int = 2,
        face_similarity_threshold: float = 0.40,
    ) -> None:
        self.phone_reject_threshold = phone_reject_threshold
        self.phone_review_threshold = phone_review_threshold
        self.device_reject_threshold = device_reject_threshold
        self.device_review_threshold = device_review_threshold
        self.name_similarity_threshold = name_similarity_threshold
        self.dob_mismatch_years_threshold = dob_mismatch_years_threshold
        self.velocity_reject_threshold = velocity_reject_threshold
        self.velocity_review_threshold = velocity_review_threshold
        self.face_similarity_threshold = face_similarity_threshold

    def predict_risk(self, features: ApplicationEvidence) -> RiskAssessment:
        """
        Evaluates multi-source evidence against the deterministic ruleset.
        ARCHITECTURE.md §2 guarantee: returns calibrated risk_score, risk_tier,
        and evidence_attributions.
        """
        start_time = time.perf_counter()
        violations: list[EvidenceAttribution] = []
        missing_signals: list[str] = []

        # ---------------------------------------------------------------------
        # 1. Relational Graph & Contagion Rules
        # ---------------------------------------------------------------------
        if features.has_confirmed_fraud_neighbor or features.fraud_neighbor_count > 0:
            count = max(1, features.fraud_neighbor_count)
            violations.append(
                EvidenceAttribution(
                    rule_id="CONFIRMED_FRAUD_LINK",
                    signal_name="graph_contagion",
                    severity="REJECT",
                    score_impact=1.0,
                    description=(
                        f"Applicant directly shares identifiers with {count} "
                        "confirmed fraud account(s)."
                    ),
                    evidence_status=EvidenceStatus.ANOMALOUS,
                    details={"fraud_neighbor_count": count},
                )
            )

        if features.shared_identity_degree >= 1:
            violations.append(
                EvidenceAttribution(
                    rule_id="IDENTITY_REUSE_COLLISION",
                    signal_name="shared_identity",
                    severity="REJECT",
                    score_impact=0.85,
                    description=(
                        f"National ID number was previously linked to "
                        f"{features.shared_identity_degree} other applicant(s)."
                    ),
                    evidence_status=EvidenceStatus.ANOMALOUS,
                    details={"shared_identity_degree": features.shared_identity_degree},
                )
            )

        if features.shared_phone_degree >= self.phone_reject_threshold:
            violations.append(
                EvidenceAttribution(
                    rule_id="R1_HIGH_PHONE_REUSE",
                    signal_name="shared_phone",
                    severity="REJECT",
                    score_impact=0.60,
                    description=(
                        f"Phone number linked to {features.shared_phone_degree} applications "
                        f"(threshold: >= {self.phone_reject_threshold}). Suspected burner SIM ring."
                    ),
                    evidence_status=EvidenceStatus.ANOMALOUS,
                    details={
                        "shared_phone_degree": features.shared_phone_degree,
                        "threshold": self.phone_reject_threshold,
                    },
                )
            )
        elif features.shared_phone_degree >= self.phone_review_threshold:
            violations.append(
                EvidenceAttribution(
                    rule_id="PHONE_REUSE_MODERATE",
                    signal_name="shared_phone",
                    severity="REVIEW",
                    score_impact=0.35,
                    description=(
                        f"Phone number shared with {features.shared_phone_degree - 1} other "
                        "application (possible family/household sharing)."
                    ),
                    evidence_status=EvidenceStatus.ANOMALOUS,
                    details={"shared_phone_degree": features.shared_phone_degree},
                )
            )

        if features.shared_device_degree >= self.device_reject_threshold:
            violations.append(
                EvidenceAttribution(
                    rule_id="R2_HIGH_DEVICE_REUSE",
                    signal_name="shared_device",
                    severity="REJECT",
                    score_impact=0.60,
                    description=(
                        f"Device fingerprint linked to {features.shared_device_degree} applications "
                        f"(threshold: >= {self.device_reject_threshold}). Suspected device farm."
                    ),
                    evidence_status=EvidenceStatus.ANOMALOUS,
                    details={
                        "shared_device_degree": features.shared_device_degree,
                        "threshold": self.device_reject_threshold,
                    },
                )
            )
        elif features.shared_device_degree >= self.device_review_threshold:
            violations.append(
                EvidenceAttribution(
                    rule_id="DEVICE_REUSE_MODERATE",
                    signal_name="shared_device",
                    severity="REVIEW",
                    score_impact=0.35,
                    description=(
                        f"Device fingerprint shared with {features.shared_device_degree - 1} "
                        "other application."
                    ),
                    evidence_status=EvidenceStatus.ANOMALOUS,
                    details={"shared_device_degree": features.shared_device_degree},
                )
            )

        # ---------------------------------------------------------------------
        # 2. Document Demographics & Cryptographic QR Verification
        # ---------------------------------------------------------------------
        if features.secure_qr_status == "TAMPERED" or features.secure_qr_valid is False:
            violations.append(
                EvidenceAttribution(
                    rule_id="R7_SECURE_QR_TAMPERED",
                    signal_name="secure_qr",
                    severity="REJECT",
                    score_impact=0.95,
                    description="Aadhaar Secure QR digital signature verification failed or payload tampered.",
                    evidence_status=EvidenceStatus.ANOMALOUS,
                    details={"secure_qr_status": features.secure_qr_status},
                )
            )
        elif features.secure_qr_status == "UNAVAILABLE":
            missing_signals.append("secure_qr")

        if features.id_checksum_valid is False:
            violations.append(
                EvidenceAttribution(
                    rule_id="R5_INVALID_ID_CHECKSUM",
                    signal_name="id_checksum",
                    severity="REJECT",
                    score_impact=0.75,
                    description="Official identity document checksum (Verhoeff/PAN format) failed.",
                    evidence_status=EvidenceStatus.ANOMALOUS,
                    details={"id_checksum_valid": False},
                )
            )
        elif features.id_checksum_valid is None:
            missing_signals.append("id_checksum")

        if features.ocr_name_similarity is not None:
            if features.ocr_name_similarity < self.name_similarity_threshold:
                violations.append(
                    EvidenceAttribution(
                        rule_id="R3_NAME_CONTRADICTION",
                        signal_name="name_consistency",
                        severity="REVIEW",
                        score_impact=0.35,
                        description=(
                            f"OCR name similarity ({features.ocr_name_similarity:.2f}) is below "
                            f"threshold ({self.name_similarity_threshold})."
                        ),
                        evidence_status=EvidenceStatus.ANOMALOUS,
                        details={"ocr_name_similarity": features.ocr_name_similarity},
                    )
                )
        else:
            missing_signals.append("ocr_name_similarity")

        if features.dob_mismatch_years is not None:
            if features.dob_mismatch_years > self.dob_mismatch_years_threshold:
                violations.append(
                    EvidenceAttribution(
                        rule_id="R4_DOB_CONTRADICTION",
                        signal_name="dob_consistency",
                        severity="REVIEW",
                        score_impact=0.30,
                        description=(
                            f"Extracted DOB mismatch ({features.dob_mismatch_years:.1f} years) exceeds "
                            f"{self.dob_mismatch_years_threshold} years threshold."
                        ),
                        evidence_status=EvidenceStatus.ANOMALOUS,
                        details={"dob_mismatch_years": features.dob_mismatch_years},
                    )
                )
        else:
            missing_signals.append("dob_mismatch_years")

        # ---------------------------------------------------------------------
        # 3. Biometric & Media Forensics Signals
        # ---------------------------------------------------------------------
        if features.face_similarity_status == EvidenceStatus.ANOMALOUS or (
            features.face_cosine_similarity is not None
            and features.face_cosine_similarity < self.face_similarity_threshold
        ):
            sim_val = features.face_cosine_similarity or 0.0
            violations.append(
                EvidenceAttribution(
                    rule_id="DOCUMENT_FACE_MISMATCH",
                    signal_name="face_similarity",
                    severity="REVIEW",
                    score_impact=0.40,
                    description=f"Facial similarity ({sim_val:.2f}) indicates potential identity impersonation.",
                    evidence_status=EvidenceStatus.ANOMALOUS,
                    details={"face_cosine_similarity": sim_val},
                )
            )
        elif features.face_similarity_status == EvidenceStatus.UNAVAILABLE and features.face_cosine_similarity is None:
            missing_signals.append("face_similarity")

        if (
            features.image_dimension_status == EvidenceStatus.ANOMALOUS
            or features.exif_status == EvidenceStatus.ANOMALOUS
        ):
            violations.append(
                EvidenceAttribution(
                    rule_id="IMAGE_AI_SYNTHETIC",
                    signal_name="image_metadata",
                    severity="REVIEW",
                    score_impact=0.40,
                    description="Image dimensions or metadata indicate generative AI tool synthesis.",
                    evidence_status=EvidenceStatus.ANOMALOUS,
                    details={
                        "image_dimension_status": features.image_dimension_status.value,
                        "exif_status": features.exif_status.value,
                    },
                )
            )

        if features.jpeg_quality_status == EvidenceStatus.ANOMALOUS:
            violations.append(
                EvidenceAttribution(
                    rule_id="JPEG_QUALITY_SYNTHETIC",
                    signal_name="jpeg_quality",
                    severity="REVIEW",
                    score_impact=0.30,
                    description="JPEG quantization profile matches automated synthetic export.",
                    evidence_status=EvidenceStatus.ANOMALOUS,
                    details={"jpeg_quality_status": features.jpeg_quality_status.value},
                )
            )

        if features.ela_status == EvidenceStatus.ANOMALOUS:
            violations.append(
                EvidenceAttribution(
                    rule_id="IMAGE_ELA_TAMPERED",
                    signal_name="error_level_analysis",
                    severity="REVIEW",
                    score_impact=0.35,
                    description="Error Level Analysis detected localized recompression splicing.",
                    evidence_status=EvidenceStatus.ANOMALOUS,
                    details={"ela_max_diff": features.ela_max_diff},
                )
            )

        if features.fft_status == EvidenceStatus.ANOMALOUS:
            violations.append(
                EvidenceAttribution(
                    rule_id="IMAGE_FFT_GRID_ANOMALY",
                    signal_name="fft_spectral_grid",
                    severity="REVIEW",
                    score_impact=0.30,
                    description="2D FFT power spectrum detected periodic GAN checkerboard grid artifacts.",
                    evidence_status=EvidenceStatus.ANOMALOUS,
                    details={"fft_grid_ratio": features.fft_grid_ratio},
                )
            )

        # ---------------------------------------------------------------------
        # 4. Behavioral Velocity Rules
        # ---------------------------------------------------------------------
        max_vel = max(features.velocity_1h_applications, features.device_velocity_1h)
        if max_vel >= self.velocity_reject_threshold:
            violations.append(
                EvidenceAttribution(
                    rule_id="R6_VELOCITY_BURST",
                    signal_name="velocity_burst",
                    severity="REJECT",
                    score_impact=0.50,
                    description=f"Submission velocity burst: {max_vel} applications within 1 hour.",
                    evidence_status=EvidenceStatus.ANOMALOUS,
                    details={"max_velocity_1h": max_vel},
                )
            )
        elif max_vel >= self.velocity_review_threshold:
            violations.append(
                EvidenceAttribution(
                    rule_id="VELOCITY_MODERATE",
                    signal_name="velocity_burst",
                    severity="REVIEW",
                    score_impact=0.25,
                    description=f"Elevated velocity: {max_vel} applications within 1 hour.",
                    evidence_status=EvidenceStatus.ANOMALOUS,
                    details={"max_velocity_1h": max_vel},
                )
            )

        # ---------------------------------------------------------------------
        # 5. Missing Signals Handling (Architecture Invariant Rule 4)
        # ---------------------------------------------------------------------
        attributions: list[EvidenceAttribution] = list(violations)
        for sig in missing_signals:
            attributions.append(
                EvidenceAttribution(
                    rule_id=f"MISSING_{sig.upper()}",
                    signal_name=sig,
                    severity="INFO",
                    score_impact=0.0,
                    description=f"Evidence signal '{sig}' was unavailable during evaluation.",
                    evidence_status=EvidenceStatus.UNAVAILABLE,
                    details={},
                )
            )

        # ---------------------------------------------------------------------
        # 6. Calibrated Score & Tier Aggregation
        # ---------------------------------------------------------------------
        if not violations:
            risk_score = 0.04  # Clean baseline rate
            risk_tier = RiskTier.APPROVE
            reason_codes = ["CLEAN"]
            summary = "Application passed all deterministic screening rules."
        else:
            reason_codes = [v.rule_id for v in violations]
            has_reject = any(v.severity == "REJECT" for v in violations)
            has_review = any(v.severity == "REVIEW" for v in violations)

            base_impact = sum(v.score_impact for v in violations)

            if has_reject:
                risk_tier = RiskTier.REJECT
                # Calibrated range for REJECT is [0.75, 0.99]
                risk_score = round(min(0.99, max(0.75, base_impact)), 2)
                summary = f"Rejected: {len(violations)} rule violations detected including critical triggers."
            elif has_review:
                risk_tier = RiskTier.REVIEW
                # Calibrated range for REVIEW is [0.30, 0.74]
                risk_score = round(min(0.74, max(0.30, min(0.65, base_impact))), 2)
                summary = f"Flagged for review: {len(violations)} moderate rule anomalies detected."
            else:
                risk_tier = RiskTier.APPROVE
                risk_score = round(min(0.20, max(0.04, base_impact)), 2)
                summary = "Passed screening with minor observations."

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RiskAssessment(
            applicant_id=features.applicant_id,
            model_name="M0_Rules",
            risk_score=risk_score,
            risk_tier=risk_tier,
            reason_codes=reason_codes,
            attributions=attributions,
            missing_signals=missing_signals,
            execution_time_ms=elapsed_ms,
            summary=summary,
        )


def evaluate_m0_rules_dict(evidence: dict[str, Any]) -> tuple[float, list[str]]:
    """
    Convenience helper implementing the simplified dict contract from
    MODELS_TRAINING_SPEC.md §3.3 for benchmark scripts.
    """
    app_ev = ApplicationEvidence(
        applicant_id=str(evidence.get("applicant_id", "app_benchmark")),
        shared_phone_degree=int(evidence.get("shared_phone_count", 0)),
        shared_device_degree=int(evidence.get("shared_device_count", 0)),
        ocr_name_similarity=evidence.get("ocr_name_similarity"),
        dob_mismatch_years=evidence.get("dob_age_mismatch_years"),
        id_checksum_valid=evidence.get("id_checksum_valid"),
        velocity_1h_applications=int(evidence.get("velocity_1h_applications", 1)),
    )
    model = DeterministicM0Model()
    assessment = model.predict_risk(app_ev)
    return assessment.risk_score, assessment.reason_codes
