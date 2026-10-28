import time
from dataclasses import dataclass, field
from typing import Any
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.modules.graph.engine import _to_pseudonymized_hash
from app.modules.graph.models import ApplicantEdge, ApplicantNode, EdgeType


@dataclass
class RuleViolation:
    rule_id: str
    severity: str  # "REJECT", "REVIEW", "INFO"
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)
    score_impact: float = 0.0


@dataclass
class RuleEvaluationResult:
    applicant_id: str
    risk_score: float
    risk_tier: str  # "APPROVE", "REVIEW", "REJECT"
    reason_codes: list[str]
    violations: list[RuleViolation]
    shared_counts: dict[str, int]
    execution_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicant_id": self.applicant_id,
            "risk_score": self.risk_score,
            "risk_tier": self.risk_tier,
            "reason_codes": self.reason_codes,
            "violations": [
                {
                    "rule_id": v.rule_id,
                    "severity": v.severity,
                    "description": v.description,
                    "evidence": v.evidence,
                    "score_impact": v.score_impact,
                }
                for v in self.violations
            ],
            "shared_counts": self.shared_counts,
            "execution_time_ms": self.execution_time_ms,
        }


class M0RuleEngine:
    """
    Synchronous fast-path rule engine (M0 Baseline).
    Executes in < 15ms during application ingestion using indexed 1-hop lookups.
    Differentiates benign sharing (e.g. family sharing degree 2) from organized ring attacks (degree >= 3).
    """

    def __init__(
        self,
        db: Session,
        phone_review_threshold: int = 2,
        phone_reject_threshold: int = 3,
        device_review_threshold: int = 2,
        device_reject_threshold: int = 3,
    ) -> None:
        self.db = db
        self.phone_review_threshold = phone_review_threshold
        self.phone_reject_threshold = phone_reject_threshold
        self.device_review_threshold = device_review_threshold
        self.device_reject_threshold = device_reject_threshold

    def evaluate_applicant(
        self,
        applicant_id: str,
        identifiers: dict[str, Any],
        application_context: dict[str, Any] | None = None,
    ) -> RuleEvaluationResult:
        """
        Evaluates 1-hop graph collisions, confirmed fraud connections, and applicant context.
        Returns a deterministic decision with human-readable reason codes.
        """
        start_time = time.perf_counter()
        violations: list[RuleViolation] = []
        ctx = application_context or {}

        # 1. Normalize/hash identifiers
        phone_hash = _to_pseudonymized_hash(identifiers.get("phone"), is_numeric_id=True)
        device_hash = _to_pseudonymized_hash(
            identifiers.get("device_fingerprint") or identifiers.get("device_hash")
        )
        identity_hash = _to_pseudonymized_hash(
            identifiers.get("identity_number")
            or identifiers.get("aadhaar")
            or identifiers.get("ssn"),
            is_numeric_id=True,
        )
        email_hash = _to_pseudonymized_hash(identifiers.get("email"))

        shared_counts: dict[str, int] = {
            "phone": 0,
            "device": 0,
            "identity": 0,
            "email": 0,
        }

        other_phone_apps: list[str] = []
        other_device_apps: list[str] = []
        other_identity_apps: list[str] = []

        # 2. Query 1-hop collisions using composite indices on (target_hash, edge_type)
        if phone_hash:
            other_phone_apps = list(
                self.db.scalars(
                    select(ApplicantEdge.source_hash).where(
                        ApplicantEdge.target_hash == phone_hash,
                        ApplicantEdge.edge_type == EdgeType.HAS_PHONE.value,
                        ApplicantEdge.source_hash != applicant_id,
                    )
                ).all()
            )
            shared_counts["phone"] = len(set(other_phone_apps))

        if device_hash:
            other_device_apps = list(
                self.db.scalars(
                    select(ApplicantEdge.source_hash).where(
                        ApplicantEdge.target_hash == device_hash,
                        ApplicantEdge.edge_type == EdgeType.HAS_DEVICE.value,
                        ApplicantEdge.source_hash != applicant_id,
                    )
                ).all()
            )
            shared_counts["device"] = len(set(other_device_apps))

        if identity_hash:
            other_identity_apps = list(
                self.db.scalars(
                    select(ApplicantEdge.source_hash).where(
                        ApplicantEdge.target_hash == identity_hash,
                        ApplicantEdge.edge_type == EdgeType.HAS_IDENTITY.value,
                        ApplicantEdge.source_hash != applicant_id,
                    )
                ).all()
            )
            shared_counts["identity"] = len(set(other_identity_apps))

        if email_hash:
            other_email_apps = list(
                self.db.scalars(
                    select(ApplicantEdge.source_hash).where(
                        ApplicantEdge.target_hash == email_hash,
                        ApplicantEdge.edge_type == EdgeType.HAS_EMAIL.value,
                        ApplicantEdge.source_hash != applicant_id,
                    )
                ).all()
            )
            shared_counts["email"] = len(set(other_email_apps))

        # 3. Check for direct links to confirmed fraud accounts
        all_neighbor_apps = set(other_phone_apps) | set(other_device_apps) | set(other_identity_apps)
        if all_neighbor_apps:
            fraud_neighbors = list(
                self.db.scalars(
                    select(ApplicantNode.applicant_id).where(
                        ApplicantNode.applicant_id.in_(all_neighbor_apps),
                        or_(
                            ApplicantNode.is_fraud.is_(True),
                            ApplicantNode.status == "fraud",
                            ApplicantNode.status == "rejected",
                            ApplicantNode.risk_score >= 0.90,
                        ),
                    )
                ).all()
            )
            if fraud_neighbors:
                violations.append(
                    RuleViolation(
                        rule_id="CONFIRMED_FRAUD_LINK",
                        severity="REJECT",
                        description=(
                            f"Applicant shares identifiers with {len(fraud_neighbors)} "
                            "confirmed fraud account(s)."
                        ),
                        evidence={"fraud_applicants": fraud_neighbors},
                        score_impact=1.0,
                    )
                )

        # 4. Aadhaar / National ID reuse check
        if shared_counts["identity"] >= 1:
            violations.append(
                RuleViolation(
                    rule_id="IDENTITY_REUSE_COLLISION",
                    severity="REJECT",
                    description=(
                        "Aadhaar/National ID number was previously registered under "
                        f"{shared_counts['identity']} different applicant profile(s)."
                    ),
                    evidence={"colliding_applicants": list(set(other_identity_apps))},
                    score_impact=0.85,
                )
            )

        phone_degree = shared_counts["phone"] + 1 if shared_counts["phone"] > 0 else 1
        device_degree = shared_counts["device"] + 1 if shared_counts["device"] > 0 else 1

        # 5. Phone reuse checks
        if phone_degree >= self.phone_reject_threshold:
            violations.append(
                RuleViolation(
                    rule_id="R1_HIGH_PHONE_REUSE",
                    severity="REJECT",
                    description=(
                        f"Phone number is linked to {phone_degree} applications "
                        f"(threshold: >= {self.phone_reject_threshold}). Suspected burner SIM ring."
                    ),
                    evidence={
                        "phone_degree": phone_degree,
                        "shared_count": shared_counts["phone"],
                        "threshold": self.phone_reject_threshold,
                    },
                    score_impact=0.60,
                )
            )
        elif phone_degree >= self.phone_review_threshold:
            violations.append(
                RuleViolation(
                    rule_id="PHONE_REUSE_MODERATE",
                    severity="REVIEW",
                    description=(
                        f"Phone number is shared with {shared_counts['phone']} other application "
                        "(possible benign family/household sharing)."
                    ),
                    evidence={
                        "phone_degree": phone_degree,
                        "shared_count": shared_counts["phone"],
                    },
                    score_impact=0.35,
                )
            )

        # 6. Device reuse checks
        if device_degree >= self.device_reject_threshold:
            violations.append(
                RuleViolation(
                    rule_id="R2_HIGH_DEVICE_REUSE",
                    severity="REJECT",
                    description=(
                        f"Device fingerprint is linked to {device_degree} applications "
                        f"(threshold: >= {self.device_reject_threshold}). Suspected emulator/fraud farm."
                    ),
                    evidence={
                        "device_degree": device_degree,
                        "shared_count": shared_counts["device"],
                        "threshold": self.device_reject_threshold,
                    },
                    score_impact=0.60,
                )
            )
        elif device_degree >= self.device_review_threshold:
            violations.append(
                RuleViolation(
                    rule_id="DEVICE_REUSE_MODERATE",
                    severity="REVIEW",
                    description=(
                        f"Device fingerprint is shared with {shared_counts['device']} other application."
                    ),
                    evidence={
                        "device_degree": device_degree,
                        "shared_count": shared_counts["device"],
                    },
                    score_impact=0.35,
                )
            )

        # 7. Application context & cross-modal contradiction checks
        ocr_sim = ctx.get("ocr_name_similarity")
        if ocr_sim is not None and float(ocr_sim) < 0.75:
            violations.append(
                RuleViolation(
                    rule_id="R3_NAME_CONTRADICTION",
                    severity="REVIEW",
                    description=f"OCR extracted name similarity ({ocr_sim:.2f}) is below 0.75 threshold.",
                    evidence={"ocr_name_similarity": ocr_sim},
                    score_impact=0.35,
                )
            )

        dob_mismatch = ctx.get("dob_age_mismatch_years")
        if dob_mismatch is not None and float(dob_mismatch) > 2.0:
            violations.append(
                RuleViolation(
                    rule_id="R4_DOB_CONTRADICTION",
                    severity="REVIEW",
                    description=f"DOB mismatch of {dob_mismatch} years exceeds 2-year threshold.",
                    evidence={"dob_age_mismatch_years": dob_mismatch},
                    score_impact=0.30,
                )
            )

        if ctx.get("id_checksum_valid") is False:
            violations.append(
                RuleViolation(
                    rule_id="R5_INVALID_ID_CHECKSUM",
                    severity="REJECT",
                    description="Official identity document checksum algorithm (Verhoeff/Luhn) failed.",
                    evidence={"id_checksum_valid": False},
                    score_impact=0.75,
                )
            )

        velocity_1h = ctx.get("velocity_1h_applications")
        if velocity_1h is not None and int(velocity_1h) >= 4:
            violations.append(
                RuleViolation(
                    rule_id="R6_VELOCITY_BURST",
                    severity="REJECT",
                    description=f"Velocity burst: {velocity_1h} applications submitted in last 1 hour.",
                    evidence={"velocity_1h_applications": velocity_1h},
                    score_impact=0.50,
                )
            )

        # 8. Compute calibrated score and risk tier
        if not violations:
            risk_score = 0.04
            risk_tier = "APPROVE"
            reason_codes = ["CLEAN"]
        else:
            reason_codes = [v.rule_id for v in violations]
            has_reject = any(v.severity == "REJECT" for v in violations)
            has_review = any(v.severity == "REVIEW" for v in violations)

            base_impact = sum(v.score_impact for v in violations)
            risk_score = round(min(0.99, max(0.10, base_impact)), 2)

            if has_reject or risk_score >= 0.70:
                risk_tier = "REJECT"
            elif has_review or risk_score >= 0.30:
                risk_tier = "REVIEW"
            else:
                risk_tier = "APPROVE"

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return RuleEvaluationResult(
            applicant_id=applicant_id,
            risk_score=risk_score,
            risk_tier=risk_tier,
            reason_codes=reason_codes,
            violations=violations,
            shared_counts=shared_counts,
            execution_time_ms=duration_ms,
        )
