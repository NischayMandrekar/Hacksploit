from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from app.core.fusion import (
    ApplicationEvidence,
    EvidenceAttribution,
    FraudModel,
    RiskAssessment,
    RiskTier,
)


class EvidenceAttributionSchema(BaseModel):
    rule_id: str
    signal_name: str
    severity: str
    score_impact: float
    description: str
    evidence_status: str
    details: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class RiskAssessmentResponse(BaseModel):
    applicant_id: str
    model_name: str
    risk_score: float
    risk_tier: str
    reason_codes: list[str]
    attributions: list[EvidenceAttributionSchema] = Field(default_factory=list)
    missing_signals: list[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    summary: str = ""

    model_config = ConfigDict(from_attributes=True)


__all__ = [
    "ApplicationEvidence",
    "EvidenceAttribution",
    "EvidenceAttributionSchema",
    "FraudModel",
    "RiskAssessment",
    "RiskAssessmentResponse",
    "RiskTier",
]
