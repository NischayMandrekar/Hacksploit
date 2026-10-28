"""
ARGUS M0 — Deterministic Rule Engine
=====================================
Executes in <10ms synchronously during application ingestion.
Any rule trigger immediately routes the application to analyst review
or flags it for high-depth forensic processing.

Rules are hard-coded fraud heuristics derived from known synthetic identity
attack patterns (SIM farms, device farms, cross-modal contradictions).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Rule Codes & Thresholds
# ---------------------------------------------------------------------------
RULE_THRESHOLDS = {
    "R1_HIGH_PHONE_REUSE":    ("shared_phone_count",        3,    ">="),
    "R2_HIGH_DEVICE_REUSE":   ("shared_device_count",       3,    ">="),
    "R3_NAME_CONTRADICTION":  ("ocr_name_similarity",       0.75, "<"),
    "R4_DOB_CONTRADICTION":   ("dob_age_mismatch_years",    2,    ">"),
    "R5_INVALID_ID_CHECKSUM": ("id_checksum_valid",         True, "not"),
    "R6_VELOCITY_BURST":      ("velocity_1h_applications",  4,    ">="),
}


@dataclass
class M0Result:
    rule_score: float
    triggered_rules: List[str]
    routed_to_review: bool = field(init=False)

    def __post_init__(self):
        self.routed_to_review = len(self.triggered_rules) > 0


def evaluate_m0_rules(evidence: dict) -> M0Result:
    """
    Evaluate deterministic KYC rules against an applicant evidence dict.

    Parameters
    ----------
    evidence : dict
        Keys correspond to RULE_THRESHOLDS field names (see above).

    Returns
    -------
    M0Result
        rule_score     : float in [0.0, 1.0] — 0.25 per triggered rule, capped at 1.0
        triggered_rules: list of rule code strings that fired
        routed_to_review: True when any rule triggers
    """
    triggered: List[str] = []

    # --- R1: High phone-number reuse (burner SIM sharing) ---
    if evidence.get("shared_phone_count", 0) >= 3:
        triggered.append("R1_HIGH_PHONE_REUSE")

    # --- R2: High device fingerprint reuse (emulator / device farm) ---
    if evidence.get("shared_device_count", 0) >= 3:
        triggered.append("R2_HIGH_DEVICE_REUSE")

    # --- R3: OCR-extracted name diverges from claimed name ---
    if evidence.get("ocr_name_similarity", 1.0) < 0.75:
        triggered.append("R3_NAME_CONTRADICTION")

    # --- R4: Age computed from DOB contradicts claimed age ---
    if evidence.get("dob_age_mismatch_years", 0) > 2:
        triggered.append("R4_DOB_CONTRADICTION")

    # --- R5: National-ID / Aadhaar Verhoeff checksum failure ---
    if not evidence.get("id_checksum_valid", True):
        triggered.append("R5_INVALID_ID_CHECKSUM")

    # --- R6: Application velocity burst (> 4 in 1 hour) ---
    if evidence.get("velocity_1h_applications", 0) >= 4:
        triggered.append("R6_VELOCITY_BURST")

    rule_score = min(1.0, len(triggered) * 0.25)
    return M0Result(rule_score=rule_score, triggered_rules=triggered)


# ---------------------------------------------------------------------------
# Quick sanity demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = {
        "shared_phone_count": 5,
        "shared_device_count": 1,
        "ocr_name_similarity": 0.60,
        "dob_age_mismatch_years": 3,
        "id_checksum_valid": False,
        "velocity_1h_applications": 6,
    }
    result = evaluate_m0_rules(sample)
    print(f"M0 Rule Score : {result.rule_score}")
    print(f"Triggered     : {result.triggered_rules}")
    print(f"Route Review? : {result.routed_to_review}")
