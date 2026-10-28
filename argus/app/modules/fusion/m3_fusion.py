"""
ARGUS M3 — Calibrated Multi-Source Fusion (Core Novelty)
==========================================================
M3 fuses the outputs of M1 (tabular), M2 (graph), M0 (rules), document
forensics, biometrics, and explicit missing-evidence flags into a single
calibrated probability.

Calibration Method : Isotonic Regression via CalibratedClassifierCV(cv=3)
Guarantee          : P(Fraud | score = 0.20) ≈ 20%  (true observed likelihood)

Three-tier decision engine
--------------------------
  APPROVE       p  < 0.20   → Frictionless instant onboarding
  FLAG_REVIEW   0.20 ≤ p < 0.75 → Fraud Analyst Queue + SHAP reason codes
  REJECT        p  ≥ 0.75   → Coordinated synthetic fraud / severe contradiction

Input vector (X_M3)
--------------------
  m1_prob, m2_prob, m0_rule_score,
  ocr_contradiction, id_checksum_valid, tampering_ela_score,
  face_cosine_dist, is_doc_missing, is_selfie_missing, is_graph_isolated
"""

from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve,
)


# ---------------------------------------------------------------------------
# Input feature names for M3
# ---------------------------------------------------------------------------
M3_FEATURE_COLS: list[str] = [
    "m1_prob",
    "m2_prob",
    "m0_rule_score",
    "ocr_contradiction",
    "id_checksum_valid",
    "tampering_ela_score",
    "face_cosine_dist",
    "is_doc_missing",
    "is_selfie_missing",
    "is_graph_isolated",
]

# Decision thresholds — tuned on validation to freeze FPR ≤ 2%
APPROVE_THRESHOLD = 0.20
REJECT_THRESHOLD  = 0.75

# Hard-override rules (bypass probability tiers)
HARD_OVERRIDE_ELA_SCORE    = 0.85
HARD_OVERRIDE_CHECKSUM_BAD = 0       # 0 = invalid


# ---------------------------------------------------------------------------
# Risk tier classification
# ---------------------------------------------------------------------------

def classify_risk_tier(
    calibrated_prob: float,
    evidence: dict,
) -> tuple[str, str]:
    """
    Map a calibrated M3 probability to an operational decision tier.

    Hard overrides fire BEFORE probabilistic thresholds.

    Parameters
    ----------
    calibrated_prob : float — M3 calibrated fraud probability [0, 1].
    evidence        : dict  — Raw evidence flags (id_checksum_valid, tampering_ela_score).

    Returns
    -------
    (tier, recommended_action)
    """
    # --- Hard contradiction override ---
    checksum_bad   = evidence.get("id_checksum_valid", 1) == HARD_OVERRIDE_CHECKSUM_BAD
    extreme_tamper = evidence.get("tampering_ela_score", 0.0) > HARD_OVERRIDE_ELA_SCORE
    if checksum_bad and extreme_tamper:
        return (
            "REJECT",
            "Automatic rejection: Critical document forgery confirmed by "
            "Verhoeff checksum failure + extreme ELA splice score.",
        )

    # --- Probabilistic tiers ---
    if calibrated_prob < APPROVE_THRESHOLD:
        return ("APPROVE", "Low risk: Frictionless automated account approval.")
    elif calibrated_prob < REJECT_THRESHOLD:
        return (
            "FLAG_REVIEW",
            "Moderate risk / anomaly detected: Queued for fraud analyst investigation.",
        )
    else:
        return ("REJECT", "High risk: Coordinated synthetic fraud pattern detected.")


# ---------------------------------------------------------------------------
# M3 Model Wrapper
# ---------------------------------------------------------------------------

class M3CalibratedFusionModel:
    """
    Calibrated multi-source fusion model.

    Wraps a LGBMClassifier base estimator inside
    CalibratedClassifierCV(method='isotonic', cv=3) to ensure that
    predicted probabilities reflect true observed fraud likelihood.
    """

    BASE_PARAMS: dict = {
        "objective":        "binary",
        "n_estimators":     160,
        "learning_rate":    0.04,
        "max_depth":        4,
        "num_leaves":       15,
        "scale_pos_weight": 4.0,
        "random_state":     42,
        "verbosity":        -1,
    }

    def __init__(self, base_params: dict | None = None):
        params = base_params or self.BASE_PARAMS
        base = lgb.LGBMClassifier(**params)
        self._model = CalibratedClassifierCV(
            estimator=base,
            method="isotonic",
            cv=3,
        )
        self.feature_cols = M3_FEATURE_COLS

    # ------------------------------------------------------------------
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> "M3CalibratedFusionModel":
        self._model.fit(X_train[self.feature_cols], y_train)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated fraud probability, shape (n,)."""
        return self._model.predict_proba(X[self.feature_cols])[:, 1]

    # ------------------------------------------------------------------
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
        preds = self.predict_proba(X_test)
        negatives = preds[y_test == 0]
        fpr5_thresh = np.percentile(negatives, 95.0)
        recall_at_5fpr = float(np.mean(preds[y_test == 1] >= fpr5_thresh))

        return {
            "pr_auc":             round(average_precision_score(y_test, preds), 4),
            "roc_auc":            round(roc_auc_score(y_test, preds), 4),
            "brier_score":        round(brier_score_loss(y_test, preds), 4),
            "recall_at_5pct_fpr": round(recall_at_5fpr, 4),
        }

    # ------------------------------------------------------------------
    def score_and_explain(
        self,
        X_row: pd.DataFrame,
        evidence: dict,
    ) -> dict:
        """
        Score a single applicant and produce an explainability dossier.
        (SHAP values require the ``shap`` package; falls back gracefully.)

        Returns
        -------
        dict with keys: applicant_id, calibrated_risk_score, risk_tier,
                        recommended_action, top_risk_contributors
        """
        prob = float(self.predict_proba(X_row)[0])
        tier, action = classify_risk_tier(prob, evidence)

        # --- Feature contribution (SHAP if available, else raw values) ---
        contributors: list[dict] = []
        try:
            import shap
            explainer = shap.TreeExplainer(
                self._model.calibrated_classifiers_[0].estimator
            )
            shap_vals = explainer.shap_values(X_row[self.feature_cols])
            # shap_vals shape: (n_classes, n_samples, n_features) for LGBM
            sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
            for fname, impact in sorted(
                zip(self.feature_cols, sv),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]:
                contributors.append({
                    "feature": fname,
                    "shap_impact": round(float(impact), 4),
                    "raw_value": round(float(X_row[fname].iloc[0]), 4),
                })
        except Exception:
            # Fall back to raw feature values sorted by magnitude
            for fname in self.feature_cols:
                contributors.append({
                    "feature":    fname,
                    "raw_value":  round(float(X_row[fname].iloc[0]), 4),
                })

        return {
            "calibrated_risk_score": round(prob, 4),
            "risk_tier":             tier,
            "recommended_action":    action,
            "top_risk_contributors": contributors,
        }

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: str | Path) -> "M3CalibratedFusionModel":
        instance = cls.__new__(cls)
        instance.feature_cols = M3_FEATURE_COLS
        instance._model = joblib.load(path)
        return instance
