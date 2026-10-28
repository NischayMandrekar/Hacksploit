"""
ARGUS M1 — Tabular LightGBM Baseline
======================================
Evaluates predictive signal from non-relational applicant records and
temporal behavioral velocity, establishing the ML baseline BEFORE any
graph-topology or forensic features are introduced.

Dataset origin  : Bank Account Fraud (BAF) — NeurIPS 2022 (Feedzai)
Training split  : Time-based (months 0-5 train | 6 val | 7 test) to prevent leakage.
Target variable : fraud_bool  (1 = synthetic/account fraud, 0 = legitimate)
Class imbalance : ~1.1% fraud prevalence → handled via scale_pos_weight
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from pathlib import Path


# ---------------------------------------------------------------------------
# Feature columns consumed by M1 (non-relational)
# ---------------------------------------------------------------------------
M1_FEATURE_COLS: list[str] = [
    "customer_age",
    "income_lakhs",
    "account_type_risk",
    "velocity_1h",
    "velocity_24h",
]

# Hyperparameters tuned for ~1–6 % fraud prevalence on BAF-equivalent data
M1_PARAMS: dict = {
    "objective":        "binary",
    "boosting_type":    "gbdt",
    "metric":           "binary_logloss",
    "learning_rate":    0.03,
    "num_leaves":       31,
    "max_depth":        5,
    "min_child_samples": 50,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 6.0,   # ≈ (n_neg / n_pos) for ~6% fraud
    "n_estimators":     120,
    "random_state":     42,
    "verbosity":        -1,
    "n_jobs":           -1,
}


class M1TabularModel:
    """
    Thin wrapper around LGBMClassifier for the tabular-only baseline.

    Usage
    -----
    model = M1TabularModel()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X)          # shape (n,)
    prauc = model.evaluate(X_test, y_test)
    model.save("weights/m1_tabular.joblib")
    model = M1TabularModel.load("weights/m1_tabular.joblib")
    """

    def __init__(self, params: dict | None = None):
        self.params = params or M1_PARAMS
        self._clf = lgb.LGBMClassifier(**self.params)
        self.feature_cols = M1_FEATURE_COLS

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> "M1TabularModel":
        """Train M1.  Optionally provide a validation set for early stopping."""
        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val[self.feature_cols], y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=50, verbose=False)
            ]
        self._clf.fit(X_train[self.feature_cols], y_train, **fit_kwargs)
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probability scores, shape (n,)."""
        return self._clf.predict_proba(X[self.feature_cols])[:, 1]

    # ------------------------------------------------------------------
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
        preds = self.predict_proba(X_test)
        return {
            "pr_auc":  round(average_precision_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, preds), 4),
        }

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, path)

    @classmethod
    def load(cls, path: str | Path) -> "M1TabularModel":
        instance = cls.__new__(cls)
        instance.feature_cols = M1_FEATURE_COLS
        instance.params = M1_PARAMS
        instance._clf = joblib.load(path)
        return instance
