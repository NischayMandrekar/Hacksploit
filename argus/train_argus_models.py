"""
ARGUS — Real-Data Training Harness (IEEE-CIS Fraud Detection)
==============================================================
Trains M1, M2, M3 on the real IEEE-CIS Kaggle competition dataset.

Dataset required in argus/dataset/:
    train_transaction.csv
    train_identity.csv

Download first:
    python download_dataset.py

Then train:
    $env:PYTHONUTF8=1; python train_argus_models.py

For faster dev runs (use 20% of data):
    $env:PYTHONUTF8=1; python train_argus_models.py --subsample 0.2
"""

from __future__ import annotations

import sys
import json
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve,
)

# ── Local modules ─────────────────────────────────────────────────────────────
from app.data.ieee_cis_loader import (
    load_and_engineer, time_split,
    M1_FEATURE_COLS, M2_FEATURE_COLS, M3_FEATURE_COLS,
)
from app.modules.tabular.m1_tabular import M1TabularModel
from app.modules.graph.m2_graph     import M2GraphModel
from app.modules.fusion.m3_fusion   import M3CalibratedFusionModel, classify_risk_tier

# ── Config ────────────────────────────────────────────────────────────────────
ARTIFACT_DIR = os.path.join("app", "modules", "fusion", "weights")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

BANNER = "=" * 80
LABEL  = "isFraud"


def _header(title: str) -> None:
    print(f"\n{'-' * 80}\n>>> {title}\n{'-' * 80}")


# ── Override M1/M2 feature cols for IEEE-CIS ──────────────────────────────────
# (the model wrappers use class-level defaults; we override here at fit time)

def _m1_fit(df_tr, y_tr, df_va, y_va):
    from lightgbm import LGBMClassifier, early_stopping
    n_pos = y_tr.sum(); n_neg = len(y_tr) - n_pos
    spw = round(n_neg / max(n_pos, 1), 1)
    clf = LGBMClassifier(
        objective="binary", boosting_type="gbdt",
        learning_rate=0.03, num_leaves=63, max_depth=7,
        min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, n_estimators=500,
        random_state=42, verbosity=-1, n_jobs=-1,
    )
    clf.fit(
        df_tr[M1_FEATURE_COLS], y_tr,
        eval_X=df_va[M1_FEATURE_COLS], eval_y=y_va,
        callbacks=[early_stopping(stopping_rounds=40, verbose=False)],
    )
    return clf


def _m2_fit(df_tr, y_tr, df_va, y_va):
    from lightgbm import LGBMClassifier, early_stopping
    n_pos = y_tr.sum(); n_neg = len(y_tr) - n_pos
    spw = round(n_neg / max(n_pos, 1), 1)
    clf = LGBMClassifier(
        objective="binary", boosting_type="gbdt",
        learning_rate=0.03, num_leaves=63, max_depth=8,
        min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, n_estimators=500,
        random_state=42, verbosity=-1, n_jobs=-1,
    )
    clf.fit(
        df_tr[M2_FEATURE_COLS], y_tr,
        eval_X=df_va[M2_FEATURE_COLS], eval_y=y_va,
        callbacks=[early_stopping(stopping_rounds=40, verbose=False)],
    )
    return clf


def build_fusion_df(df, m1_clf, m2_clf) -> pd.DataFrame:
    return pd.DataFrame({
        "m1_prob":             m1_clf.predict_proba(df[M1_FEATURE_COLS])[:, 1],
        "m2_prob":             m2_clf.predict_proba(df[M2_FEATURE_COLS])[:, 1],
        "m0_rule_score":       df["m0_rule_score"].values,
        "ocr_contradiction":   df["ocr_contradiction"].values,
        "id_checksum_valid":   df["id_checksum_valid"].values,
        "tampering_ela_score": df["tampering_ela_score"].values,
        "face_cosine_dist":    df["face_cosine_dist"].values,
        "is_doc_missing":      df["is_identity_missing"].values,
        "is_selfie_missing":   df["is_device_missing"].values,
        "is_graph_isolated":   df["is_graph_isolated"].values,
    }, index=df.index)


def metrics(y_true, y_pred) -> dict:
    pr  = round(average_precision_score(y_true, y_pred), 4)
    roc = round(roc_auc_score(y_true, y_pred), 4)
    brier = round(brier_score_loss(y_true, y_pred), 4)
    neg_preds = y_pred[y_true == 0]
    thresh5   = np.percentile(neg_preds, 95.0)
    recall5   = round(float(np.mean(y_pred[y_true == 1] >= thresh5)), 4)
    return {"pr_auc": pr, "roc_auc": roc, "brier": brier, "recall_at_5pct_fpr": recall5}


def attack_matrix(test_df, y_test, m1_clf, m2_clf, m3_model):
    """Adversarial edge-deletion and camouflage benchmark."""
    rng = np.random.default_rng(99)
    results = []

    def _score(df_copy):
        m1_p = m1_clf.predict_proba(df_copy[M1_FEATURE_COLS])[:, 1]
        m2_p = m2_clf.predict_proba(df_copy[M2_FEATURE_COLS])[:, 1]
        X_f  = build_fusion_df(df_copy, m1_clf, m2_clf)
        m3_p = m3_model.predict_proba(X_f)
        m0_p = df_copy["m0_rule_score"].values
        return {
            "M0 (Rules)":   f"{average_precision_score(y_test, m0_p):.4f}",
            "M1 (Tabular)": f"{average_precision_score(y_test, m1_p):.4f}",
            "M2 (Graph)":   f"{average_precision_score(y_test, m2_p):.4f}",
            "M3 (ARGUS)":   f"{average_precision_score(y_test, m3_p):.4f}",
        }

    # Clean
    r = _score(test_df.copy()); r["Scenario"] = "Clean Baseline"; results.append(r)

    # 10% card sharing deleted
    df_10 = test_df.copy()
    df_10.loc[rng.binomial(1, 0.10, len(df_10)).astype(bool), "shared_card1_count"] = 1
    r = _score(df_10); r["Scenario"] = "10% Edge Deletion"; results.append(r)

    # 20% deletion
    df_20 = test_df.copy()
    mask = rng.binomial(1, 0.20, len(df_20)).astype(bool)
    df_20.loc[mask, ["shared_card1_count", "shared_device_count"]] = 1
    r = _score(df_20); r["Scenario"] = "20% Edge Deletion"; results.append(r)

    # 30% deletion
    df_30 = test_df.copy()
    mask = rng.binomial(1, 0.30, len(df_30)).astype(bool)
    df_30.loc[mask, ["shared_card1_count", "shared_device_count", "shared_addr1_count"]] = 1
    r = _score(df_30); r["Scenario"] = "30% Edge Deletion"; results.append(r)

    # Adversarial camouflage
    df_camo = test_df.copy()
    fraud_mask = y_test == 1
    df_camo.loc[fraud_mask, "shared_card1_count"]  = 2
    df_camo.loc[fraud_mask, "shared_device_count"] = 2
    r = _score(df_camo); r["Scenario"] = "Adversarial Camouflage"; results.append(r)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def train_and_evaluate(subsample: float | None = None) -> None:
    t0 = time.time()

    print(BANNER)
    print("  PROJECT ARGUS — IEEE-CIS Real Fraud Dataset Training")
    print("  M1 (Tabular) | M2 (Graph-Enhanced) | M3 (Calibrated Fusion)")
    print(BANNER)

    # ── 0. Load data ──────────────────────────────────────────────────────
    _header("[Phase 0] Loading & Engineering IEEE-CIS Dataset")
    df = load_and_engineer(subsample_frac=subsample)
    train_df, val_df, test_df = time_split(df)
    y_tr = train_df[LABEL].values
    y_va = val_df[LABEL].values
    y_te = test_df[LABEL].values

    print(f"  Train: {len(train_df):,} rows  |  fraud: {y_tr.mean()*100:.2f}%")
    print(f"  Val  : {len(val_df):,} rows  |  fraud: {y_va.mean()*100:.2f}%")
    print(f"  Test : {len(test_df):,} rows  |  fraud: {y_te.mean()*100:.2f}%")

    # ── 1. M1 ─────────────────────────────────────────────────────────────
    _header("[Phase 1] Training M1: Tabular LightGBM")
    m1_clf = _m1_fit(train_df, y_tr, val_df, y_va)
    m1_te  = m1_clf.predict_proba(test_df[M1_FEATURE_COLS])[:, 1]
    m1_m   = metrics(y_te, m1_te)
    print(f"  [M1] PR-AUC: {m1_m['pr_auc']:.4f}  ROC-AUC: {m1_m['roc_auc']:.4f}")

    # ── 2. M2 ─────────────────────────────────────────────────────────────
    _header("[Phase 2] Training M2: Graph-Enhanced LightGBM")
    m2_clf = _m2_fit(train_df, y_tr, val_df, y_va)
    m2_te  = m2_clf.predict_proba(test_df[M2_FEATURE_COLS])[:, 1]
    m2_m   = metrics(y_te, m2_te)
    graph_lift = m2_m["pr_auc"] - m1_m["pr_auc"]
    print(f"  [M2] PR-AUC: {m2_m['pr_auc']:.4f}  ROC-AUC: {m2_m['roc_auc']:.4f}  Lift: {graph_lift:+.4f}")

    # ── 3. M3 ─────────────────────────────────────────────────────────────
    _header("[Phase 3] Training M3: Calibrated Multi-Source Fusion")
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV

    X_ftr = build_fusion_df(train_df, m1_clf, m2_clf)
    X_fte = build_fusion_df(test_df,  m1_clf, m2_clf)

    base = LGBMClassifier(
        objective="binary", n_estimators=200,
        learning_rate=0.04, max_depth=4, num_leaves=15,
        scale_pos_weight=round((y_tr == 0).sum() / max(y_tr.sum(), 1) * 0.5, 1),
        random_state=42, verbosity=-1,
    )
    m3_cal = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
    m3_cal.fit(X_ftr, y_tr)

    m3_te = m3_cal.predict_proba(X_fte)[:, 1]
    m3_m  = metrics(y_te, m3_te)
    fusion_lift = m3_m["pr_auc"] - m2_m["pr_auc"]

    print(f"  [M3] PR-AUC          : {m3_m['pr_auc']:.4f}")
    print(f"  [M3] ROC-AUC         : {m3_m['roc_auc']:.4f}")
    print(f"  [M3] Brier Score     : {m3_m['brier']:.4f}  (lower = better calibration)")
    print(f"  [M3] Recall @ 5% FPR : {m3_m['recall_at_5pct_fpr']*100:.2f}%  (target >= 60%)")
    print(f"  [M3] Lift over M2    : {fusion_lift:+.4f}  |  Total over M1: {m3_m['pr_auc']-m1_m['pr_auc']:+.4f}")

    # ── 4. Adversarial matrix ──────────────────────────────────────────────
    _header("[Phase 4] Adversarial Camouflage & Edge-Deletion Benchmark")

    class _M3Wrapper:
        def predict_proba(self, X):
            return m3_cal.predict_proba(X)[:, 1]

    atk_results = attack_matrix(test_df, y_te, m1_clf, m2_clf, _M3Wrapper())
    try:
        from tabulate import tabulate
        print("\n" + tabulate(atk_results, headers="keys", tablefmt="github"))
    except ImportError:
        print("\n" + pd.DataFrame(atk_results).to_string(index=False))

    # ── 5. Tier distribution ───────────────────────────────────────────────
    _header("[Phase 5] Risk Tier Distribution on Test Set")
    approve = (m3_te < 0.20).mean() * 100
    review  = ((m3_te >= 0.20) & (m3_te < 0.75)).mean() * 100
    reject  = (m3_te >= 0.75).mean() * 100
    print(f"  APPROVE      (p < 0.20)       : {approve:.1f}%")
    print(f"  FLAG_REVIEW  (0.20 <= p < 0.75): {review:.1f}%")
    print(f"  REJECT       (p >= 0.75)       : {reject:.1f}%")

    # ── 6. Serialize ───────────────────────────────────────────────────────
    _header("[Phase 6] Serializing Model Weights")
    joblib.dump(m1_clf,  os.path.join(ARTIFACT_DIR, "m1_tabular.joblib"))
    joblib.dump(m2_clf,  os.path.join(ARTIFACT_DIR, "m2_graph.joblib"))
    joblib.dump(m3_cal,  os.path.join(ARTIFACT_DIR, "m3_fusion.joblib"))

    # Also save to models/ as .pt for PyTorch-native access
    import torch
    MODELS_DIR = "models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, clf in [("m1_tabular", m1_clf), ("m2_graph", m2_clf), ("m3_fusion", m3_cal)]:
        torch.save({"model": clf, "format": "sklearn/lgbm", "name": name},
                   os.path.join(MODELS_DIR, f"{name}.pt"))


    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "IEEE-CIS Fraud Detection (Kaggle)",
        "subsample_frac": subsample,
        "n_train": int(len(train_df)), "n_val": int(len(val_df)), "n_test": int(len(test_df)),
        "fraud_rate_pct": round(float(y_te.mean() * 100), 2),
        "metrics": {
            "m1_tabular": m1_m,
            "m2_graph":   m2_m,
            "m3_fusion":  m3_m,
        },
        "lifts": {
            "graph_over_tabular":   round(graph_lift, 4),
            "fusion_over_graph":    round(fusion_lift, 4),
            "total_over_tabular":   round(m3_m["pr_auc"] - m1_m["pr_auc"], 4),
        },
        "feature_names": {
            "m1": M1_FEATURE_COLS,
            "m2": M2_FEATURE_COLS,
            "m3": M3_FEATURE_COLS,
        },
        "adversarial_matrix": atk_results,
    }
    with open(os.path.join(ARTIFACT_DIR, "benchmark_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  [DONE] m1_tabular.joblib")
    print(f"  [DONE] m2_graph.joblib")
    print(f"  [DONE] m3_fusion.joblib")
    print(f"  [DONE] benchmark_report.json")
    print(f"\n  Total time: {elapsed:.1f}s")
    print(BANNER)
    print("  TRAINING COMPLETE on REAL IEEE-CIS DATA — models calibrated & serialized.")
    print(BANNER)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=float, default=None,
                        help="Fraction of data to use (e.g. 0.2 for fast dev runs)")
    args = parser.parse_args()
    train_and_evaluate(subsample=args.subsample)
