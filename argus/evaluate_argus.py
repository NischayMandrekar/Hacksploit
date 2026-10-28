"""
ARGUS — Standalone Evaluation Script
======================================
Loads trained M1/M2/M3 weights and runs a comprehensive evaluation on the
held-out test set (final 15% by TransactionDT, never seen during training).

Usage:
    $env:PYTHONUTF8=1; python evaluate_argus.py
    $env:PYTHONUTF8=1; python evaluate_argus.py --subsample 0.2   # fast dev run
    $env:PYTHONUTF8=1; python evaluate_argus.py --shap             # enable SHAP

Outputs:
    * Console report with all metrics
    * app/modules/fusion/weights/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -- Local imports -------------------------------------------------------------
from app.data.ieee_cis_loader import (
    load_and_engineer,
    time_split,
    M1_FEATURE_COLS,
    M2_FEATURE_COLS,
    M3_FEATURE_COLS,
)

# -- Sklearn metrics -----------------------------------------------------------
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)

# -- Config --------------------------------------------------------------------
ARTIFACT_DIR   = os.path.join("app", "modules", "fusion", "weights")
REPORT_PATH    = os.path.join(ARTIFACT_DIR, "eval_report.json")

APPROVE_THRESH = 0.20
REJECT_THRESH  = 0.75

BANNER = "=" * 80


def _header(title: str) -> None:
    print(f"\n{'-' * 80}\n>>> {title}\n{'-' * 80}")


# -- Helpers -------------------------------------------------------------------

def build_fusion_df(df: pd.DataFrame, m1_clf, m2_clf) -> pd.DataFrame:
    return pd.DataFrame(
        {
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
        },
        index=df.index,
    )


def core_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    pr    = average_precision_score(y_true, y_pred)
    roc   = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    neg_preds = y_pred[y_true == 0]
    thresh5   = np.percentile(neg_preds, 95.0)
    thresh1   = np.percentile(neg_preds, 99.0)
    recall5   = float(np.mean(y_pred[y_true == 1] >= thresh5))
    recall1   = float(np.mean(y_pred[y_true == 1] >= thresh1))
    return {
        "pr_auc":             round(pr, 4),
        "roc_auc":            round(roc, 4),
        "brier_score":        round(brier, 4),
        "recall_at_5pct_fpr": round(recall5, 4),
        "recall_at_1pct_fpr": round(recall1, 4),
    }


def threshold_sweep(y_true: np.ndarray, y_score: np.ndarray) -> list:
    rows = []
    for t in np.arange(0.05, 0.96, 0.05):
        y_pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        fpr  = fp / max(fp + tn, 1)
        rows.append({
            "threshold": round(float(t), 2),
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "fpr":       round(fpr, 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })
    return rows


def calibration_bins(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> list:
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bin":               f"[{lo:.1f},{hi:.1f})",
            "count":             int(mask.sum()),
            "mean_predicted":    round(float(y_score[mask].mean()), 4),
            "actual_fraud_rate": round(float(y_true[mask].mean()), 4),
        })
    return rows


def tier_breakdown(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    approve_mask = y_score < APPROVE_THRESH
    review_mask  = (y_score >= APPROVE_THRESH) & (y_score < REJECT_THRESH)
    reject_mask  = y_score >= REJECT_THRESH

    def _stats(mask):
        n  = int(mask.sum())
        fr = float(y_true[mask].mean()) if n > 0 else 0.0
        return {"count": n, "pct_of_total": round(n / len(y_true) * 100, 2), "fraud_rate": round(fr, 4)}

    return {
        "APPROVE":     _stats(approve_mask),
        "FLAG_REVIEW": _stats(review_mask),
        "REJECT":      _stats(reject_mask),
    }


def adversarial_matrix(test_df, y_test, m1_clf, m2_clf, m3_cal) -> list:
    rng = np.random.default_rng(99)
    results = []

    def _score(df_copy):
        m1_p = m1_clf.predict_proba(df_copy[M1_FEATURE_COLS])[:, 1]
        m2_p = m2_clf.predict_proba(df_copy[M2_FEATURE_COLS])[:, 1]
        X_f  = build_fusion_df(df_copy, m1_clf, m2_clf)
        m3_p = m3_cal.predict_proba(X_f)[:, 1]
        m0_p = df_copy["m0_rule_score"].values
        return {
            "M0 (Rules)":   round(float(average_precision_score(y_test, m0_p)), 4),
            "M1 (Tabular)": round(float(average_precision_score(y_test, m1_p)), 4),
            "M2 (Graph)":   round(float(average_precision_score(y_test, m2_p)), 4),
            "M3 (ARGUS)":   round(float(average_precision_score(y_test, m3_p)), 4),
        }

    r = _score(test_df.copy()); r["Scenario"] = "Clean Baseline"; results.append(r)

    df_10 = test_df.copy()
    df_10.loc[rng.binomial(1, 0.10, len(df_10)).astype(bool), "shared_card1_count"] = 1
    r = _score(df_10); r["Scenario"] = "10% Edge Deletion"; results.append(r)

    df_20 = test_df.copy()
    df_20.loc[rng.binomial(1, 0.20, len(df_20)).astype(bool),
              ["shared_card1_count", "shared_device_count"]] = 1
    r = _score(df_20); r["Scenario"] = "20% Edge Deletion"; results.append(r)

    df_30 = test_df.copy()
    df_30.loc[rng.binomial(1, 0.30, len(df_30)).astype(bool),
              ["shared_card1_count", "shared_device_count", "shared_addr1_count"]] = 1
    r = _score(df_30); r["Scenario"] = "30% Edge Deletion"; results.append(r)

    df_camo = test_df.copy()
    fraud_mask = y_test == 1
    df_camo.loc[fraud_mask, "shared_card1_count"]  = 2
    df_camo.loc[fraud_mask, "shared_device_count"] = 2
    r = _score(df_camo); r["Scenario"] = "Adversarial Camouflage"; results.append(r)

    return results


def shap_importance(m3_cal, X_test_fusion: pd.DataFrame) -> list:
    try:
        import shap
        base_est = m3_cal.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(base_est)
        sample = X_test_fusion.sample(min(1000, len(X_test_fusion)), random_state=42)
        sv = explainer.shap_values(sample[M3_FEATURE_COLS])
        if isinstance(sv, list):
            sv = sv[1]
        mean_abs = np.abs(sv).mean(axis=0)
        return [
            {"feature": f, "mean_abs_shap": round(float(v), 6)}
            for f, v in sorted(zip(M3_FEATURE_COLS, mean_abs), key=lambda x: -x[1])
        ]
    except Exception as e:
        print(f"  [SHAP] Skipped: {e}")
        return []


# -- Main ----------------------------------------------------------------------

def evaluate(subsample: float | None = None, run_shap: bool = False) -> None:
    t0 = time.time()

    print(BANNER)
    print("  PROJECT ARGUS -- Evaluation on Held-Out Test Set")
    print("  M1 (Tabular) | M2 (Graph-Enhanced) | M3 (Calibrated Fusion)")
    print(BANNER)

    # Phase 0: Load weights
    _header("[Phase 0] Loading Serialized Model Weights")
    for fname in ["m1_tabular.joblib", "m2_graph.joblib", "m3_fusion.joblib"]:
        path = os.path.join(ARTIFACT_DIR, fname)
        if not os.path.exists(path):
            print(f"\n  ERROR: {path} not found.")
            print("  Run `python train_argus_models.py` first.\n")
            sys.exit(1)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  [OK] {fname}  ({size_mb:.1f} MB)")

    m1_clf = joblib.load(os.path.join(ARTIFACT_DIR, "m1_tabular.joblib"))
    m2_clf = joblib.load(os.path.join(ARTIFACT_DIR, "m2_graph.joblib"))
    m3_cal = joblib.load(os.path.join(ARTIFACT_DIR, "m3_fusion.joblib"))

    # Phase 1: Load data
    _header("[Phase 1] Loading & Engineering IEEE-CIS Test Split")
    df = load_and_engineer(subsample_frac=subsample)
    _, _, test_df = time_split(df)
    y_test = test_df["isFraud"].values
    print(f"  Test set : {len(test_df):,} rows  |  "
          f"Fraud: {y_test.mean()*100:.2f}%  ({y_test.sum():,} fraud cases)")

    # Phase 2: Predictions
    _header("[Phase 2] Generating Predictions")
    m1_probs  = m1_clf.predict_proba(test_df[M1_FEATURE_COLS])[:, 1]
    m2_probs  = m2_clf.predict_proba(test_df[M2_FEATURE_COLS])[:, 1]
    X_fusion  = build_fusion_df(test_df, m1_clf, m2_clf)
    m3_probs  = m3_cal.predict_proba(X_fusion)[:, 1]
    print("  [OK] M1, M2, M3 predictions generated.")

    # Phase 3: Core metrics
    _header("[Phase 3] Core Metrics")
    m1_m = core_metrics(y_test, m1_probs)
    m2_m = core_metrics(y_test, m2_probs)
    m3_m = core_metrics(y_test, m3_probs)

    col_w = 14
    print(f"  {'Metric':<26} {'M1 Tabular':>{col_w}} {'M2 Graph':>{col_w}} {'M3 ARGUS':>{col_w}}")
    print("  " + "-" * (26 + col_w * 3 + 2))
    for k in m3_m:
        print(f"  {k:<26} {m1_m[k]:>{col_w}.4f} {m2_m[k]:>{col_w}.4f} {m3_m[k]:>{col_w}.4f}")

    graph_lift  = round(m2_m["pr_auc"] - m1_m["pr_auc"], 4)
    fusion_lift = round(m3_m["pr_auc"] - m2_m["pr_auc"], 4)
    total_lift  = round(m3_m["pr_auc"] - m1_m["pr_auc"], 4)
    print(f"\n  Lift  Graph/M1: {graph_lift:+.4f} | Fusion/M2: {fusion_lift:+.4f} | ARGUS/M1: {total_lift:+.4f}")

    # Phase 4: Threshold sweep
    _header("[Phase 4] M3 Threshold Sweep (Precision / Recall / F1 / FPR)")
    sweep = threshold_sweep(y_test, m3_probs)
    try:
        from tabulate import tabulate
        print(tabulate(sweep, headers="keys", tablefmt="github", floatfmt=".4f"))
    except ImportError:
        print(pd.DataFrame(sweep).to_string(index=False))

    # Phase 5: Calibration
    _header("[Phase 5] Calibration Check  (Predicted vs Actual Fraud Rate)")
    calib = calibration_bins(y_test, m3_probs, n_bins=10)
    try:
        from tabulate import tabulate
        print(tabulate(calib, headers="keys", tablefmt="github", floatfmt=".4f"))
    except ImportError:
        print(pd.DataFrame(calib).to_string(index=False))

    # Phase 6: Tier distribution
    _header("[Phase 6] Risk Tier Distribution & Fraud Capture")
    tiers = tier_breakdown(y_test, m3_probs)
    print(f"\n  {'Tier':<14} {'Count':>8} {'% Total':>8} {'Fraud Rate':>12}")
    print("  " + "-" * 46)
    for tier, stats in tiers.items():
        bar = "█" * int(stats["fraud_rate"] * 30)
        print(f"  {tier:<14} {stats['count']:>8,} {stats['pct_of_total']:>7.1f}%  "
              f"{stats['fraud_rate']*100:>9.2f}%  {bar}")

    total_fraud   = int(y_test.sum())
    approve_fraud = int(((y_test == 1) & (m3_probs < APPROVE_THRESH)).sum())
    reject_fraud  = int(((y_test == 1) & (m3_probs >= REJECT_THRESH)).sum())
    review_fraud  = total_fraud - approve_fraud - reject_fraud
    print(f"\n  Auto-approved fraud cases  : {approve_fraud:,} / {total_fraud:,}  "
          f"({approve_fraud/total_fraud*100:.1f}% leakage)")
    print(f"  Auto-rejected fraud cases  : {reject_fraud:,} / {total_fraud:,}  "
          f"({reject_fraud/total_fraud*100:.1f}% auto-blocked)")
    print(f"  Fraud in review queue      : {review_fraud:,} / {total_fraud:,}  "
          f"({review_fraud/total_fraud*100:.1f}%)")

    # Phase 7: Adversarial
    _header("[Phase 7] Adversarial Robustness Benchmark")
    atk = adversarial_matrix(test_df, y_test, m1_clf, m2_clf, m3_cal)
    try:
        from tabulate import tabulate
        print(tabulate(atk, headers="keys", tablefmt="github"))
    except ImportError:
        print(pd.DataFrame(atk).to_string(index=False))

    # Phase 8: SHAP
    shap_results = []
    if run_shap:
        _header("[Phase 8] SHAP Feature Importance (M3 Fusion Layer)")
        shap_results = shap_importance(m3_cal, X_fusion)
        if shap_results:
            print(f"\n  {'Feature':<25} {'Mean |SHAP|':>14}")
            print("  " + "-" * 42)
            max_v = shap_results[0]["mean_abs_shap"]
            for r in shap_results:
                bar = "█" * int(r["mean_abs_shap"] / max_v * 20)
                print(f"  {r['feature']:<25} {r['mean_abs_shap']:>14.6f}  {bar}")
    else:
        print("\n  [Phase 8] SHAP skipped — pass --shap to enable")

    # Phase 9: Confusion matrices
    _header("[Phase 9] Confusion Matrix at Operating Thresholds")
    for thresh, label in [(APPROVE_THRESH, f"APPROVE gate (p >= {APPROVE_THRESH})"),
                          (REJECT_THRESH,  f"REJECT  gate (p >= {REJECT_THRESH})")]:
        y_bin = (m3_probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_bin, labels=[0, 1]).ravel()
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        print(f"\n  {label}")
        print(f"    TP={tp:>7,}  FP={fp:>7,}")
        print(f"    FN={fn:>7,}  TN={tn:>7,}")
        print(f"    Precision={prec:.4f}  Recall={rec:.4f}  Specificity={spec:.4f}")

    # Phase 10: Serialize
    _header("[Phase 10] Saving Evaluation Report")
    report = {
        "generated_at":        time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset":             "IEEE-CIS Fraud Detection (Kaggle) -- Test Split",
        "subsample_frac":      subsample,
        "n_test":              int(len(test_df)),
        "fraud_rate_pct":      round(float(y_test.mean() * 100), 2),
        "metrics": {
            "m1_tabular": m1_m,
            "m2_graph":   m2_m,
            "m3_fusion":  m3_m,
        },
        "lifts": {
            "graph_over_tabular": graph_lift,
            "fusion_over_graph":  fusion_lift,
            "total_over_tabular": total_lift,
        },
        "tier_distribution":   tiers,
        "calibration_bins":    calib,
        "threshold_sweep":     sweep,
        "adversarial_matrix":  atk,
        "shap_importance_m3":  shap_results,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  [OK] Saved -> {REPORT_PATH}")

    elapsed = time.time() - t0
    print(f"\n  Total evaluation time: {elapsed:.1f}s")
    print(BANNER)
    print("  ARGUS EVALUATION COMPLETE")
    print(BANNER)


# -- Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARGUS model evaluation")
    parser.add_argument("--subsample", type=float, default=None,
                        help="Fraction of data for fast dev runs (e.g. 0.2)")
    parser.add_argument("--shap", action="store_true",
                        help="Run SHAP feature importance (requires shap, slower)")
    args = parser.parse_args()
    evaluate(subsample=args.subsample, run_shap=args.shap)
