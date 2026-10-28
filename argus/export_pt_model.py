"""
ARGUS -- PyTorch MLP Fusion Export & Benchmark
================================================
Trains a PyTorch MLP on the same M3 fusion feature vector, saves the model
as  app/modules/fusion/weights/m3_fusion_mlp.pt  and runs a head-to-head
benchmark against the existing LightGBM M3.

Architecture (M3-MLP):
    Input  : 10 features  (same M3 feature vector as LightGBM fusion)
    Hidden : [64 -> 32 -> 16]  with BatchNorm + Dropout(0.3)
    Output : sigmoid -> calibrated probability
    Loss   : BCEWithLogitsLoss  (weighted for class imbalance)
    Optim  : AdamW  lr=1e-3, wd=1e-4
    Epochs : 60  with ReduceLROnPlateau

Usage:
    $env:PYTHONUTF8=1; python export_pt_model.py
    $env:PYTHONUTF8=1; python export_pt_model.py --subsample 0.2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# -- Local imports -------------------------------------------------------------
from app.data.ieee_cis_loader import (
    load_and_engineer,
    time_split,
    M1_FEATURE_COLS,
    M2_FEATURE_COLS,
    M3_FEATURE_COLS,
)
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)

# -- Config --------------------------------------------------------------------
ARTIFACT_DIR = os.path.join("app", "modules", "fusion", "weights")
MODELS_DIR   = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
PT_PATH      = os.path.join(MODELS_DIR, "m3_fusion_mlp.pt")
REPORT_PATH  = os.path.join(ARTIFACT_DIR, "pt_benchmark_report.json")

APPROVE_THRESH = 0.20
REJECT_THRESH  = 0.75
BANNER         = "=" * 80
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _header(title: str) -> None:
    print(f"\n{'-' * 80}\n>>> {title}\n{'-' * 80}")


# -- Build fusion feature matrix -----------------------------------------------

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


# -- PyTorch model architecture ------------------------------------------------

class ArgusMLPFusion(nn.Module):
    """
    Lightweight MLP fusion head for ARGUS.

    Input  : 10-dim M3 feature vector
    Output : scalar logit  (apply sigmoid for probability)
    """

    def __init__(self, in_features: int = 10, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 1),
        )
        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).cpu().numpy()


# -- Metrics -------------------------------------------------------------------

def core_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    pr    = average_precision_score(y_true, y_pred)
    roc   = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    neg   = y_pred[y_true == 0]
    r5    = float(np.mean(y_pred[y_true == 1] >= np.percentile(neg, 95.0)))
    r1    = float(np.mean(y_pred[y_true == 1] >= np.percentile(neg, 99.0)))
    return {
        "pr_auc":             round(pr, 4),
        "roc_auc":            round(roc, 4),
        "brier_score":        round(brier, 4),
        "recall_at_5pct_fpr": round(r5, 4),
        "recall_at_1pct_fpr": round(r1, 4),
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


def adversarial_matrix(test_df, y_test, m1_clf, m2_clf, mlp_model) -> list:
    rng = np.random.default_rng(99)
    results = []

    def _score(df_copy):
        X_f  = build_fusion_df(df_copy, m1_clf, m2_clf)
        x_t  = torch.tensor(X_f[M3_FEATURE_COLS].values, dtype=torch.float32).to(DEVICE)
        mlp_model.eval()
        with torch.no_grad():
            mlp_p = torch.sigmoid(mlp_model(x_t)).cpu().numpy()
        m0_p = df_copy["m0_rule_score"].values
        m1_p = m1_clf.predict_proba(df_copy[M1_FEATURE_COLS])[:, 1]
        m2_p = m2_clf.predict_proba(df_copy[M2_FEATURE_COLS])[:, 1]
        lgbm_m3 = joblib.load(os.path.join(ARTIFACT_DIR, "m3_fusion.joblib"))
        lgbm_p  = lgbm_m3.predict_proba(X_f)[:, 1]
        return {
            "M0 (Rules)":    round(float(average_precision_score(y_test, m0_p)), 4),
            "M1 (Tabular)":  round(float(average_precision_score(y_test, m1_p)), 4),
            "M2 (Graph)":    round(float(average_precision_score(y_test, m2_p)), 4),
            "M3 (LightGBM)": round(float(average_precision_score(y_test, lgbm_p)), 4),
            "M3 (MLP .pt)":  round(float(average_precision_score(y_test, mlp_p)), 4),
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
    df_camo.loc[y_test == 1, "shared_card1_count"]  = 2
    df_camo.loc[y_test == 1, "shared_device_count"] = 2
    r = _score(df_camo); r["Scenario"] = "Adversarial Camouflage"; results.append(r)

    return results


# -- Training ------------------------------------------------------------------

def train_mlp(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    epochs: int = 60,
    batch_size: int = 2048,
) -> ArgusMLPFusion:

    n_pos = y_tr.sum(); n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1) * 0.5], dtype=torch.float32).to(DEVICE)

    model = ArgusMLPFusion(in_features=X_tr.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(DEVICE)
    y_va_t = torch.tensor(y_va, dtype=torch.float32).to(DEVICE)

    ds     = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    best_pr_auc = 0.0
    best_state  = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        # Validation
        model.eval()
        with torch.no_grad():
            va_probs = torch.sigmoid(model(X_va_t)).cpu().numpy()
        val_pr = average_precision_score(y_va, va_probs)
        scheduler.step(val_pr)

        if val_pr > best_pr_auc:
            best_pr_auc = val_pr
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = epoch_loss / len(X_tr)
            lr_now   = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:>3}/{epochs}  loss={avg_loss:.5f}  "
                  f"val_pr_auc={val_pr:.4f}  lr={lr_now:.2e}  "
                  f"{'*best*' if val_pr >= best_pr_auc else ''}")

    model.load_state_dict(best_state)
    print(f"  Best val PR-AUC: {best_pr_auc:.4f}")
    return model


# -- Main ----------------------------------------------------------------------

def main(subsample: float | None = None) -> None:
    t0 = time.time()
    print(BANNER)
    print("  ARGUS -- PyTorch MLP Fusion  |  Export & Head-to-Head Benchmark")
    print(BANNER)

    # Phase 0: check prerequisites
    _header("[Phase 0] Checking Prerequisites")
    for fname in ["m1_tabular.joblib", "m2_graph.joblib", "m3_fusion.joblib"]:
        p = os.path.join(ARTIFACT_DIR, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p} — run train_argus_models.py first")
        print(f"  [OK] {fname}")
    print(f"  Device: {DEVICE}")

    m1_clf = joblib.load(os.path.join(ARTIFACT_DIR, "m1_tabular.joblib"))
    m2_clf = joblib.load(os.path.join(ARTIFACT_DIR, "m2_graph.joblib"))
    m3_lgb = joblib.load(os.path.join(ARTIFACT_DIR, "m3_fusion.joblib"))

    # Phase 1: Load data
    _header("[Phase 1] Loading & Engineering IEEE-CIS Dataset")
    df = load_and_engineer(subsample_frac=subsample)
    train_df, val_df, test_df = time_split(df)
    y_tr = train_df["isFraud"].values
    y_va = val_df["isFraud"].values
    y_te = test_df["isFraud"].values
    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")
    print(f"  Test fraud cases: {y_te.sum():,} / {len(y_te):,} ({y_te.mean()*100:.2f}%)")

    # Phase 2: Build M3 feature matrices
    _header("[Phase 2] Building M3 Fusion Feature Matrices")
    X_tr_f = build_fusion_df(train_df, m1_clf, m2_clf)[M3_FEATURE_COLS].values.astype(np.float32)
    X_va_f = build_fusion_df(val_df,   m1_clf, m2_clf)[M3_FEATURE_COLS].values.astype(np.float32)
    X_te_f = build_fusion_df(test_df,  m1_clf, m2_clf)[M3_FEATURE_COLS].values.astype(np.float32)
    print(f"  Feature matrix shape: {X_tr_f.shape}  (10 M3 features)")

    # Phase 3: Train PyTorch MLP
    _header("[Phase 3] Training PyTorch MLP Fusion (ArgusMLPFusion)")
    print(f"  Architecture: Linear(10->64) BN ReLU Drop | Linear(64->32) BN ReLU Drop | "
          f"Linear(32->16) BN ReLU | Linear(16->1)")
    mlp = train_mlp(X_tr_f, y_tr, X_va_f, y_va, epochs=60)

    # Phase 4: Save .pt
    _header("[Phase 4] Saving PyTorch Model (.pt)")
    checkpoint = {
        "model_state_dict": mlp.state_dict(),
        "architecture":     "ArgusMLPFusion",
        "in_features":      10,
        "feature_names":    M3_FEATURE_COLS,
        "trained_on":       "IEEE-CIS Fraud Detection (Kaggle) — Train split 70%",
        "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(checkpoint, PT_PATH)
    size_mb = os.path.getsize(PT_PATH) / 1e6
    print(f"  [OK] Saved -> {PT_PATH}  ({size_mb:.2f} MB)")

    # Phase 5: Test-set predictions
    _header("[Phase 5] Generating Test-Set Predictions")
    x_te_t    = torch.tensor(X_te_f, dtype=torch.float32).to(DEVICE)
    mlp_probs = mlp.predict_proba(x_te_t)
    lgb_probs = m3_lgb.predict_proba(
        pd.DataFrame(X_te_f, columns=M3_FEATURE_COLS)
    )[:, 1]
    m1_probs  = m1_clf.predict_proba(test_df[M1_FEATURE_COLS])[:, 1]
    m2_probs  = m2_clf.predict_proba(test_df[M2_FEATURE_COLS])[:, 1]
    print("  [OK] Predictions ready.")

    # Phase 6: Core metrics comparison
    _header("[Phase 6] Core Metrics — M3 LightGBM  vs  M3 MLP (.pt)")
    m1_m   = core_metrics(y_te, m1_probs)
    m2_m   = core_metrics(y_te, m2_probs)
    lgb_m  = core_metrics(y_te, lgb_probs)
    mlp_m  = core_metrics(y_te, mlp_probs)

    col_w = 14
    print(f"  {'Metric':<26} {'M1 Tabular':>{col_w}} {'M2 Graph':>{col_w}} "
          f"{'M3 LightGBM':>{col_w}} {'M3 MLP .pt':>{col_w}}")
    print("  " + "-" * (26 + col_w * 4 + 4))
    for k in mlp_m:
        print(f"  {k:<26} {m1_m[k]:>{col_w}.4f} {m2_m[k]:>{col_w}.4f} "
              f"{lgb_m[k]:>{col_w}.4f} {mlp_m[k]:>{col_w}.4f}")

    delta_pr  = round(mlp_m["pr_auc"]  - lgb_m["pr_auc"],  4)
    delta_roc = round(mlp_m["roc_auc"] - lgb_m["roc_auc"], 4)
    print(f"\n  MLP vs LightGBM delta  PR-AUC: {delta_pr:+.4f}  ROC-AUC: {delta_roc:+.4f}")

    # Phase 7: Threshold sweep (MLP)
    _header("[Phase 7] M3 MLP Threshold Sweep")
    sweep = threshold_sweep(y_te, mlp_probs)
    try:
        from tabulate import tabulate
        print(tabulate(sweep, headers="keys", tablefmt="github", floatfmt=".4f"))
    except ImportError:
        print(pd.DataFrame(sweep).to_string(index=False))

    # Phase 8: Tier distribution (MLP)
    _header("[Phase 8] Risk Tier Distribution (MLP)")
    approve_m = mlp_probs < APPROVE_THRESH
    review_m  = (mlp_probs >= APPROVE_THRESH) & (mlp_probs < REJECT_THRESH)
    reject_m  = mlp_probs >= REJECT_THRESH
    total_fr  = y_te.sum()

    print(f"\n  {'Tier':<14} {'Count':>8} {'% Total':>8} {'Fraud Rate':>12}")
    print("  " + "-" * 46)
    for label, mask in [("APPROVE", approve_m), ("FLAG_REVIEW", review_m), ("REJECT", reject_m)]:
        n  = mask.sum()
        fr = y_te[mask].mean() if n > 0 else 0.0
        bar = "█" * int(fr * 30)
        print(f"  {label:<14} {n:>8,} {n/len(y_te)*100:>7.1f}%  {fr*100:>9.2f}%  {bar}")

    ap_fr = int(((y_te == 1) & approve_m).sum())
    rj_fr = int(((y_te == 1) & reject_m).sum())
    rv_fr = int(total_fr) - ap_fr - rj_fr
    print(f"\n  Auto-approved fraud  : {ap_fr:,} / {int(total_fr):,}  ({ap_fr/total_fr*100:.1f}% leakage)")
    print(f"  Auto-rejected fraud  : {rj_fr:,} / {int(total_fr):,}  ({rj_fr/total_fr*100:.1f}% auto-blocked)")
    print(f"  Fraud in review queue: {rv_fr:,} / {int(total_fr):,}  ({rv_fr/total_fr*100:.1f}%)")

    # Phase 9: Adversarial matrix
    _header("[Phase 9] Adversarial Robustness — LightGBM vs MLP (.pt)")
    atk = adversarial_matrix(test_df, y_te, m1_clf, m2_clf, mlp)
    try:
        from tabulate import tabulate
        print(tabulate(atk, headers="keys", tablefmt="github"))
    except ImportError:
        print(pd.DataFrame(atk).to_string(index=False))

    # Phase 10: Confusion matrices
    _header("[Phase 10] Confusion Matrices — MLP at Operating Thresholds")
    for thresh, label in [(APPROVE_THRESH, f"APPROVE gate (p >= {APPROVE_THRESH})"),
                          (REJECT_THRESH,  f"REJECT  gate (p >= {REJECT_THRESH})")]:
        y_bin = (mlp_probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, y_bin, labels=[0, 1]).ravel()
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        print(f"\n  {label}")
        print(f"    TP={tp:>7,}  FP={fp:>7,}")
        print(f"    FN={fn:>7,}  TN={tn:>7,}")
        print(f"    Precision={prec:.4f}  Recall={rec:.4f}  Specificity={spec:.4f}")

    # Phase 11: Save report
    _header("[Phase 11] Saving PT Benchmark Report")
    report = {
        "generated_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "pt_model_path":  PT_PATH,
        "device":         str(DEVICE),
        "architecture":   "ArgusMLPFusion  Linear(10->64->32->16->1) BN+Dropout(0.3)",
        "n_parameters":   sum(p.numel() for p in mlp.parameters()),
        "n_test":         int(len(test_df)),
        "fraud_rate_pct": round(float(y_te.mean() * 100), 2),
        "metrics": {
            "m1_tabular":   m1_m,
            "m2_graph":     m2_m,
            "m3_lightgbm":  lgb_m,
            "m3_mlp_pt":    mlp_m,
        },
        "delta_mlp_vs_lgbm": {
            "pr_auc":  delta_pr,
            "roc_auc": delta_roc,
        },
        "adversarial_matrix": atk,
        "threshold_sweep_mlp": sweep,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  [OK] Saved -> {REPORT_PATH}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(BANNER)
    print("  ARGUS PyTorch MLP EXPORT & BENCHMARK COMPLETE")
    print(BANNER)


# -- Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=float, default=None)
    args = parser.parse_args()
    main(subsample=args.subsample)
