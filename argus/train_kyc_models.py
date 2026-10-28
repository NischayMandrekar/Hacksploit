"""
ARGUS KYC Training Pipeline  —  train_kyc_models.py
======================================================
Trains four models on a synthetic Indian e-KYC benchmark that matches the
Hacksploit 2026 PS3 domain: Aadhaar/PAN fraud, burner-SIM rings, ELA tampering,
face-cosine biometric distance, and velocity bursts.

Models trained:
  M1  — Tabular LightGBM   (5 behavioral/demographic features)
  M2  — Graph LightGBM     (M1 features + 4 ego-network topology features)
  M3  — Calibrated Fusion  (10-dim KYC evidence vector, CalibratedClassifierCV isotonic)
  MLP — PyTorch Fusion     (same 10-dim vector, ArgusMLPFusion architecture)

All four models saved to  models/  as  .pt  files.

Usage:
    $env:PYTHONUTF8=1; python train_kyc_models.py
    $env:PYTHONUTF8=1; python train_kyc_models.py --samples 20000
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    brier_score_loss, confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR   = "models"
WEIGHTS_DIR  = os.path.join("app", "modules", "fusion", "weights")
REPORT_PATH  = os.path.join(WEIGHTS_DIR, "kyc_benchmark_report.json")
BANNER       = "=" * 80
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

M1_COLS = ["customer_age", "income_lakhs", "account_type_risk", "velocity_1h", "velocity_24h"]
M2_COLS = M1_COLS + ["shared_phone_degree", "shared_device_degree", "ego_density", "cycle_count"]
M3_COLS = [
    "m1_prob", "m2_prob", "m0_rule_score",
    "ocr_contradiction", "id_checksum_valid",
    "tampering_ela_score", "face_cosine_dist",
    "is_doc_missing", "is_selfie_missing", "is_graph_isolated",
]

APPROVE_T = 0.20
REJECT_T  = 0.75

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def _header(title: str) -> None:
    print(f"\n{'-' * 80}\n>>> {title}\n{'-' * 80}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Synthetic Indian e-KYC Data Generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_indian_kyc_data(n: int = 12_000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic Indian e-KYC application dataset.

    Fraud signatures modelled:
      - Jamtara/Mewat SIM-box pattern: elevated shared_phone_degree
      - Device farms: elevated shared_device_degree
      - Synthetic identity: ocr_contradiction > 0.3 + id_checksum_valid == 0
      - Document splicing: high tampering_ela_score
      - Impostor selfie: large face_cosine_dist
      - Adversarial evasion: is_doc_missing / is_selfie_missing == 1
      - Application velocity burst: velocity_1h >= 4
    """
    rng = np.random.default_rng(seed)

    # Ground truth: 6% fraud prevalence (representative of high-risk fintech)
    y = rng.binomial(1, 0.06, n)

    # ── Tabular demographics & velocity ──────────────────────────────────────
    customer_age      = rng.integers(18, 68, n)
    income_lakhs      = rng.lognormal(mean=1.5, sigma=0.6, size=n)
    account_type_risk = rng.choice([0.1, 0.3, 0.7], size=n, p=[0.60, 0.30, 0.10])
    velocity_1h       = rng.poisson(0.2, n) + y * rng.poisson(3.8, n)
    velocity_24h      = velocity_1h + rng.poisson(1.0, n) + y * rng.poisson(5.0, n)

    # ── Relational graph topology (ego-network, 2-hop) ────────────────────────
    shared_phone_degree  = 1 + rng.poisson(0.15, n) + y * rng.geometric(0.35, n)
    shared_device_degree = 1 + rng.poisson(0.08, n) + y * rng.geometric(0.40, n)
    ego_density = np.clip(
        rng.beta(1.2, 12, n) + y * rng.beta(3, 4, n), 0.0, 1.0
    )
    cycle_count = rng.poisson(0.05, n) + y * rng.poisson(1.8, n)

    # ── Document forensics & OCR (Aadhaar / PAN) ─────────────────────────────
    ocr_name_sim = np.clip(
        rng.beta(25, 2, n) - y * rng.beta(4, 3, n) * 0.65, 0.0, 1.0
    )
    ocr_contradiction = 1.0 - ocr_name_sim

    # Verhoeff / PAN checksum: legitimate almost always valid
    id_checksum_valid = np.where(
        y == 1,
        rng.choice([0, 1], size=n, p=[0.45, 0.55]),
        rng.choice([0, 1], size=n, p=[0.002, 0.998]),
    )

    # ── Media forensics & biometrics ─────────────────────────────────────────
    tampering_ela_score = np.clip(
        rng.beta(1.5, 18, n) + y * rng.beta(5, 3, n) * 0.75, 0.0, 1.0
    )
    # face cosine: clean ~0.25, impostor swap ~0.65
    face_cosine_dist = np.clip(
        rng.normal(0.25, 0.08, n) + y * rng.normal(0.40, 0.12, n), 0.0, 1.0
    )

    # ── Missing-evidence flags (adversarial evasion / sensor dropout) ─────────
    is_doc_missing = np.where(
        y == 1, rng.binomial(1, 0.08, n), rng.binomial(1, 0.015, n)
    )
    is_selfie_missing = np.where(
        y == 1, rng.binomial(1, 0.15, n), rng.binomial(1, 0.03, n)
    )
    is_graph_isolated = ((shared_phone_degree == 1) & (shared_device_degree == 1)).astype(int)

    # ── M0 deterministic rule score ───────────────────────────────────────────
    m0_triggers = (
        (shared_phone_degree  >= 3).astype(int) +
        (shared_device_degree >= 3).astype(int) +
        (ocr_contradiction     > 0.30).astype(int) +
        (id_checksum_valid    == 0).astype(int) +
        (velocity_1h          >= 4).astype(int)
    )
    m0_rule_score = np.clip(m0_triggers * 0.25, 0.0, 1.0)

    return pd.DataFrame({
        # tabular
        "customer_age":       customer_age,
        "income_lakhs":       income_lakhs,
        "account_type_risk":  account_type_risk,
        "velocity_1h":        velocity_1h,
        "velocity_24h":       velocity_24h,
        # graph
        "shared_phone_degree":  shared_phone_degree,
        "shared_device_degree": shared_device_degree,
        "ego_density":          ego_density,
        "cycle_count":          cycle_count,
        # forensics
        "ocr_contradiction":    ocr_contradiction,
        "id_checksum_valid":    id_checksum_valid,
        "tampering_ela_score":  tampering_ela_score,
        "face_cosine_dist":     face_cosine_dist,
        # missingness / flags
        "is_doc_missing":    is_doc_missing,
        "is_selfie_missing": is_selfie_missing,
        "is_graph_isolated": is_graph_isolated,
        "m0_rule_score":     m0_rule_score,
        # label
        "isFraud": y,
    })


# ══════════════════════════════════════════════════════════════════════════════
# 2. PyTorch MLP (same ArgusMLPFusion architecture)
# ══════════════════════════════════════════════════════════════════════════════

class ArgusMLPFusion(nn.Module):
    """10-dim KYC evidence vector → scalar fraud logit."""

    def __init__(self, in_features: int = 10, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16), nn.ReLU(),

            nn.Linear(16, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)).cpu().numpy()


def train_mlp(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    epochs: int = 80, batch_size: int = 512,
) -> ArgusMLPFusion:
    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
    pos_w  = torch.tensor([n_neg / max(n_pos, 1) * 0.5], dtype=torch.float32).to(DEVICE)

    model     = ArgusMLPFusion(in_features=X_tr.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6
    )

    ds     = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(DEVICE)

    best_pr, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            nn.BCEWithLogitsLoss(pos_weight=pos_w)(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            va_p = torch.sigmoid(model(X_va_t)).cpu().numpy()
        vpr = average_precision_score(y_va, va_p)
        scheduler.step(vpr)
        if vpr > best_pr:
            best_pr = vpr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:>3}/{epochs}  val_pr_auc={vpr:.4f}  lr={lr:.2e}"
                  + ("  *best*" if vpr >= best_pr else ""))

    model.load_state_dict(best_state)
    print(f"  Best val PR-AUC: {best_pr:.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 3. Metrics helpers
# ══════════════════════════════════════════════════════════════════════════════

def core_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pr    = average_precision_score(y, p)
    roc   = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    neg   = p[y == 0]
    r5    = float(np.mean(p[y == 1] >= np.percentile(neg, 95.0)))
    r1    = float(np.mean(p[y == 1] >= np.percentile(neg, 99.0)))
    return {
        "pr_auc":             round(pr,    4),
        "roc_auc":            round(roc,   4),
        "brier_score":        round(brier, 4),
        "recall_at_5pct_fpr": round(r5,   4),
        "recall_at_1pct_fpr": round(r1,   4),
    }


def tier_report(y: np.ndarray, p: np.ndarray, model_name: str) -> None:
    am = p < APPROVE_T
    rm = (p >= APPROVE_T) & (p < REJECT_T)
    jm = p >= REJECT_T
    total_fr = y.sum()
    print(f"\n  Risk tiers for {model_name}:")
    print(f"  {'Tier':<14} {'Count':>8} {'%Total':>7} {'FraudRate':>10}")
    print("  " + "-" * 42)
    for label, mask in [("APPROVE", am), ("FLAG_REVIEW", rm), ("REJECT", jm)]:
        n  = mask.sum()
        fr = y[mask].mean() * 100 if n > 0 else 0.0
        print(f"  {label:<14} {n:>8,} {n/len(y)*100:>6.1f}%  {fr:>8.2f}%")
    ap_fr = int(((y == 1) & am).sum())
    rj_fr = int(((y == 1) & jm).sum())
    print(f"\n  Auto-rejected fraud : {rj_fr:,} / {int(total_fr):,}  "
          f"({rj_fr/total_fr*100:.1f}% blocked)")
    print(f"  Approved fraud leak : {ap_fr:,} / {int(total_fr):,}  "
          f"({ap_fr/total_fr*100:.1f}% slip-through)")


def adversarial_test(
    test_df: pd.DataFrame, y_te: np.ndarray,
    m1_clf, m2_clf, m3_lgb, mlp_model,
    X_te_fuse: pd.DataFrame,
) -> list[dict]:
    rng = np.random.default_rng(99)
    results = []

    def _row(df_c: pd.DataFrame, label: str) -> dict:
        m1p = m1_clf.predict_proba(df_c[M1_COLS])[:, 1]
        m2p = m2_clf.predict_proba(df_c[M2_COLS])[:, 1]
        xf  = _fuse(df_c, m1p, m2p)
        m3p = m3_lgb.predict_proba(xf)[:, 1]
        xt  = torch.tensor(xf[M3_COLS].values, dtype=torch.float32).to(DEVICE)
        mlp_model.eval()
        with torch.no_grad():
            mlpp = torch.sigmoid(mlp_model(xt)).cpu().numpy()
        return {
            "Scenario":       label,
            "M1 (Tabular)":  round(float(average_precision_score(y_te, m1p)),  4),
            "M2 (Graph)":    round(float(average_precision_score(y_te, m2p)),  4),
            "M3 (LightGBM)": round(float(average_precision_score(y_te, m3p)),  4),
            "M3 (MLP .pt)":  round(float(average_precision_score(y_te, mlpp)), 4),
        }

    results.append(_row(test_df.copy(), "Clean Baseline"))

    df10 = test_df.copy()
    df10.loc[rng.binomial(1, 0.10, len(df10)).astype(bool), "shared_phone_degree"] = 1
    results.append(_row(df10, "10% Edge Deletion"))

    df20 = test_df.copy()
    df20.loc[rng.binomial(1, 0.20, len(df20)).astype(bool),
             ["shared_phone_degree", "shared_device_degree"]] = 1
    results.append(_row(df20, "20% Edge Deletion"))

    df30 = test_df.copy()
    df30.loc[rng.binomial(1, 0.30, len(df30)).astype(bool),
             ["shared_phone_degree", "shared_device_degree", "ego_density"]] = 0
    results.append(_row(df30, "30% Edge Deletion"))

    camo = test_df.copy()
    camo.loc[y_te == 1, "shared_phone_degree"]  = 2
    camo.loc[y_te == 1, "shared_device_degree"] = 2
    camo.loc[y_te == 1, "ego_density"]          = camo.loc[y_te == 1, "ego_density"] * 0.4
    results.append(_row(camo, "Adversarial Camouflage"))

    return results


def _fuse(df: pd.DataFrame, m1p: np.ndarray, m2p: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "m1_prob":             m1p,
        "m2_prob":             m2p,
        "m0_rule_score":       df["m0_rule_score"].values,
        "ocr_contradiction":   df["ocr_contradiction"].values,
        "id_checksum_valid":   df["id_checksum_valid"].values,
        "tampering_ela_score": df["tampering_ela_score"].values,
        "face_cosine_dist":    df["face_cosine_dist"].values,
        "is_doc_missing":      df["is_doc_missing"].values,
        "is_selfie_missing":   df["is_selfie_missing"].values,
        "is_graph_isolated":   df["is_graph_isolated"].values,
    }, index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main(n_samples: int = 12_000) -> None:
    t0 = time.time()
    print(BANNER)
    print("  ARGUS — KYC Domain Training Pipeline")
    print("  Dataset: Synthetic Indian e-KYC Benchmark")
    print(f"  Samples: {n_samples:,}  |  Device: {DEVICE}")
    print(BANNER)

    # ── Phase 1: Generate data ────────────────────────────────────────────────
    _header("[Phase 1] Generating Synthetic Indian e-KYC Dataset")
    df = generate_indian_kyc_data(n_samples)
    fraud_rate = df["isFraud"].mean() * 100
    print(f"  Shape: {df.shape}  |  Fraud rate: {fraud_rate:.2f}%")

    # Time-ordered 70 / 15 / 15 split (no shuffle to prevent leakage)
    n_tr = int(len(df) * 0.70)
    n_va = int(len(df) * 0.15)
    train_df = df.iloc[:n_tr].copy()
    val_df   = df.iloc[n_tr:n_tr + n_va].copy()
    test_df  = df.iloc[n_tr + n_va:].copy()

    y_tr = train_df["isFraud"].values
    y_va = val_df["isFraud"].values
    y_te = test_df["isFraud"].values

    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")
    print(f"  Test fraud cases: {y_te.sum():,} ({y_te.mean()*100:.2f}%)")

    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    # ── Phase 2: Train M1 ─────────────────────────────────────────────────────
    _header("[Phase 2] Training M1 — Tabular LightGBM")
    print(f"  Features ({len(M1_COLS)}): {M1_COLS}")
    m1_clf = lgb.LGBMClassifier(
        objective="binary", boosting_type="gbdt",
        learning_rate=0.03, num_leaves=31, max_depth=5,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, n_estimators=300, random_state=42, verbosity=-1,
    )
    m1_clf.fit(
        train_df[M1_COLS], y_tr,
        eval_set=[(val_df[M1_COLS], y_va)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(period=-1)],
    )
    m1_te = m1_clf.predict_proba(test_df[M1_COLS])[:, 1]
    m1_m  = core_metrics(y_te, m1_te)
    print(f"  Test PR-AUC={m1_m['pr_auc']:.4f}  ROC-AUC={m1_m['roc_auc']:.4f}  Brier={m1_m['brier_score']:.4f}")

    # ── Phase 3: Train M2 ─────────────────────────────────────────────────────
    _header("[Phase 3] Training M2 — Graph-Enhanced LightGBM")
    print(f"  Features ({len(M2_COLS)}): {M2_COLS}")
    m2_clf = lgb.LGBMClassifier(
        objective="binary", boosting_type="gbdt",
        learning_rate=0.03, num_leaves=31, max_depth=6,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, n_estimators=300, random_state=42, verbosity=-1,
    )
    m2_clf.fit(
        train_df[M2_COLS], y_tr,
        eval_set=[(val_df[M2_COLS], y_va)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(period=-1)],
    )
    m2_te = m2_clf.predict_proba(test_df[M2_COLS])[:, 1]
    m2_m  = core_metrics(y_te, m2_te)
    print(f"  Test PR-AUC={m2_m['pr_auc']:.4f}  ROC-AUC={m2_m['roc_auc']:.4f}  "
          f"Graph lift: {m2_m['pr_auc'] - m1_m['pr_auc']:+.4f}")

    # ── Phase 4: Build M3 fusion feature matrices ─────────────────────────────
    _header("[Phase 4] Building M3 Fusion Feature Matrices (10-dim KYC vector)")
    m1_tr = m1_clf.predict_proba(train_df[M1_COLS])[:, 1]
    m2_tr = m2_clf.predict_proba(train_df[M2_COLS])[:, 1]
    m1_va = m1_clf.predict_proba(val_df[M1_COLS])[:, 1]
    m2_va = m2_clf.predict_proba(val_df[M2_COLS])[:, 1]

    X_tr_f = _fuse(train_df, m1_tr, m2_tr)
    X_va_f = _fuse(val_df,   m1_va, m2_va)
    X_te_f = _fuse(test_df,  m1_te, m2_te)
    print(f"  M3 feature vector: {M3_COLS}")

    # ── Phase 5: Train M3 LightGBM (calibrated) ───────────────────────────────
    _header("[Phase 5] Training M3 — Calibrated LightGBM Fusion")
    base_m3 = lgb.LGBMClassifier(
        objective="binary", n_estimators=200,
        learning_rate=0.04, max_depth=4, num_leaves=15,
        scale_pos_weight=spw * 0.5, random_state=42, verbosity=-1,
    )
    m3_lgb = CalibratedClassifierCV(estimator=base_m3, method="isotonic", cv=3)
    m3_lgb.fit(X_tr_f[M3_COLS], y_tr)
    m3_te = m3_lgb.predict_proba(X_te_f[M3_COLS])[:, 1]
    m3_m  = core_metrics(y_te, m3_te)
    print(f"  Test PR-AUC={m3_m['pr_auc']:.4f}  ROC-AUC={m3_m['roc_auc']:.4f}  "
          f"Brier={m3_m['brier_score']:.4f}  "
          f"Total lift: {m3_m['pr_auc'] - m1_m['pr_auc']:+.4f}")
    tier_report(y_te, m3_te, "M3 LightGBM")

    # ── Phase 6: Train M3 PyTorch MLP ─────────────────────────────────────────
    _header("[Phase 6] Training M3-MLP — PyTorch ArgusMLPFusion (80 epochs)")
    X_tr_np = X_tr_f[M3_COLS].values.astype(np.float32)
    X_va_np = X_va_f[M3_COLS].values.astype(np.float32)
    X_te_np = X_te_f[M3_COLS].values.astype(np.float32)

    mlp = train_mlp(X_tr_np, y_tr, X_va_np, y_va, epochs=80)
    x_te_t  = torch.tensor(X_te_np, dtype=torch.float32).to(DEVICE)
    mlp_te  = mlp.predict_proba(x_te_t)
    mlp_m   = core_metrics(y_te, mlp_te)
    print(f"\n  Test PR-AUC={mlp_m['pr_auc']:.4f}  ROC-AUC={mlp_m['roc_auc']:.4f}  "
          f"Brier={mlp_m['brier_score']:.4f}")
    tier_report(y_te, mlp_te, "M3 MLP (.pt)")

    # ── Phase 7: Benchmark table ───────────────────────────────────────────────
    _header("[Phase 7] Core Metrics — Full Comparison")
    col = 14
    print(f"  {'Metric':<26} {'M1 Tabular':>{col}} {'M2 Graph':>{col}} "
          f"{'M3 LightGBM':>{col}} {'M3 MLP .pt':>{col}}")
    print("  " + "-" * (26 + col * 4 + 4))
    for k in mlp_m:
        print(f"  {k:<26} {m1_m[k]:>{col}.4f} {m2_m[k]:>{col}.4f} "
              f"{m3_m[k]:>{col}.4f} {mlp_m[k]:>{col}.4f}")
    print(f"\n  MLP vs LGBM  PR-AUC delta: {mlp_m['pr_auc'] - m3_m['pr_auc']:+.4f}  "
          f"ROC-AUC delta: {mlp_m['roc_auc'] - m3_m['roc_auc']:+.4f}")

    # ── Phase 8: Adversarial robustness ───────────────────────────────────────
    _header("[Phase 8] Adversarial Robustness Matrix")
    atk = adversarial_test(test_df, y_te, m1_clf, m2_clf, m3_lgb, mlp, X_te_f)
    try:
        from tabulate import tabulate
        print(tabulate(atk, headers="keys", tablefmt="github"))
    except ImportError:
        print(pd.DataFrame(atk).to_string(index=False))

    # ── Phase 9: Save all models to models/ as .pt ────────────────────────────
    _header("[Phase 9] Saving All Models → models/*.pt")

    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    # M1
    torch.save({
        "model": m1_clf, "format": "lightgbm/sklearn",
        "feature_names": M1_COLS, "domain": "Indian e-KYC",
        "metrics": m1_m, "timestamp": ts,
    }, os.path.join(MODELS_DIR, "m1_kyc_tabular.pt"))
    print(f"  [OK] models/m1_kyc_tabular.pt  "
          f"({os.path.getsize(os.path.join(MODELS_DIR,'m1_kyc_tabular.pt'))/1e3:.0f} KB)")

    # M2
    torch.save({
        "model": m2_clf, "format": "lightgbm/sklearn",
        "feature_names": M2_COLS, "domain": "Indian e-KYC",
        "metrics": m2_m, "timestamp": ts,
    }, os.path.join(MODELS_DIR, "m2_kyc_graph.pt"))
    print(f"  [OK] models/m2_kyc_graph.pt  "
          f"({os.path.getsize(os.path.join(MODELS_DIR,'m2_kyc_graph.pt'))/1e3:.0f} KB)")

    # M3 LightGBM
    torch.save({
        "model": m3_lgb, "format": "lightgbm/sklearn/calibrated",
        "feature_names": M3_COLS, "domain": "Indian e-KYC",
        "metrics": m3_m, "timestamp": ts,
    }, os.path.join(MODELS_DIR, "m3_kyc_fusion.pt"))
    print(f"  [OK] models/m3_kyc_fusion.pt  "
          f"({os.path.getsize(os.path.join(MODELS_DIR,'m3_kyc_fusion.pt'))/1e6:.2f} MB)")

    # M3 MLP
    torch.save({
        "model_state_dict": mlp.state_dict(),
        "architecture": "ArgusMLPFusion",
        "in_features": 10, "feature_names": M3_COLS,
        "domain": "Indian e-KYC",
        "metrics": mlp_m, "timestamp": ts,
    }, os.path.join(MODELS_DIR, "m3_kyc_mlp.pt"))
    print(f"  [OK] models/m3_kyc_mlp.pt  "
          f"({os.path.getsize(os.path.join(MODELS_DIR,'m3_kyc_mlp.pt'))/1e6:.2f} MB)")

    # Also dump LightGBM .joblib for backwards compat
    joblib.dump(m1_clf, os.path.join(WEIGHTS_DIR, "m1_kyc_tabular.joblib"))
    joblib.dump(m2_clf, os.path.join(WEIGHTS_DIR, "m2_kyc_graph.joblib"))
    joblib.dump(m3_lgb, os.path.join(WEIGHTS_DIR, "m3_kyc_fusion.joblib"))
    print(f"  [OK] .joblib copies in {WEIGHTS_DIR}/")

    # ── Phase 10: Save benchmark report ───────────────────────────────────────
    _header("[Phase 10] Saving KYC Benchmark Report")
    report = {
        "generated_at":   ts,
        "domain":         "Indian e-KYC Synthetic Benchmark",
        "n_train":        int(len(train_df)),
        "n_val":          int(len(val_df)),
        "n_test":         int(len(test_df)),
        "fraud_rate_pct": round(float(y_te.mean() * 100), 2),
        "feature_sets":   {"m1": M1_COLS, "m2": M2_COLS, "m3": M3_COLS},
        "metrics": {
            "m1_tabular":   m1_m,
            "m2_graph":     m2_m,
            "m3_lgbm":      m3_m,
            "m3_mlp_pt":    mlp_m,
        },
        "lifts": {
            "graph_over_tabular":  round(m2_m["pr_auc"]  - m1_m["pr_auc"],  4),
            "fusion_over_graph":   round(m3_m["pr_auc"]  - m2_m["pr_auc"],  4),
            "total_lift_m3_lgbm":  round(m3_m["pr_auc"]  - m1_m["pr_auc"],  4),
            "total_lift_m3_mlp":   round(mlp_m["pr_auc"] - m1_m["pr_auc"],  4),
        },
        "adversarial_matrix": atk,
        "models": {
            "m1_kyc_tabular.pt": "LightGBM (5 tabular features)",
            "m2_kyc_graph.pt":   "LightGBM (9 tabular+graph features)",
            "m3_kyc_fusion.pt":  "CalibratedClassifierCV(LightGBM, isotonic, cv=3)",
            "m3_kyc_mlp.pt":     "ArgusMLPFusion PyTorch Linear(10→64→32→16→1)",
        },
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  [OK] {REPORT_PATH}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(BANNER)
    print("  ARGUS KYC TRAINING COMPLETE")
    print(f"  models/ now contains:")
    for f in sorted(os.listdir(MODELS_DIR)):
        sz = os.path.getsize(os.path.join(MODELS_DIR, f))
        print(f"    {f:<30}  {sz/1e3:>7.1f} KB")
    print(BANNER)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=12_000,
                    help="Number of synthetic KYC records to generate (default 12000)")
    args = ap.parse_args()
    main(n_samples=args.samples)
