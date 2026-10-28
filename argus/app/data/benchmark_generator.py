"""
ARGUS Synthetic Indian e-KYC Benchmark Dataset Generator
=========================================================
Generates 12,000 reproducible synthetic applicant records that simulate:
  - Ground truth synthetic identity fraud rings (6% prevalence)
  - Burner SIM sharing and device farm patterns (Jamtara / Mewat syndicates)
  - OCR name transliteration drift (Laxmi / Lakshmi, Mohd / Mohammad)
  - Aadhaar Verhoeff checksum failures in unvalidated numbers
  - Error Level Analysis (ELA) tampering scores
  - Biometric face cosine distance distributions
  - Explicit missing-signal flags (adversarial sensor dropout)
  - M0 deterministic rule scores (pre-computed for fusion)

The distribution is calibrated against:
  - BAF (NeurIPS 2022) tabular feature ranges
  - Neo4j Synthetic Identity Benchmark topological degree distributions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Default generation parameters
# ---------------------------------------------------------------------------
DEFAULT_N_SAMPLES    = 12_000
DEFAULT_FRAUD_RATE   = 0.06     # 6% — representative of high-risk fintech intake
DEFAULT_SEED         = 42

# Train / Val / Test proportions
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_indian_kyc_benchmark(
    n_samples: int = DEFAULT_N_SAMPLES,
    fraud_rate: float = DEFAULT_FRAUD_RATE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic Indian e-KYC benchmark DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        customer_age, income_lakhs, account_type_risk,
        velocity_1h, velocity_24h,
        shared_phone_degree, shared_device_degree,
        ego_density, cycle_count,
        ocr_contradiction, id_checksum_valid,
        tampering_ela_score, face_cosine_dist,
        is_doc_missing, is_selfie_missing, is_graph_isolated,
        m0_rule_score,
        target_fraud  (label)
    """
    rng = np.random.default_rng(seed)

    # --- Ground truth labels ---
    y = rng.binomial(1, fraud_rate, n_samples)

    # ── 1. Tabular Demographics & Behavioral Velocity ──────────────────────
    customer_age      = rng.integers(18, 68, n_samples)
    income_lakhs      = rng.lognormal(mean=1.5, sigma=0.6, size=n_samples)
    account_type_risk = rng.choice([0.1, 0.3, 0.7], size=n_samples, p=[0.6, 0.3, 0.1])

    # Fraudsters submit many applications in a short window (velocity burst)
    velocity_1h  = (rng.poisson(0.2, n_samples)
                    + y * rng.poisson(3.8, n_samples))
    velocity_24h = (velocity_1h
                    + rng.poisson(1.0, n_samples)
                    + y * rng.poisson(5.0, n_samples))

    # ── 2. Relational Graph Topology (Burner SIMs / Device Farms) ──────────
    shared_phone_degree  = (1
                            + rng.poisson(0.15, n_samples)
                            + y * rng.geometric(0.35, n_samples))
    shared_device_degree = (1
                            + rng.poisson(0.08, n_samples)
                            + y * rng.geometric(0.40, n_samples))
    ego_density          = np.clip(
        rng.beta(1.2, 12, n_samples)
        + y * rng.beta(3, 4, n_samples),
        0.0, 1.0,
    )
    cycle_count          = (rng.poisson(0.05, n_samples)
                            + y * rng.poisson(1.8, n_samples))

    # ── 3. Document Forensics & Cross-Modal Demographics ───────────────────
    # Legitimate applicants: very high OCR name similarity (0.90–1.0)
    # Fraud applicants: name discrepancies from transliteration attacks
    ocr_name_sim    = np.clip(
        rng.beta(25, 2, n_samples)
        - y * (rng.beta(4, 3, n_samples) * 0.65),
        0.0, 1.0,
    )
    ocr_contradiction = 1.0 - ocr_name_sim

    # Verhoeff / PAN checksum: legit ≈ 99.8% valid; fraud ≈ 55% valid
    id_checksum_valid = np.where(
        y == 1,
        rng.choice([0, 1], size=n_samples, p=[0.45, 0.55]),
        rng.choice([0, 1], size=n_samples, p=[0.002, 0.998]),
    )

    # ── 4. Media Forensics & Biometrics ────────────────────────────────────
    # ELA splice score: clean docs score low; forged / spliced docs score high
    tampering_ela_score = np.clip(
        rng.beta(1.5, 18, n_samples)
        + y * (rng.beta(5, 3, n_samples) * 0.75),
        0.0, 1.0,
    )
    # Cosine distance: genuine selfie↔ID ≈ 0.25; impostor swap ≈ 0.65
    face_cosine_dist = np.clip(
        rng.normal(0.25, 0.08, n_samples)
        + y * rng.normal(0.40, 0.12, n_samples),
        0.0, 1.5,
    )

    # ── 5. Missing-Signal Flags (Adversarial Evasion / Sensor Dropout) ─────
    is_doc_missing   = np.where(
        y == 1,
        rng.binomial(1, 0.08, n_samples),
        rng.binomial(1, 0.015, n_samples),
    )
    is_selfie_missing = np.where(
        y == 1,
        rng.binomial(1, 0.15, n_samples),
        rng.binomial(1, 0.03, n_samples),
    )
    # Isolated = no shared phone or device (fraudster using fresh burner)
    is_graph_isolated = np.where(
        (shared_phone_degree == 1) & (shared_device_degree == 1), 1, 0
    )

    # ── 6. M0 Deterministic Rule Score (pre-computed for M3 fusion) ─────────
    m0_triggers = (
        (shared_phone_degree  >= 3).astype(int)
        + (shared_device_degree >= 3).astype(int)
        + (ocr_contradiction    > 0.30).astype(int)
        + (id_checksum_valid   == 0).astype(int)
        + (velocity_1h          >= 4).astype(int)
    )
    m0_rule_score = np.clip(m0_triggers * 0.25, 0.0, 1.0)

    # ── Assemble DataFrame ──────────────────────────────────────────────────
    df = pd.DataFrame({
        # Tabular
        "customer_age":        customer_age,
        "income_lakhs":        income_lakhs,
        "account_type_risk":   account_type_risk,
        "velocity_1h":         velocity_1h,
        "velocity_24h":        velocity_24h,
        # Graph topology
        "shared_phone_degree":  shared_phone_degree,
        "shared_device_degree": shared_device_degree,
        "ego_density":          ego_density,
        "cycle_count":          cycle_count,
        # Forensics
        "ocr_contradiction":    ocr_contradiction,
        "id_checksum_valid":    id_checksum_valid,
        "tampering_ela_score":  tampering_ela_score,
        "face_cosine_dist":     face_cosine_dist,
        # Missingness
        "is_doc_missing":       is_doc_missing,
        "is_selfie_missing":    is_selfie_missing,
        "is_graph_isolated":    is_graph_isolated,
        # Pre-computed
        "m0_rule_score":        m0_rule_score,
        # Label
        "target_fraud":         y,
    })
    return df


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac:   float = VAL_FRAC,
    seed:       int   = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / val / test split that mirrors time-based splitting.

    For real BAF data you would split by month index.  For the synthetic
    benchmark we use stratified random splitting (fraud class preserved).

    Returns
    -------
    (train_df, val_df, test_df)
    """
    label_col = "target_fraud"
    test_size  = 1.0 - train_frac - val_frac

    train_df, test_df = train_test_split(
        df,
        test_size=round(test_size + val_frac, 4),
        stratify=df[label_col],
        random_state=seed,
    )
    # val / test — val_frac as fraction of remaining non-train data
    val_relative = round(val_frac / (test_size + val_frac), 4)
    val_df, test_df = train_test_split(
        test_df,
        test_size=round(1.0 - val_relative, 4),
        stratify=test_df[label_col],
        random_state=seed,
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = generate_indian_kyc_benchmark()
    tr, va, te = temporal_split(df)
    fraud_pct = df["target_fraud"].mean() * 100
    print(f"Total samples  : {len(df):,}")
    print(f"Fraud prevalence: {fraud_pct:.2f}%")
    print(f"Train / Val / Test : {len(tr):,} / {len(va):,} / {len(te):,}")
    print(df.describe().to_string())
