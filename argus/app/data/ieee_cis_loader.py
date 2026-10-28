"""
IEEE-CIS Fraud Detection — Data Loader & Feature Engineer
===========================================================
Loads the real IEEE-CIS Kaggle competition dataset and engineers
ARGUS-compatible features for M1, M2, and M3.

Dataset files (placed in argus/dataset/):
    train_transaction.csv   ~590k rows, 394 columns (label: isFraud)
    train_identity.csv      ~144k rows, 41 columns  (join key: TransactionID)

Feature Engineering Strategy
-----------------------------
Tabular (M1):
    TransactionAmt, card features, address, dist, C-count features,
    D-timedelta features, M-match flags

Graph / Identity-Sharing (M2):
    shared_card1_count      — how many transactions share the same card1 (proxy phone/device)
    shared_addr1_count      — how many transactions share addr1 (proxy location ring)
    shared_device_count     — transactions sharing identical DeviceInfo
    identity_nulls          — # missing identity columns (adversarial evasion signal)
    addr_mismatch           — addr1 vs addr2 difference (cross-modal contradiction)

Forensic / Missingness flags (M3 fusion):
    is_identity_missing     — 1 if applicant has NO match in identity table
    is_device_missing       — 1 if DeviceInfo is NaN
    M_match_score           — mean of M1–M9 binary match flags (document consistency)
    id_checksum_valid       — proxy: all M-flags agree (no contradictions)
    ocr_contradiction       — 1 - M_match_score
    tampering_ela_score     — V-feature derived anomaly score (normalised)
    face_cosine_dist        — id_02 (normalised) as biometric proxy
    m0_rule_score           — pre-computed from engineered rule triggers

Time-based split:
    Train: earliest 70% of TransactionDT
    Val:   next 15%
    Test:  final 15%
    (Prevents future-to-past leakage — critical for fraud detection.)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


DATASET_DIR = Path(__file__).parent.parent.parent / "dataset"

TRANSACTION_CSV = DATASET_DIR / "train_transaction.csv"
IDENTITY_CSV    = DATASET_DIR / "train_identity.csv"

# ── Feature column lists ─────────────────────────────────────────────────────

# Raw columns kept from transaction file
TRANSACTION_KEEP = [
    "TransactionID", "TransactionDT", "isFraud",
    "TransactionAmt",
    "card1", "card2", "card3", "card5",            # card identifiers
    "addr1", "addr2",                               # billing/shipping address codes
    "dist1", "dist2",                               # distances
    "P_emaildomain", "R_emaildomain",               # email domains
    # C-features: counts of transactions with same card/address/email
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
    # D-features: time deltas between transactions
    "D1", "D2", "D3", "D4", "D10", "D15",
    # M-features: match flags (name, address, etc.)
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    # Vesta V-features: pick the top informative ones
    "V1", "V2", "V3", "V4", "V12", "V13", "V20", "V30", "V38", "V44",
    "V45", "V53", "V54", "V67", "V75", "V76", "V82", "V91", "V127",
    "V220", "V258", "V261", "V264", "V267", "V271", "V274", "V277",
    "V279", "V283", "V285", "V307", "V310", "V314", "V317",
]

# Raw columns kept from identity file
IDENTITY_KEEP = [
    "TransactionID",
    "id_01", "id_02", "id_03", "id_05", "id_06",
    "id_09", "id_10", "id_11", "id_13", "id_14",
    "id_17", "id_19", "id_20",
    "DeviceType", "DeviceInfo",
]

# M1 tabular features (non-relational)
M1_FEATURE_COLS = [
    "TransactionAmt_log",
    "card1", "card2", "card3", "card5",
    "addr1", "addr2",
    "dist1", "dist2",
    "C1", "C2", "C5", "C6", "C11", "C13", "C14",
    "D1", "D2", "D4", "D10", "D15",
    "M_match_score",
    "P_emaildomain_enc", "R_emaildomain_enc",
]

# M2 adds graph / identity-sharing features
M2_FEATURE_COLS = M1_FEATURE_COLS + [
    "shared_card1_count",
    "shared_addr1_count",
    "shared_device_count",
    "identity_nulls",
    "addr_mismatch",
]

# M3 fusion input
M3_FEATURE_COLS = [
    "m1_prob",
    "m2_prob",
    "m0_rule_score",
    "ocr_contradiction",
    "id_checksum_valid",
    "tampering_ela_score",
    "face_cosine_dist",
    "is_doc_missing",          # proxy: is_identity_missing
    "is_selfie_missing",       # proxy: is_device_missing
    "is_graph_isolated",       # no shared card or device
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_load_cols(path: Path, keep: list[str]) -> pd.DataFrame:
    """Load CSV keeping only columns that actually exist in the file."""
    available = pd.read_csv(path, nrows=0).columns.tolist()
    cols = [c for c in keep if c in available]
    return pd.read_csv(path, usecols=cols, low_memory=False)


def _encode_m_flags(df: pd.DataFrame) -> pd.Series:
    """Convert M1-M9 (T/F/NaN) to a [0,1] match score."""
    m_cols = [c for c in ["M1","M2","M3","M4","M5","M6","M7","M8","M9"] if c in df.columns]
    if not m_cols:
        return pd.Series(np.zeros(len(df)), index=df.index)
    binary = df[m_cols].apply(lambda s: s.map({"T": 1, "F": 0}))
    return binary.mean(axis=1).fillna(0.5)


def _frequency_encode(df: pd.DataFrame, col: str) -> pd.Series:
    freq = df[col].value_counts(normalize=True)
    return df[col].map(freq).fillna(0).astype(float)


def _sharing_count(df: pd.DataFrame, col: str, max_cap: int = 50) -> pd.Series:
    """
    For each row, count how many other rows share the same value of `col`.
    Returns the count clipped at max_cap, NaN rows → 0.
    """
    counts = df[col].map(df[col].value_counts())
    return counts.fillna(0).clip(upper=max_cap).astype(float)


# ── Main loader ───────────────────────────────────────────────────────────────

def load_and_engineer(
    transaction_csv: str | Path = TRANSACTION_CSV,
    identity_csv:    str | Path = IDENTITY_CSV,
    subsample_frac:  float | None = None,
    seed:            int = 42,
) -> pd.DataFrame:
    """
    Load, merge, and engineer all ARGUS features from the IEEE-CIS files.

    Parameters
    ----------
    transaction_csv : path to train_transaction.csv
    identity_csv    : path to train_identity.csv
    subsample_frac  : float in (0,1] — use a random fraction for fast dev runs
    seed            : reproducibility seed

    Returns
    -------
    pd.DataFrame with all M1 + M2 + M3 compatible columns.
    """
    print("  Loading transaction file...")
    tx = _safe_load_cols(Path(transaction_csv), TRANSACTION_KEEP)

    print("  Loading identity file...")
    idf = _safe_load_cols(Path(identity_csv), IDENTITY_KEEP)

    print("  Merging on TransactionID...")
    df = tx.merge(idf, on="TransactionID", how="left")

    if subsample_frac and subsample_frac < 1.0:
        df = df.sample(frac=subsample_frac, random_state=seed).reset_index(drop=True)
        print(f"  Subsampled to {len(df):,} rows ({subsample_frac*100:.0f}%)")

    # ── Basic cleaning ────────────────────────────────────────────────────
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # ── Email domain encoding ─────────────────────────────────────────────
    for col in ["P_emaildomain", "R_emaildomain"]:
        enc_col = col + "_enc"
        df[enc_col] = _frequency_encode(df, col) if col in df.columns else 0.0

    # ── M-flag match score ────────────────────────────────────────────────
    df["M_match_score"] = _encode_m_flags(df)
    df["ocr_contradiction"] = 1.0 - df["M_match_score"]
    # id_checksum_valid: 1 if all available M flags agree (score ≈ 1.0 or 0.0)
    df["id_checksum_valid"] = (df["M_match_score"] >= 0.9).astype(int)

    # ── Graph / identity-sharing features ─────────────────────────────────
    print("  Engineering graph-sharing features...")
    df["shared_card1_count"]  = _sharing_count(df, "card1")  if "card1"  in df.columns else 0.0
    df["shared_addr1_count"]  = _sharing_count(df, "addr1")  if "addr1"  in df.columns else 0.0
    df["shared_device_count"] = _sharing_count(df, "DeviceInfo") if "DeviceInfo" in df.columns else 0.0

    # addr mismatch (billing vs shipping state divergence)
    if "addr1" in df.columns and "addr2" in df.columns:
        df["addr_mismatch"] = (df["addr1"].fillna(-1) != df["addr2"].fillna(-2)).astype(float)
    else:
        df["addr_mismatch"] = 0.0

    # identity null count (adversarial evasion: missing many identity fields)
    id_cols = [c for c in IDENTITY_KEEP if c not in ("TransactionID", "DeviceType", "DeviceInfo")]
    present_id_cols = [c for c in id_cols if c in df.columns]
    df["identity_nulls"] = df[present_id_cols].isnull().sum(axis=1).astype(float)

    # ── Missingness flags ─────────────────────────────────────────────────
    df["is_identity_missing"] = (~df["TransactionID"].isin(idf["TransactionID"])).astype(int)
    df["is_device_missing"]   = df["DeviceInfo"].isnull().astype(int) if "DeviceInfo" in df.columns else 1

    # is_graph_isolated: no meaningful card/device sharing
    df["is_graph_isolated"] = (
        (df["shared_card1_count"] <= 1) & (df["shared_device_count"] <= 1)
    ).astype(int)

    # ── Forensic / biometric proxies ─────────────────────────────────────
    # tampering_ela_score proxy: normalise id_02 (numerical identity anomaly score)
    if "id_02" in df.columns:
        raw = df["id_02"].fillna(0).clip(lower=0)
        df["tampering_ela_score"] = (raw / (raw.max() + 1e-9)).clip(0, 1)
    else:
        df["tampering_ela_score"] = 0.0

    # face_cosine_dist proxy: id_01 (account-age–based distance)
    if "id_01" in df.columns:
        raw = df["id_01"].abs().fillna(0)
        df["face_cosine_dist"] = (raw / (raw.max() + 1e-9)).clip(0, 1)
    else:
        df["face_cosine_dist"] = 0.0

    # ── M0 rule score ─────────────────────────────────────────────────────
    m0_triggers = (
        (df["shared_card1_count"]  >= 5).astype(int)
        + (df["shared_device_count"] >= 5).astype(int)
        + (df["ocr_contradiction"]   > 0.4).astype(int)
        + (df["id_checksum_valid"]  == 0).astype(int)
        + (df["addr_mismatch"]       == 1).astype(int)
    )
    df["m0_rule_score"] = np.clip(m0_triggers * 0.25, 0.0, 1.0)

    # ── Fill remaining NaNs with median ───────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # ── Encode any remaining categoricals as integers ─────────────────────
    for col in ["card4", "card6", "DeviceType"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    print(f"  Final shape: {df.shape} | Fraud rate: {df['isFraud'].mean()*100:.2f}%")
    return df


def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split on TransactionDT — preserves temporal ordering
    to prevent future-to-past label leakage.
    """
    df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * train_frac)
    n_val   = int(n * (train_frac + val_frac))
    return (
        df_sorted.iloc[:n_train].copy(),
        df_sorted.iloc[n_train:n_val].copy(),
        df_sorted.iloc[n_val:].copy(),
    )


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_and_engineer(subsample_frac=0.1)
    tr, va, te = time_split(df)
    print(f"\nTrain: {len(tr):,} | Val: {len(va):,} | Test: {len(te):,}")
    print(f"Train fraud rate: {tr['isFraud'].mean()*100:.2f}%")
    print(f"Test fraud rate : {te['isFraud'].mean()*100:.2f}%")
