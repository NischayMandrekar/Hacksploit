"""
Download IEEE-CIS Fraud Detection dataset from Kaggle into argus/dataset/
Reads KAGGLE_TOKEN from .env (never committed to git).
"""
import os
import shutil
from pathlib import Path

# ── Load .env ────────────────────────────────────────────────────────────────
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

token = os.environ.get("KAGGLE_TOKEN", "")
if not token:
    raise RuntimeError("KAGGLE_TOKEN not found in .env")

# ── Authenticate programmatically ─────────────────────────────────────────────
from kagglehub.config import set_kaggle_api_token
set_kaggle_api_token(token)
print("Kaggle token set.")

# ── Download ──────────────────────────────────────────────────────────────────
import kagglehub

DEST = Path(__file__).parent / "dataset"
DEST.mkdir(exist_ok=True)

print("Downloading IEEE-CIS Fraud Detection competition files...")
print("(This may take several minutes — ~650 MB total)\n")

path = kagglehub.competition_download("ieee-fraud-detection")
print(f"\nDownloaded to cache: {path}")

# ── Copy only the files we need ───────────────────────────────────────────────
src = Path(path)
needed = ["train_transaction.csv", "train_identity.csv"]

for fname in needed:
    matches = list(src.rglob(fname))
    if matches:
        shutil.copy2(matches[0], DEST / fname)
        size_mb = (DEST / fname).stat().st_size / 1e6
        print(f"  [OK] {fname}  ->  dataset/{fname}  ({size_mb:.1f} MB)")
    else:
        print(f"  [WARN] {fname} not found.")
        print(f"         Make sure you accepted the competition rules at:")
        print(f"         https://www.kaggle.com/c/ieee-fraud-detection")

print(f"\nDataset ready at: {DEST.resolve()}")
print("\nNext — run training:")
print("  $env:PYTHONUTF8=1; .venv\\Scripts\\python.exe train_argus_models.py")
