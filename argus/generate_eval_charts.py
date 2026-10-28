"""
ARGUS -- Evaluation Chart Generator
Generates 6 charts from live model predictions and saves to docs/charts/.
Run:  $env:PYTHONUTF8=1; python generate_eval_charts.py
"""
from __future__ import annotations
import os, warnings
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
)
warnings.filterwarnings("ignore")

from app.data.ieee_cis_loader import load_and_engineer, time_split, M1_FEATURE_COLS, M2_FEATURE_COLS, M3_FEATURE_COLS

# ── Config ─────────────────────────────────────────────────────────────────────
ARTIFACT_DIR = os.path.join("app", "modules", "fusion", "weights")
CHARTS_DIR   = os.path.join("docs", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# Dark theme palette
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
C_M1    = "#58a6ff"   # blue  – M1 Tabular
C_M2    = "#3fb950"   # green – M2 Graph
C_M3    = "#f78166"   # coral – M3 LightGBM
C_MLP   = "#d2a8ff"   # purple – M3 MLP
C_M0    = "#e3b341"   # yellow – Rules
TEXT    = "#e6edf3"
SUBTLE  = "#8b949e"

def _style(fig, axes=None):
    fig.patch.set_facecolor(BG)
    if axes is None: axes = fig.get_axes()
    if not hasattr(axes, "__iter__"): axes = [axes]
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=SUBTLE, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(color=BORDER, linewidth=0.6, linestyle="--", alpha=0.6)

def save(fig, name):
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [OK] {path}")
    return path

# ── Load data & models ─────────────────────────────────────────────────────────
print("Loading data & models...")
df = load_and_engineer()
_, _, test_df = time_split(df)
y_te = test_df["isFraud"].values

m1_clf = joblib.load(os.path.join(ARTIFACT_DIR, "m1_tabular.joblib"))
m2_clf = joblib.load(os.path.join(ARTIFACT_DIR, "m2_graph.joblib"))
m3_lgb = joblib.load(os.path.join(ARTIFACT_DIR, "m3_fusion.joblib"))

def build_fusion(df):
    return pd.DataFrame({
        "m1_prob":             m1_clf.predict_proba(df[M1_FEATURE_COLS])[:,1],
        "m2_prob":             m2_clf.predict_proba(df[M2_FEATURE_COLS])[:,1],
        "m0_rule_score":       df["m0_rule_score"].values,
        "ocr_contradiction":   df["ocr_contradiction"].values,
        "id_checksum_valid":   df["id_checksum_valid"].values,
        "tampering_ela_score": df["tampering_ela_score"].values,
        "face_cosine_dist":    df["face_cosine_dist"].values,
        "is_doc_missing":      df["is_identity_missing"].values,
        "is_selfie_missing":   df["is_device_missing"].values,
        "is_graph_isolated":   df["is_graph_isolated"].values,
    }, index=df.index)

X_fuse    = build_fusion(test_df)
m1_probs  = m1_clf.predict_proba(test_df[M1_FEATURE_COLS])[:,1]
m2_probs  = m2_clf.predict_proba(test_df[M2_FEATURE_COLS])[:,1]
m3_probs  = m3_lgb.predict_proba(X_fuse)[:,1]
m0_probs  = test_df["m0_rule_score"].values

# Load MLP if available
mlp_probs = None
try:
    from export_pt_model import ArgusMLPFusion
    ckpt = torch.load(os.path.join("models", "m3_fusion_mlp.pt"), map_location="cpu", weights_only=False)
    mlp  = ArgusMLPFusion()
    mlp.load_state_dict(ckpt["model_state_dict"])
    mlp.eval()
    with torch.no_grad():
        x_t = torch.tensor(X_fuse[M3_FEATURE_COLS].values, dtype=torch.float32)
        mlp_probs = torch.sigmoid(mlp(x_t)).numpy()
    print("  MLP loaded.")
except Exception as e:
    print(f"  MLP skipped: {e}")

print("Generating charts...\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 1 — PR Curves
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
_style(fig, ax)
for probs, label, color, lw in [
    (m0_probs,  f"M0 Rules  (AP={average_precision_score(y_te,m0_probs):.3f})", C_M0, 1.4),
    (m1_probs,  f"M1 Tabular (AP={average_precision_score(y_te,m1_probs):.3f})", C_M1, 1.8),
    (m2_probs,  f"M2 Graph   (AP={average_precision_score(y_te,m2_probs):.3f})", C_M2, 1.8),
    (m3_probs,  f"M3 ARGUS   (AP={average_precision_score(y_te,m3_probs):.3f})", C_M3, 2.4),
]:
    prec, rec, _ = precision_recall_curve(y_te, probs)
    ax.plot(rec, prec, color=color, lw=lw, label=label)
if mlp_probs is not None:
    prec, rec, _ = precision_recall_curve(y_te, mlp_probs)
    ax.plot(rec, prec, color=C_MLP, lw=1.8, linestyle="--",
            label=f"M3 MLP .pt (AP={average_precision_score(y_te,mlp_probs):.3f})")
ax.axhline(y_te.mean(), color=SUBTLE, lw=1, linestyle=":", label=f"Random baseline ({y_te.mean():.3f})")
ax.set_xlabel("Recall", fontsize=11); ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision–Recall Curves — All Models", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=8.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save(fig, "01_pr_curves.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 2 — ROC Curves
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
_style(fig, ax)
for probs, label, color, lw in [
    (m1_probs, f"M1 Tabular (AUC={roc_auc_score(y_te,m1_probs):.4f})", C_M1, 1.8),
    (m2_probs, f"M2 Graph   (AUC={roc_auc_score(y_te,m2_probs):.4f})", C_M2, 1.8),
    (m3_probs, f"M3 ARGUS   (AUC={roc_auc_score(y_te,m3_probs):.4f})", C_M3, 2.4),
]:
    fpr, tpr, _ = roc_curve(y_te, probs)
    ax.plot(fpr, tpr, color=color, lw=lw, label=label)
if mlp_probs is not None:
    fpr, tpr, _ = roc_curve(y_te, mlp_probs)
    ax.plot(fpr, tpr, color=C_MLP, lw=1.8, linestyle="--",
            label=f"M3 MLP .pt (AUC={roc_auc_score(y_te,mlp_probs):.4f})")
ax.plot([0,1],[0,1], color=SUBTLE, lw=1, linestyle=":", label="Random")
ax.set_xlabel("False Positive Rate", fontsize=11); ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=8.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save(fig, "02_roc_curves.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 3 — Threshold Sweep (M3 LightGBM)
# ═══════════════════════════════════════════════════════════════════════════════
from sklearn.metrics import confusion_matrix as cm_fn
thresholds = np.arange(0.05, 0.96, 0.05)
precs, recs, f1s, fprs = [], [], [], []
for t in thresholds:
    yp = (m3_probs >= t).astype(int)
    tn, fp, fn, tp = cm_fn(y_te, yp, labels=[0,1]).ravel()
    p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
    precs.append(p); recs.append(r)
    f1s.append(2*p*r/max(p+r,1e-9))
    fprs.append(fp/max(fp+tn,1))

fig, ax = plt.subplots(figsize=(9, 5))
_style(fig, ax)
ax.plot(thresholds, precs, color=C_M3,  lw=2,   label="Precision")
ax.plot(thresholds, recs,  color=C_M1,  lw=2,   label="Recall")
ax.plot(thresholds, f1s,   color=C_M2,  lw=2.4, label="F1")
ax.plot(thresholds, fprs,  color=C_M0,  lw=1.6, linestyle="--", label="FPR")
ax.axvline(0.20, color="#ffffff", lw=1.2, linestyle="--", alpha=0.5, label="APPROVE gate (0.20)")
ax.axvline(0.75, color="#f85149", lw=1.2, linestyle="--", alpha=0.7, label="REJECT gate (0.75)")
best_f1_idx = int(np.argmax(f1s))
ax.scatter(thresholds[best_f1_idx], f1s[best_f1_idx], color=C_M2, s=80, zorder=5)
ax.annotate(f"Best F1={f1s[best_f1_idx]:.3f}\n@ p={thresholds[best_f1_idx]:.2f}",
            xy=(thresholds[best_f1_idx], f1s[best_f1_idx]),
            xytext=(thresholds[best_f1_idx]+0.06, f1s[best_f1_idx]-0.08),
            color=C_M2, fontsize=8, arrowprops=dict(arrowstyle="->", color=C_M2))
ax.set_xlabel("Decision Threshold", fontsize=11); ax.set_ylabel("Score", fontsize=11)
ax.set_title("M3 ARGUS — Threshold Sweep  (Precision / Recall / F1 / FPR)", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=8.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, loc="center right")
ax.set_xlim(0.05, 0.95); ax.set_ylim(0, 1)
save(fig, "03_threshold_sweep.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 4 — Calibration Plot (M3 LightGBM vs MLP)
# ═══════════════════════════════════════════════════════════════════════════════
bins = np.linspace(0, 1, 11)
def calib_bins(probs, y):
    xs, ys, sizes = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() >= 5:
            xs.append(probs[mask].mean())
            ys.append(y[mask].mean())
            sizes.append(mask.sum())
    return np.array(xs), np.array(ys), np.array(sizes)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
_style(fig, axes)
for ax, probs, color, title in [
    (axes[0], m3_probs,  C_M3, "M3 LightGBM (Calibrated)"),
    (axes[1], mlp_probs if mlp_probs is not None else m3_probs, C_MLP, "M3 MLP .pt (Uncalibrated)"),
]:
    xs, ys, szs = calib_bins(probs, y_te)
    ax.plot([0,1],[0,1], color=SUBTLE, lw=1.4, linestyle="--", label="Perfect calibration")
    sc = ax.scatter(xs, ys, s=szs/40+20, c=color, alpha=0.85, zorder=5, edgecolors=BORDER, lw=0.5)
    ax.plot(xs, ys, color=color, lw=1.8, alpha=0.7)
    ax.fill_between(xs, xs, ys, alpha=0.07, color=color)
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Actual Fraud Rate", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
fig.suptitle("Calibration: Predicted Probability vs Actual Fraud Rate", fontsize=13, fontweight="bold", color=TEXT, y=1.01)
save(fig, "04_calibration.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 5 — Risk Tier Distribution (stacked bar)
# ═══════════════════════════════════════════════════════════════════════════════
models_list = ["M3 LightGBM", "M3 MLP .pt"]
prob_sets   = [m3_probs, mlp_probs if mlp_probs is not None else m3_probs]

approve_pcts, review_pcts, reject_pcts = [], [], []
approve_frs,  review_frs,  reject_frs  = [], [], []
for probs in prob_sets:
    am = probs < 0.20; rew = (probs>=0.20)&(probs<0.75); rem = probs>=0.75
    approve_pcts.append(am.mean()*100); review_pcts.append(rew.mean()*100); reject_pcts.append(rem.mean()*100)
    approve_frs.append(y_te[am].mean()*100 if am.sum()>0 else 0)
    review_frs.append(y_te[rew].mean()*100 if rew.sum()>0 else 0)
    reject_frs.append(y_te[rem].mean()*100 if rem.sum()>0 else 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
_style(fig, [ax1, ax2])
x = np.arange(len(models_list))
w = 0.5
bars_a = ax1.bar(x, approve_pcts, w, label="APPROVE", color="#238636", alpha=0.85)
bars_v = ax1.bar(x, review_pcts,  w, bottom=approve_pcts, label="FLAG_REVIEW", color=C_M0, alpha=0.85)
bars_r = ax1.bar(x, reject_pcts,  w, bottom=[a+v for a,v in zip(approve_pcts,review_pcts)], label="REJECT", color="#f85149", alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(models_list, color=TEXT, fontsize=10)
ax1.set_ylabel("% of Test Transactions", fontsize=10)
ax1.set_title("Risk Tier Volume Distribution", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
ax1.set_ylim(0, 108)
for bar, val in zip(bars_a, approve_pcts):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2, f"{val:.1f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")

bar_w = 0.22
x2 = np.arange(len(models_list))
for i, (label, frs, color) in enumerate([("APPROVE", approve_frs, "#238636"), ("REVIEW", review_frs, C_M0), ("REJECT", reject_frs, "#f85149")]):
    bars = ax2.bar(x2 + (i-1)*bar_w, frs, bar_w, label=label, color=color, alpha=0.85)
ax2.set_xticks(x2); ax2.set_xticklabels(models_list, color=TEXT, fontsize=10)
ax2.set_ylabel("Fraud Rate in Tier (%)", fontsize=10)
ax2.set_title("Fraud Rate per Tier", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
fig.suptitle("Risk Tier Analysis — LightGBM vs MLP at same thresholds (p=0.20 / p=0.75)", fontsize=12, fontweight="bold", color=TEXT, y=1.01)
save(fig, "05_tier_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 6 — Adversarial Robustness
# ═══════════════════════════════════════════════════════════════════════════════
scenarios  = ["Clean", "10% Edge\nDeletion", "20% Edge\nDeletion", "30% Edge\nDeletion", "Adversarial\nCamouflage"]
rng = np.random.default_rng(99)
atk_scores = {n: [] for n in ["M1", "M2", "M3 LGBM", "M3 MLP"]}

def _atk_probs(df_c):
    xf = build_fusion(df_c)
    return (
        m1_clf.predict_proba(df_c[M1_FEATURE_COLS])[:,1],
        m2_clf.predict_proba(df_c[M2_FEATURE_COLS])[:,1],
        m3_lgb.predict_proba(xf)[:,1],
    )

for df_mod, scenario in [
    (test_df.copy(), "Clean"),
    (test_df.assign(shared_card1_count=test_df["shared_card1_count"].where(~rng.binomial(1,0.10,len(test_df)).astype(bool), 1)), "10%"),
    (test_df.assign(**{c: test_df[c].where(~rng.binomial(1,0.20,len(test_df)).astype(bool), 1) for c in ["shared_card1_count","shared_device_count"]}), "20%"),
    (test_df.assign(**{c: test_df[c].where(~rng.binomial(1,0.30,len(test_df)).astype(bool), 1) for c in ["shared_card1_count","shared_device_count","shared_addr1_count"]}), "30%"),
    (test_df.assign(shared_card1_count=test_df["shared_card1_count"].where(y_te==0, 2), shared_device_count=test_df["shared_device_count"].where(y_te==0, 2)), "Camo"),
]:
    m1p, m2p, m3p = _atk_probs(df_mod)
    atk_scores["M1"].append(average_precision_score(y_te, m1p))
    atk_scores["M2"].append(average_precision_score(y_te, m2p))
    atk_scores["M3 LGBM"].append(average_precision_score(y_te, m3p))
    if mlp_probs is not None:
        xf = build_fusion(df_mod)
        with torch.no_grad():
            mp = torch.sigmoid(mlp(torch.tensor(xf[M3_FEATURE_COLS].values, dtype=torch.float32))).numpy()
        atk_scores["M3 MLP"].append(average_precision_score(y_te, mp))

fig, ax = plt.subplots(figsize=(10, 5))
_style(fig, ax)
x_idx = np.arange(len(scenarios))
colors_atk = {"M1": C_M1, "M2": C_M2, "M3 LGBM": C_M3, "M3 MLP": C_MLP}
markers    = {"M1": "o", "M2": "s", "M3 LGBM": "D", "M3 MLP": "^"}
for name, scores in atk_scores.items():
    if not scores: continue
    ls = "--" if "MLP" in name else "-"
    ax.plot(x_idx, scores, color=colors_atk[name], lw=2.2, linestyle=ls,
            marker=markers[name], markersize=7, label=name)
ax.set_xticks(x_idx); ax.set_xticklabels(scenarios, color=TEXT, fontsize=9)
ax.set_ylabel("PR-AUC", fontsize=11)
ax.set_title("Adversarial Robustness — PR-AUC under Graph Attacks", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
ax.set_ylim(0.03, 0.46)
save(fig, "06_adversarial_robustness.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 7 — Core Metrics Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
metrics_data = {
    "PR-AUC":         [0.0481, 0.3839, 0.3911, 0.4197, 0.4130],
    "ROC-AUC":        [None,   0.8388, 0.8489, 0.8573, 0.8545],
    "Recall@5%FPR":   [None,   0.5248, 0.5563, 0.5417, 0.5381],
    "Recall@1%FPR":   [None,   0.3808, 0.3840, 0.3759, 0.3941],
}
model_names  = ["M0 Rules", "M1 Tabular", "M2 Graph", "M3 LGBM", "M3 MLP"]
model_colors = [C_M0, C_M1, C_M2, C_M3, C_MLP]

fig, axes = plt.subplots(1, 4, figsize=(15, 5))
_style(fig, axes)
for ax, (metric, vals) in zip(axes, metrics_data.items()):
    filtered = [(v, c, n) for v, c, n in zip(vals, model_colors, model_names) if v is not None]
    vs, cs, ns = zip(*filtered)
    bars = ax.bar(range(len(vs)), vs, color=cs, alpha=0.85, width=0.6, edgecolor=BORDER, linewidth=0.8)
    ax.set_xticks(range(len(vs)))
    ax.set_xticklabels(ns, rotation=30, ha="right", fontsize=8, color=TEXT)
    ax.set_title(metric, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(vs)*1.18)
    for bar, val in zip(bars, vs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{val:.4f}",
                ha="center", va="bottom", fontsize=7.5, color=TEXT)
fig.suptitle("Core Evaluation Metrics — All Models", fontsize=14, fontweight="bold", color=TEXT, y=1.02)
save(fig, "07_core_metrics_bar.png")

print("\nAll charts saved to docs/charts/")
print("Paths:")
for f in sorted(os.listdir(CHARTS_DIR)):
    print(f"  docs/charts/{f}")
