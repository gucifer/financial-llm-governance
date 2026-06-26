"""
Generate FPR-Recall operating curve for baseline and hybrid models.

Outputs: results/operating_curve.png (Figure 7 in paper)

Shows the full threshold-sweep tradeoff curve for XGBoost baseline (default
threshold, scale_pos_weight=1) and hybrid (cost-sensitive + IF blend), with
known operating points for tuned GCN and GAT overlaid as markers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, roc_curve
from xgboost import XGBClassifier

# --- path setup so we can import src modules ---
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_elliptic, temporal_split, feature_columns  # noqa: E402
from src.baseline import full_metrics  # noqa: E402


def _fpr_recall_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Return (fpr_array, recall_array) sweeping all unique thresholds."""
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    return fpr, tpr  # tpr == recall for binary classification


def main():
    print("Loading Elliptic dataset...")
    X_full, y_full, _ = load_elliptic()
    X_train, X_test, y_train, y_test = temporal_split(X_full, y_full)

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    feat_cols = feature_columns(X_train)
    X_tr = X_train[feat_cols].values
    X_te = X_test[feat_cols].values
    y_tr = y_train.values
    y_te = y_test.values

    n_licit = int((y_tr == 0).sum())
    n_illicit = int((y_tr == 1).sum())
    spw = round(n_licit / max(n_illicit, 1), 2)

    # --- Baseline XGBoost (default threshold, no cost weighting) ---
    print("Training baseline XGBoost...")
    clf_base = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6,
        scale_pos_weight=1.0, eval_metric="logloss", random_state=0, verbosity=0,
    )
    clf_base.fit(X_tr, y_tr)
    y_score_base = clf_base.predict_proba(X_te)[:, 1]
    fpr_base, rec_base = _fpr_recall_curve(y_te, y_score_base)

    # Baseline operating point (threshold=0.5)
    y_pred_base = (y_score_base >= 0.5).astype(int)
    m_base = full_metrics(y_te, y_pred_base, y_score_base)
    op_base_fpr = m_base["false_positive_rate"]
    op_base_rec = m_base["recall"]

    # --- Hybrid XGBoost (cost-sensitive + IF blend, seed=0 representative run) ---
    print("Training hybrid XGBoost (seed 0)...")
    clf_hybrid = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6,
        scale_pos_weight=spw, eval_metric="logloss", random_state=0, verbosity=0,
    )
    clf_hybrid.fit(X_tr, y_tr)
    y_score_xgb_hybrid = clf_hybrid.predict_proba(X_te)[:, 1]

    # Apply IF blend
    licit_mask = y_tr == 0
    iso = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    iso.fit(X_tr[licit_mask])
    raw_if = -iso.decision_function(X_te)
    if_score = (raw_if - raw_if.min()) / (raw_if.max() - raw_if.min() + 1e-9)
    blend_weight = 0.15
    y_score_hybrid = (1 - blend_weight) * y_score_xgb_hybrid + blend_weight * if_score
    fpr_hybrid, rec_hybrid = _fpr_recall_curve(y_te, y_score_hybrid)

    # Hybrid operating point (avg threshold from paper = 0.8324)
    t_opt = 0.8324
    y_pred_hybrid = (y_score_hybrid >= t_opt).astype(int)
    m_hybrid = full_metrics(y_te, y_pred_hybrid, y_score_hybrid)
    op_hybrid_fpr = m_hybrid["false_positive_rate"]
    op_hybrid_rec = m_hybrid["recall"]

    # --- Known GNN operating points (from full_results.json) ---
    gnn_points = {
        "GCN (default)":         {"fpr": 0.0206, "rec": 0.6104, "marker": "^", "color": "#e67e22"},
        "GAT (default)":         {"fpr": 0.1526, "rec": 0.7516, "marker": "v", "color": "#8e44ad"},
        "GCN (tuned)":           {"fpr": 0.0017, "rec": 0.2108, "marker": "^", "color": "#e67e22",
                                   "facecolor": "none"},
        "GAT (tuned)":           {"fpr": 0.0130, "rec": 0.2545, "marker": "v", "color": "#8e44ad",
                                   "facecolor": "none"},
    }

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(fpr_base, rec_base, color="#e74c3c", lw=1.5, label="XGBoost baseline (curve)", zorder=2)
    ax.plot(fpr_hybrid, rec_hybrid, color="#2980b9", lw=1.5, label="Hybrid (curve)", zorder=2)

    # Operating points on curves
    ax.scatter([op_base_fpr], [op_base_rec], color="#e74c3c", s=100, zorder=5,
               marker="o", label=f"Baseline op. pt. (t=0.5, FPR={op_base_fpr:.4f}, Rec={op_base_rec:.4f})")
    ax.scatter([op_hybrid_fpr], [op_hybrid_rec], color="#2980b9", s=100, zorder=5,
               marker="D", label=f"Hybrid op. pt. (t=0.83, FPR={op_hybrid_fpr:.4f}, Rec={op_hybrid_rec:.4f})")

    # Recall floor
    ax.axhline(0.65, color="gray", lw=1.0, ls="--", alpha=0.7, label="Recall floor (0.65)")

    # GNN markers
    for name, gp in gnn_points.items():
        fc = gp.get("facecolor", gp["color"])
        ax.scatter([gp["fpr"]], [gp["rec"]], marker=gp["marker"], s=90, zorder=4,
                   edgecolors=gp["color"], facecolors=fc, linewidths=1.5, label=name)

    ax.set_xlabel("False-Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("Recall (TPR)", fontsize=11)
    ax.set_title("FPR–Recall Operating Curve\n(Elliptic dataset, temporal split)", fontsize=11)
    ax.set_xlim(-0.002, 0.20)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(True, alpha=0.3)

    out_path = ROOT / "results" / "operating_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
