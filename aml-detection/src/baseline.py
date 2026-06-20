"""
Supervised baseline for AML false-positive-rate measurement on the Elliptic dataset.

Reproduces the XGBoost baseline from Feedzai section 4.1 and extends the metrics
to include the false-positive rate (FPR) — the primary AML industry pain point.

The AML industry currently operates at a 90–95% false-positive rate on flagged
transactions (Coelho, De Simoni & Prenio, FSI Insights No. 18, BIS, 2019, p. 3).
This module establishes a measurable baseline FPR on public data to ground
that industry-wide figure in a reproducible experimental result.

Regulatory alignment:
  FS AI RMF (Feb 2026) — Pillar 2 (Risk Identification): control objectives
  require institutions to measure and document model error rates.
  FinCEN Strategic Plan 2022–2025: explicitly targets ML-based FP reduction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from .data_loader import feature_columns


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """FPR = FP / (FP + TN) — fraction of licit transactions wrongly flagged."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """FNR = FN / (FN + TP) — fraction of illicit transactions missed."""
    _, _, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0


def full_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    """Return a standardised metrics dictionary."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_illicit": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "false_positive_rate": round(false_positive_rate(y_true, y_pred), 4),
        "false_negative_rate": round(false_negative_rate(y_true, y_pred), 4),
    }
    if y_score is not None:
        metrics["auc_roc"] = round(roc_auc_score(y_true, y_score), 4)
    return metrics


def run_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_runs: int = 5,
    threshold: float = 0.5,
) -> dict:
    """
    XGBoost supervised baseline — 5 runs, average metrics.

    Matches the Feedzai section 4.1 experimental setup:
      - Temporal split: train time steps 1–34, test 35–49
      - Only labeled transactions
      - Default classification threshold (0.5)

    Parameters
    ----------
    threshold:
        Decision threshold for converting probabilities to binary labels.
        Default 0.5 matches the Feedzai baseline.

    Returns
    -------
    Dictionary with per-run predictions and averaged metrics.
    """
    feat_cols = feature_columns(X_train)
    X_tr = X_train[feat_cols].values
    X_te = X_test[feat_cols].values
    y_tr = y_train.values
    y_te = y_test.values

    run_metrics = []
    all_scores = []

    for seed in range(n_runs):
        clf = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        )
        clf.fit(X_tr, y_tr)
        y_score = clf.predict_proba(X_te)[:, 1]
        y_pred = (y_score >= threshold).astype(int)
        run_metrics.append(full_metrics(y_te, y_pred, y_score))
        all_scores.append(y_score)

    avg_metrics = {
        k: round(float(np.mean([m[k] for m in run_metrics])), 4)
        for k in run_metrics[0]
        if isinstance(run_metrics[0][k], (int, float))
    }
    avg_metrics["threshold"] = threshold
    avg_metrics["n_runs"] = n_runs

    return {
        "avg_metrics": avg_metrics,
        "per_run_metrics": run_metrics,
        "mean_probability_scores": np.mean(all_scores, axis=0),
    }


def metrics_per_timestep(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Compute per-time-step FPR and recall to show temporal stability."""
    records = []
    for ts in sorted(X_test["time_step"].unique()):
        mask = X_test["time_step"] == ts
        y_t = y_test[mask].values
        y_p = y_pred[mask]
        if len(np.unique(y_t)) < 2:
            continue
        records.append({
            "time_step": int(ts),
            "false_positive_rate": round(false_positive_rate(y_t, y_p), 4),
            "recall": round(recall_score(y_t, y_p, zero_division=0), 4),
            "f1_illicit": round(f1_score(y_t, y_p, pos_label=1, zero_division=0), 4),
            "n_samples": int(mask.sum()),
            "n_illicit": int(y_t.sum()),
        })
    return pd.DataFrame(records)
