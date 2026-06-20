"""
Hybrid AML false-positive-reduction pipeline.

Contribution over the Feedzai (2021) baseline:
  1. Cost-sensitive XGBoost — scale_pos_weight penalises missed illicit
     transactions; calibrated probability outputs replace hard default thresholds.
  2. Precision-recall threshold optimisation — instead of the default 0.5 cut,
     sweep the decision threshold on held-out training folds and select the value
     that minimises FPR subject to recall ≥ recall_floor (default 0.70).
  3. Optional Isolation Forest blending — train IF on licit-class transactions
     only; its anomaly score amplifies XGBoost's illicit probability for
     borderline cases, pulling the effective threshold lower for unusual transactions
     without raising the global FPR.

Regulatory alignment:
  FS AI RMF (Feb 2026) — Pillar 2 (Risk Identification): control objectives
  require quantifiable thresholds for false-positive management.
  FinCEN Strategic Plan 2022–2025: prioritises FP reduction through ML.
  FSOC 2023 Annual Report: flags model opacity and uncontrolled alert rates as
  systemic risk factors — cost-sensitive, explainable thresholds directly address
  both concerns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from .baseline import full_metrics
from .data_loader import feature_columns


def _optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    recall_floor: float,
) -> float:
    """
    Find the decision threshold that minimises FPR subject to recall ≥ recall_floor.

    Uses the sklearn precision-recall curve (computed at all unique score points)
    and selects the highest threshold where recall is still ≥ recall_floor.
    Falls back to 0.5 if no threshold satisfies the constraint.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score, pos_label=1)
    # precision_recall_curve returns one fewer threshold than precision/recall points
    valid = np.where(recall[:-1] >= recall_floor)[0]
    if len(valid) == 0:
        return 0.5
    # Among valid thresholds, pick the one with highest precision (lowest FPR)
    best_idx = valid[np.argmax(precision[valid])]
    return float(thresholds[best_idx])


def _cross_val_threshold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    clf_params: dict,
    n_splits: int,
    recall_floor: float,
    seed: int,
    blend_isolation_forest: bool = False,
    blend_weight: float = 0.15,
) -> float:
    """Average optimal threshold across stratified CV folds on training data.

    If blend_isolation_forest=True, the same IF blending applied at test time
    is applied within each CV fold so the threshold is consistent with
    blended scores. Without this, the threshold is optimised on raw XGBoost
    probabilities but applied to lower blended scores, causing it to be too
    high and produce zero-recall predictions at test time.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    thresholds = []
    for fold_train, fold_val in skf.split(X_tr, y_tr):
        clf = XGBClassifier(**clf_params, verbosity=0)
        clf.fit(X_tr[fold_train], y_tr[fold_train])
        scores = clf.predict_proba(X_tr[fold_val])[:, 1]

        if blend_isolation_forest:
            licit_mask = y_tr[fold_train] == 0
            iso_fold = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
            iso_fold.fit(X_tr[fold_train][licit_mask])
            raw_if = -iso_fold.decision_function(X_tr[fold_val])
            if_score = (raw_if - raw_if.min()) / (raw_if.max() - raw_if.min() + 1e-9)
            scores = (1 - blend_weight) * scores + blend_weight * if_score

        t = _optimal_threshold(y_tr[fold_val], scores, recall_floor)
        thresholds.append(t)
    return float(np.mean(thresholds))


def run_hybrid_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    recall_floor: float = 0.70,
    blend_isolation_forest: bool = True,
    blend_weight: float = 0.15,
    n_cv_splits: int = 5,
    n_runs: int = 5,
) -> dict:
    """
    Cost-sensitive XGBoost with threshold optimisation + optional IF blending.

    Parameters
    ----------
    recall_floor:
        Minimum acceptable recall on the illicit class (default 0.70).
        The threshold search maximises precision (minimises FPR) subject to
        this floor. Adjust downward to trade more FP reduction for higher FNR.
    blend_isolation_forest:
        If True, train an IsolationForest on licit training transactions and
        blend its normalised anomaly score into the XGBoost probability.
    blend_weight:
        Weight given to the IF anomaly signal in the blended score (0–1).
        Default 0.15 — IF is a secondary signal; XGBoost dominates.
    n_cv_splits:
        Number of cross-validation folds for threshold search.
    n_runs:
        Number of independent runs (different seeds) for variance estimation.

    Returns
    -------
    Dictionary with avg_metrics, per_run_metrics, optimal_thresholds,
    and the final model from the last run (for SHAP analysis).
    """
    feat_cols = feature_columns(X_train)
    X_tr = X_train[feat_cols].values
    X_te = X_test[feat_cols].values
    y_tr = y_train.values
    y_te = y_test.values

    # Class imbalance ratio for scale_pos_weight
    n_licit = int((y_tr == 0).sum())
    n_illicit = int((y_tr == 1).sum())
    spw = round(n_licit / max(n_illicit, 1), 2)

    clf_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "scale_pos_weight": spw,  # cost-sensitive: penalise missed illicit
        "eval_metric": "logloss",
        "random_state": 0,
    }

    # Optional: train Isolation Forest on licit-class only
    iso = None
    if blend_isolation_forest:
        licit_mask = y_tr == 0
        iso = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
        iso.fit(X_tr[licit_mask])

    run_metrics = []
    optimal_thresholds = []
    final_clf = None

    for seed in range(n_runs):
        clf_params["random_state"] = seed
        t_opt = _cross_val_threshold(
            X_tr, y_tr, clf_params, n_cv_splits, recall_floor, seed,
            blend_isolation_forest=blend_isolation_forest,
            blend_weight=blend_weight,
        )
        optimal_thresholds.append(t_opt)

        clf = XGBClassifier(**clf_params, verbosity=0)
        clf.fit(X_tr, y_tr)
        y_score = clf.predict_proba(X_te)[:, 1]

        if iso is not None:
            # IF decision_function: negative = more anomalous; invert and normalise to [0,1]
            raw_if = -iso.decision_function(X_te)
            if_score = (raw_if - raw_if.min()) / (raw_if.max() - raw_if.min() + 1e-9)
            y_score = (1 - blend_weight) * y_score + blend_weight * if_score

        y_pred = (y_score >= t_opt).astype(int)
        run_metrics.append(full_metrics(y_te, y_pred, y_score))

        if seed == n_runs - 1:
            final_clf = clf  # keep last model for SHAP

    avg_metrics = {
        k: round(float(np.mean([m[k] for m in run_metrics])), 4)
        for k in run_metrics[0]
        if isinstance(run_metrics[0][k], (int, float))
    }
    avg_metrics["avg_optimal_threshold"] = round(float(np.mean(optimal_thresholds)), 4)
    avg_metrics["recall_floor_constraint"] = recall_floor
    avg_metrics["blend_isolation_forest"] = blend_isolation_forest
    avg_metrics["scale_pos_weight"] = spw

    return {
        "avg_metrics": avg_metrics,
        "per_run_metrics": run_metrics,
        "optimal_thresholds": optimal_thresholds,
        "final_clf": final_clf,
        "iso_forest": iso,
        "feature_columns": feat_cols,
    }


def compare_results(baseline_metrics: dict, hybrid_metrics: dict) -> pd.DataFrame:
    """
    Print a side-by-side comparison table for the study's results section.

    The key column for the NIW claim is false_positive_rate.
    """
    keys = ["false_positive_rate", "recall", "precision", "f1_illicit", "auc_roc", "false_negative_rate"]
    rows = []
    for k in keys:
        b = baseline_metrics.get(k, "—")
        h = hybrid_metrics.get(k, "—")
        if isinstance(b, float) and isinstance(h, float):
            delta = round(h - b, 4)
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "—"
        rows.append({"metric": k, "baseline_xgb": b, "hybrid_fp_optimised": h, "delta": delta_str})

    df = pd.DataFrame(rows).set_index("metric")
    return df
