"""
SHAP-based explainability module for AML model decisions.

Generates per-transaction SHAP feature attributions and formats them as
structured audit JSON aligned with FINRA Rule 4511 and SEC Rule 17a-4
requirements for explainable audit trails.

This bridges the legal explainability requirement — which FINRA and the SEC
impose on algorithmic decisions affecting customers — with the black-box nature
of gradient-boosted tree ensembles, a challenge the CFPB explicitly flagged in
its 2022 Circular on AI-based credit decisions.

Regulatory alignment:
  FS AI RMF (Feb 2026) — Pillar 4 (Incident Response): requires institutions
  to maintain auditable records linking model outputs to model inputs.
  FINRA Rule 4511: requires firms to make and preserve books and records, incl. automated decision documentation.
  SEC Rule 17a-4: mandates preservation of business records including
  automated decision rationales.
  FinCEN SAR requirements: Suspicious Activity Report narratives must be
  supported by documented evidence trails.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier


def build_explainer(clf: XGBClassifier) -> shap.TreeExplainer:
    """Build a SHAP TreeExplainer for an XGBoost model."""
    return shap.TreeExplainer(clf)


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Compute SHAP values for a feature matrix.

    Returns
    -------
    shap_values : np.ndarray of shape (n_samples, n_features)
        Signed SHAP contributions for the illicit class (positive = increases
        illicit risk, negative = decreases it).
    """
    X_arr = X[feature_cols].values
    sv = explainer.shap_values(X_arr)
    # TreeExplainer may return a list [licit_shap, illicit_shap] for binary
    # classification; we want the illicit class
    if isinstance(sv, list):
        return sv[1]
    return sv


def transaction_audit_record(
    tx_index: int,
    shap_values_row: np.ndarray,
    feature_cols: list[str],
    confidence_score: float,
    predicted_label: int,
    true_label: int | None = None,
    tx_id: str | None = None,
    top_k: int = 10,
    model_version: str = "xgb-fp-optimised-v1",
) -> dict[str, Any]:
    """
    Build a single FINRA-aligned audit record for one transaction decision.

    The JSON schema is designed to satisfy:
    - FINRA Rule 4511: books-and-records documentation of automated alerts
    - SEC Rule 17a-4: machine-readable audit record preservation
    - FinCEN SAR narrative support: feature-level evidence for report writing

    Parameters
    ----------
    top_k:
        Number of top contributing features to include in the audit record.
    """
    # Sort features by absolute SHAP contribution
    sorted_idx = np.argsort(-np.abs(shap_values_row))[:top_k]

    attribution = []
    for rank, i in enumerate(sorted_idx, start=1):
        contribution = float(shap_values_row[i])
        attribution.append({
            "rank": rank,
            "feature_name": feature_cols[i],
            "shap_contribution": round(contribution, 6),
            "direction": "increases_illicit_risk" if contribution > 0 else "decreases_illicit_risk",
        })

    decision = "FLAGGED_ILLICIT" if predicted_label == 1 else "CLEARED_LICIT"
    fincen_sar_trigger = predicted_label == 1 and confidence_score >= 0.80

    record: dict[str, Any] = {
        "schema_version": "1.0",
        "tx_index": int(tx_index),
        "tx_id": tx_id or f"tx_{tx_index:08d}",
        "decision": decision,
        "confidence_score": round(float(confidence_score), 6),
        "fincen_sar_trigger": fincen_sar_trigger,
        "model_version": model_version,
        "shap_attribution": attribution,
        "audit_metadata": {
            "regulatory_framework": [
                "FS AI RMF (Feb 2026) — Pillar 4: Incident Response",
                "FINRA Rule 4511 — Books and Records (General Requirements)",
                "SEC Rule 17a-4 — Records Preservation",
                "FinCEN SAR Requirements — 31 CFR 1020.320",
            ],
            "explainability_method": "SHAP TreeExplainer (Lundberg & Lee, 2017)",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    }

    if true_label is not None:
        correct = int(predicted_label == true_label)
        record["ground_truth"] = {
            "true_label": int(true_label),
            "correct_decision": bool(correct),
            "error_type": (
                None if correct
                else ("false_positive" if predicted_label == 1 else "false_negative")
            ),
        }

    return record


def generate_audit_batch(
    clf: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series | None,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    feature_cols: list[str],
    top_k: int = 10,
    max_records: int = 500,
    model_version: str = "xgb-fp-optimised-v1",
) -> list[dict]:
    """
    Generate FINRA-aligned audit records for a test batch.

    In production, this runs after every scoring batch. For the study, we
    generate records for up to max_records transactions (all flagged + a
    random sample of cleared, to keep the output tractable).
    """
    explainer = build_explainer(clf)
    shap_vals = compute_shap_values(explainer, X_test, feature_cols)

    flagged_idx = np.where(y_pred == 1)[0]
    cleared_idx = np.where(y_pred == 0)[0]

    # Include all flagged; sample cleared to balance the audit log
    n_cleared_sample = min(len(cleared_idx), max(len(flagged_idx), 50))
    rng = np.random.default_rng(42)
    cleared_sample = rng.choice(cleared_idx, size=n_cleared_sample, replace=False)
    selected = np.sort(np.concatenate([flagged_idx, cleared_sample]))

    records = []
    y_true_arr = y_test.values if y_test is not None else None

    for i in selected:
        true_label = int(y_true_arr[i]) if y_true_arr is not None else None
        record = transaction_audit_record(
            tx_index=int(i),
            shap_values_row=shap_vals[i],
            feature_cols=feature_cols,
            confidence_score=float(y_score[i]),
            predicted_label=int(y_pred[i]),
            true_label=true_label,
            top_k=top_k,
            model_version=model_version,
        )
        records.append(record)

    return records


def save_audit_log(records: list[dict], output_path: str | Path) -> None:
    """Persist audit records as newline-delimited JSON (one record per line)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def false_positive_audit_summary(records: list[dict]) -> pd.DataFrame:
    """
    Summarise false-positive records — the primary input for SAR triage.

    Returns a DataFrame of transactions predicted illicit but actually licit,
    ranked by confidence score descending (highest-confidence FPs first — these
    are the model's most costly mistakes for compliance analysts).
    """
    fps = [
        r for r in records
        if r.get("ground_truth", {}).get("error_type") == "false_positive"
    ]
    if not fps:
        return pd.DataFrame()

    rows = []
    for r in fps:
        top_feature = r["shap_attribution"][0] if r["shap_attribution"] else {}
        rows.append({
            "tx_id": r["tx_id"],
            "confidence_score": r["confidence_score"],
            "top_shap_feature": top_feature.get("feature_name", ""),
            "top_shap_contribution": top_feature.get("shap_contribution", 0.0),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("confidence_score", ascending=False)
        .reset_index(drop=True)
    )
