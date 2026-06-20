"""
Observability and evaluation harness for AML model monitoring.

Implements three monitoring concerns that align with FS AI RMF Pillar 4
(Incident Response) and NIST AI RMF Measure function:

  1. Feature distribution drift — Kolmogorov-Smirnov test comparing training
     and test-time distributions per time step. Detects when the transaction
     population has shifted enough to warrant model re-evaluation.

  2. Probability calibration — Brier score and Expected Calibration Error (ECE).
     A well-calibrated model's confidence_score is interpretable as a true
     probability of illicit activity, which is required for SAR triage workflows
     where analysts use score thresholds to prioritise case review queues.

  3. Temporal performance degradation — tracks FPR and recall across the 15
     test time steps to surface model staleness before it reaches a threshold
     that would require regulatory disclosure under FS AI RMF Pillar 4.

These monitoring checks are the "eval trigger" in the 14-step architecture
described in the main README (step 14: eval trigger → ECLIPSE feedback loop).
In an LLM-augmented workflow, the same drift signal that triggers AML model
re-evaluation also triggers a re-run of the hallucination detection suite on
any LLM-generated STR narratives from that period.

Regulatory alignment:
  FS AI RMF (Feb 2026) — Pillar 4 (Incident Response): institutions must
  document and respond to model performance degradation.
  NIST AI RMF 1.0 — Measure function: requires ongoing monitoring of AI
  system outputs and error rates.
  FSOC 2023 Annual Report: "model opacity" cited as systemic risk — calibration
  checks directly address opacity by making confidence scores auditable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


# ---------------------------------------------------------------------------
# 1. Feature distribution drift
# ---------------------------------------------------------------------------

def ks_drift_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: list[str],
    pvalue_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Kolmogorov-Smirnov test for feature drift between train and test.

    Features with p-value < pvalue_threshold are flagged as drifted.
    In production, a high drift count triggers a re-evaluation event.

    Returns
    -------
    pd.DataFrame with columns: feature, ks_statistic, p_value, drifted
    """
    records = []
    for col in feature_cols:
        result = stats.ks_2samp(X_train[col].values, X_test[col].values)
        records.append({
            "feature": col,
            "ks_statistic": round(float(result.statistic), 6),
            "p_value": round(float(result.pvalue), 6),
            "drifted": result.pvalue < pvalue_threshold,
        })
    return pd.DataFrame(records).sort_values("ks_statistic", ascending=False)


def drift_summary(drift_df: pd.DataFrame) -> dict:
    """High-level drift report for the monitoring dashboard."""
    n_features = len(drift_df)
    n_drifted = int(drift_df["drifted"].sum())
    return {
        "n_features_tested": n_features,
        "n_features_drifted": n_drifted,
        "drift_fraction": round(n_drifted / max(n_features, 1), 4),
        "top_drifted_features": drift_df[drift_df["drifted"]]["feature"].tolist()[:10],
        "alert": n_drifted > n_features * 0.20,  # >20% drifted → trigger re-eval
    }


# ---------------------------------------------------------------------------
# 2. Probability calibration
# ---------------------------------------------------------------------------

def calibration_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute Brier score and Expected Calibration Error (ECE).

    A Brier score near 0 means confidence scores are accurate probability
    estimates; ECE measures the average gap between predicted probabilities
    and empirical frequencies. Both are required for SAR triage integrity.
    """
    brier = brier_score_loss(y_true, y_score)

    fraction_pos, mean_pred_value = calibration_curve(
        y_true, y_score, n_bins=n_bins, strategy="uniform"
    )
    ece = float(np.mean(np.abs(fraction_pos - mean_pred_value)))

    return {
        "brier_score": round(float(brier), 6),
        "expected_calibration_error": round(ece, 6),
        "calibration_bins": n_bins,
        "calibration_fraction_positive": fraction_pos.tolist(),
        "calibration_mean_predicted": mean_pred_value.tolist(),
    }


# ---------------------------------------------------------------------------
# 3. Temporal performance monitoring
# ---------------------------------------------------------------------------

def temporal_performance(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred_baseline: np.ndarray,
    y_pred_hybrid: np.ndarray,
) -> pd.DataFrame:
    """
    FPR and recall per time step for both models — shows temporal stability.

    A rising FPR over time is the canonical signal that model retraining
    is needed. This plot is the primary visual in the results section.
    """
    from .baseline import false_positive_rate, metrics_per_timestep
    from sklearn.metrics import recall_score

    records = []
    for ts in sorted(X_test["time_step"].unique()):
        mask = (X_test["time_step"] == ts).values
        y_t = y_test.values[mask]
        y_b = y_pred_baseline[mask]
        y_h = y_pred_hybrid[mask]

        if len(np.unique(y_t)) < 2:
            continue

        records.append({
            "time_step": int(ts),
            "fpr_baseline": round(false_positive_rate(y_t, y_b), 4),
            "fpr_hybrid": round(false_positive_rate(y_t, y_h), 4),
            "recall_baseline": round(recall_score(y_t, y_b, zero_division=0), 4),
            "recall_hybrid": round(recall_score(y_t, y_h, zero_division=0), 4),
        })

    df = pd.DataFrame(records)
    df["fpr_reduction"] = df["fpr_baseline"] - df["fpr_hybrid"]
    return df


# ---------------------------------------------------------------------------
# LLM eval tie-in (Goal 2 narrative)
# ---------------------------------------------------------------------------

def llm_eval_alignment_note() -> str:
    """
    Returns a documentation string explaining how the AML eval trigger
    connects to the LLM hallucination detection layer (Goal 2).

    Included in the notebook as a prose section to cover the observability
    angle for the NIW write-up.
    """
    return """
LLM Eval Trigger — Connection to FS AI RMF Pillar 4
=====================================================

When the drift_summary() function raises an alert (>20% of features drifted),
or when temporal_performance() shows FPR rising >5 percentage points across
consecutive time steps, the production system triggers two parallel actions:

1. AML model re-evaluation:
   A retraining job is enqueued in the MLflow registry (step 13 in the
   architecture). Until the new model is validated, the existing model
   continues serving but all FLAGGED decisions above threshold are routed
   to a human review queue rather than auto-generating STRs.

2. LLM hallucination re-evaluation (the FS AI RMF Pillar 4 feedback loop):
   Any LLM-generated STR narrative produced during the drifted period is
   flagged for re-review. An ECLIPSE-style hallucination check (semantic
   entropy + perplexity decomposition, arXiv:2512.03107) is re-run against
   the ground-truth regulatory document corpus. This prevents a scenario
   where a drifted AML model produces unusual transaction descriptions that
   cause the LLM to hallucinate regulatory citations in the STR narrative.

This two-stage eval trigger — statistical drift detection on the ML model
coupled with hallucination re-evaluation on the LLM layer — operationalises
FS AI RMF Pillar 4 control objective requirements for continuous monitoring
of AI systems in regulated financial environments.

The pattern is novel: existing AML monitoring frameworks track only model
performance metrics. The dual trigger extends monitoring to include the
downstream LLM outputs that are increasingly being used to draft regulatory
filings — a risk that the FS AI RMF (Feb 2026) explicitly names but for
which no open-source implementation currently exists.
"""
