"""
Production-proxy validation: PaySim mobile-money fraud dataset.

Purpose: extend external validity beyond Bitcoin (Elliptic). PaySim is a
synthetic mobile-money simulator (Lopez-Rojas et al., 2016) — tabular,
balance-sheet features, traditional banking domain, public benchmark.

Protocol mirrors the Elliptic study:
  - Temporal split (train steps 1-500, test 501-744)
  - Same XGBoost baseline + hybrid FP-reduction pipeline
  - Same metrics: FPR, recall, precision, F1, AUC
  - 5 runs, SD/CI reported

Run from: financial-llm-governance/aml-detection/
  ../../.conda/python.exe paysim_experiment.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from src.baseline import full_metrics, false_positive_rate, false_negative_rate
from src.hybrid_pipeline import run_hybrid_pipeline

warnings.filterwarnings("ignore")

PAYSIM_DIR = Path(__file__).parent / "data" / "paysim"
RESULTS_FILE = Path(__file__).parent / "results" / "full_results.json"
TRAIN_CUTOFF = 500   # steps 1-500 train, 501-744 test (matches Elliptic ~70/30 ratio)
N_RUNS = 5
RECALL_FLOOR = 0.65


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_paysim(data_dir: Path) -> pd.DataFrame:
    dfs = []
    for p in sorted(data_dir.glob("*.parquet")):
        dfs.append(pd.read_parquet(p))
    df = pd.concat(dfs, ignore_index=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric feature matrix from PaySim columns."""
    # One-hot encode transaction type
    type_dummies = pd.get_dummies(df["type"], prefix="type")

    # Balance-change features
    df = df.copy()
    df["amount_log"] = np.log1p(df["amount"])
    df["orig_balance_change"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["dest_balance_change"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["orig_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["dest_balance_ratio"] = df["amount"] / (df["oldbalanceDest"] + 1)
    df["orig_zeroed"] = (df["newbalanceOrig"] == 0).astype(int)
    df["dest_zeroed"] = (df["newbalanceDest"] == 0).astype(int)

    numeric_cols = [
        "amount", "amount_log",
        "oldbalanceOrg", "newbalanceOrig", "orig_balance_change",
        "oldbalanceDest", "newbalanceDest", "dest_balance_change",
        "orig_balance_ratio", "dest_balance_ratio",
        "orig_zeroed", "dest_zeroed",
    ]
    feat = pd.concat([df[numeric_cols].reset_index(drop=True),
                      type_dummies.reset_index(drop=True)], axis=1)
    feat["step"] = df["step"].values
    return feat


def temporal_split(feat: pd.DataFrame, y: pd.Series):
    train_mask = feat["step"] <= TRAIN_CUTOFF
    test_mask  = feat["step"] >  TRAIN_CUTOFF
    feat_cols = [c for c in feat.columns if c != "step"]
    X_train = feat[train_mask][feat_cols].reset_index(drop=True)
    X_test  = feat[test_mask ][feat_cols].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)
    y_test  = y[test_mask ].reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def sd_ci(values):
    arr = np.array(values, dtype=float)
    std = float(np.std(arr, ddof=1))
    ci95 = 1.96 * std / np.sqrt(len(arr))
    return round(std, 4), round(ci95, 4)


# ---------------------------------------------------------------------------
# Baseline (XGBoost, threshold=0.5)
# ---------------------------------------------------------------------------

def run_baseline_paysim(X_train, y_train, X_test, y_test, n_runs=5, threshold=0.5):
    feat_cols = list(X_train.columns)
    X_tr = X_train[feat_cols].values
    X_te = X_test[feat_cols].values
    y_tr = y_train.values
    y_te = y_test.values

    run_metrics = []
    for seed in range(n_runs):
        clf = XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            eval_metric="logloss", random_state=seed, verbosity=0,
        )
        clf.fit(X_tr, y_tr)
        y_score = clf.predict_proba(X_te)[:, 1]
        y_pred = (y_score >= threshold).astype(int)
        run_metrics.append(full_metrics(y_te, y_pred, y_score))

    avg = {k: round(float(np.mean([m[k] for m in run_metrics])), 4)
           for k in run_metrics[0] if isinstance(run_metrics[0][k], (int, float))}
    avg["threshold"] = threshold
    avg["n_runs"] = n_runs

    # Add SD/CI
    for metric in ["precision", "recall", "f1_illicit", "false_positive_rate",
                   "false_negative_rate", "auc_roc"]:
        vals = [m[metric] for m in run_metrics if metric in m]
        if vals:
            std, ci = sd_ci(vals)
            avg[f"sd_{metric}"] = std
            avg[f"ci95_{metric}"] = ci

    return {"avg_metrics": avg, "per_run_metrics": run_metrics}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading PaySim dataset...")
    df = load_paysim(PAYSIM_DIR)
    print(f"  Rows: {len(df):,} | Fraud prevalence: {df['isFraud'].mean()*100:.2f}%")

    # Only TRANSFER and CASH_OUT transactions are labeled fraud in PaySim
    df_labeled = df[df["isFraud"].isin([0, 1])].copy()
    y_all = df_labeled["isFraud"].astype(int)

    feat = engineer_features(df_labeled)
    X_train, X_test, y_train, y_test = temporal_split(feat, y_all)

    print(f"  Train: {len(X_train):,} (illicit: {y_train.sum():,}) | "
          f"Test: {len(X_test):,} (illicit: {y_test.sum():,})")
    print(f"  Train steps: 1-{TRAIN_CUTOFF} | Test steps: {TRAIN_CUTOFF+1}-744")

    # Baseline
    print(f"\nRunning XGBoost baseline ({N_RUNS} runs)...")
    baseline = run_baseline_paysim(X_train, y_train, X_test, y_test, N_RUNS)
    bm = baseline["avg_metrics"]
    print(f"  FPR={bm['false_positive_rate']}  Recall={bm['recall']}  "
          f"Precision={bm['precision']}  F1={bm['f1_illicit']}  AUC={bm.get('auc_roc','n/a')}")

    # Hybrid
    print(f"\nRunning hybrid pipeline ({N_RUNS} runs, recall_floor={RECALL_FLOOR})...")

    # Build DataFrames with feature column names hybrid_pipeline expects
    X_train_df = pd.DataFrame(X_train.values, columns=X_train.columns)
    X_train_df["time_step"] = 1  # dummy: hybrid_pipeline needs time_step for feature_columns()
    X_test_df  = pd.DataFrame(X_test.values,  columns=X_test.columns)
    X_test_df["time_step"]  = 1

    # hybrid_pipeline's feature_columns() drops 'time_step' — so features are intact
    hybrid_result = run_hybrid_pipeline(
        X_train_df, y_train, X_test_df, y_test,
        recall_floor=RECALL_FLOOR,
        blend_isolation_forest=True,
        blend_weight=0.15,
        n_cv_splits=5,
        n_runs=N_RUNS,
    )
    hm = hybrid_result["avg_metrics"]
    print(f"  FPR={hm['false_positive_rate']}  Recall={hm['recall']}  "
          f"Precision={hm['precision']}  F1={hm['f1_illicit']}  AUC={hm.get('auc_roc','n/a')}")

    # Add SD/CI for hybrid too
    hybrid_runs = hybrid_result["per_run_metrics"]
    for metric in ["precision", "recall", "f1_illicit", "false_positive_rate",
                   "false_negative_rate", "auc_roc"]:
        vals = [m[metric] for m in hybrid_runs if metric in m]
        if vals:
            std, ci = sd_ci(vals)
            hm[f"sd_{metric}"] = std
            hm[f"ci95_{metric}"] = ci

    # Delta
    fpr_delta = round((hm['false_positive_rate'] - bm['false_positive_rate']) /
                      max(bm['false_positive_rate'], 1e-9) * 100, 1)
    print(f"\n  FPR delta: {fpr_delta:+.1f}% relative")

    # Save
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    results["paysim_baseline"] = {**bm, "dataset": "PaySim", "n_train": int(len(X_train)),
                                   "n_test": int(len(X_test)), "per_run_metrics": baseline["per_run_metrics"]}
    results["paysim_hybrid"]   = {**hm, "dataset": "PaySim", "n_train": int(len(X_train)),
                                   "n_test": int(len(X_test)), "per_run_metrics": hybrid_runs}

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nUpdated {RESULTS_FILE} with paysim_baseline + paysim_hybrid.")
    print("\n=== CROSS-DATASET SUMMARY ===")
    print(f"{'Dataset+Model':<30} {'FPR':>8} {'Recall':>8} {'F1':>8}")
    for label, m in [
        ("Elliptic baseline", results["baseline"]),
        ("Elliptic hybrid", results["hybrid"]),
        ("PaySim baseline", results["paysim_baseline"]),
        ("PaySim hybrid", results["paysim_hybrid"]),
    ]:
        print(f"{label:<30} {m.get('false_positive_rate','n/a'):>8} "
              f"{m.get('recall','n/a'):>8} {m.get('f1_illicit','n/a'):>8}")


if __name__ == "__main__":
    main()
