"""
Compute per-run SD and 95% CI for baseline and hybrid pipeline.
Updates full_results.json in-place with sd_ and ci95_ fields.
Run from: financial-llm-governance/aml-detection/
  python compute_sd_ci.py
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add aml-detection to path so src imports work
sys.path.insert(0, str(Path(__file__).parent))

from src.baseline import run_baseline
from src.data_loader import load_elliptic, temporal_split
from src.hybrid_pipeline import run_hybrid_pipeline

RESULTS_FILE = Path(__file__).parent / "results" / "full_results.json"
DATA_DIR = Path(__file__).parent / "data" / "elliptic" / "dataset"
N_RUNS = 5
METRICS_OF_INTEREST = [
    "precision", "recall", "f1_illicit",
    "false_positive_rate", "false_negative_rate", "auc_roc",
]


def sd_ci(values: list[float]) -> tuple[float, float]:
    """Return (std, half-width of 95% CI) for a list of run values."""
    arr = np.array(values, dtype=float)
    std = float(np.std(arr, ddof=1))  # sample std
    ci95 = 1.96 * std / np.sqrt(len(arr))
    return round(std, 4), round(ci95, 4)


def main():
    print("Loading Elliptic dataset...")
    X, y, _ = load_elliptic(DATA_DIR)
    X_train, X_test, y_train, y_test = temporal_split(X, y)

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Train illicit: {y_train.sum()} | Test illicit: {y_test.sum()}")

    # --- Baseline ---
    print(f"\nRunning XGBoost baseline ({N_RUNS} runs)...")
    baseline_result = run_baseline(X_train, y_train, X_test, y_test, n_runs=N_RUNS, threshold=0.5)
    baseline_runs = baseline_result["per_run_metrics"]

    # --- Hybrid ---
    print(f"Running hybrid pipeline ({N_RUNS} runs)...")
    hybrid_result = run_hybrid_pipeline(
        X_train, y_train, X_test, y_test,
        recall_floor=0.65,
        blend_isolation_forest=True,
        blend_weight=0.15,
        n_cv_splits=5,
        n_runs=N_RUNS,
    )
    hybrid_runs = hybrid_result["per_run_metrics"]

    # --- Build SD/CI blocks ---
    baseline_sd = {}
    hybrid_sd = {}
    print("\n=== Baseline SD/CI ===")
    for m in METRICS_OF_INTEREST:
        vals_b = [r[m] for r in baseline_runs if m in r]
        vals_h = [r[m] for r in hybrid_runs if m in r]
        if vals_b:
            std, ci = sd_ci(vals_b)
            baseline_sd[f"sd_{m}"] = std
            baseline_sd[f"ci95_{m}"] = ci
            print(f"  {m}: mean={np.mean(vals_b):.4f}  SD={std}  95%CI\xb1{ci}")
        if vals_h:
            std, ci = sd_ci(vals_h)
            hybrid_sd[f"sd_{m}"] = std
            hybrid_sd[f"ci95_{m}"] = ci

    print("\n=== Hybrid SD/CI ===")
    for m in METRICS_OF_INTEREST:
        if f"sd_{m}" in hybrid_sd:
            print(f"  {m}: SD={hybrid_sd[f'sd_{m}']}  95%CI\xb1{hybrid_sd[f'ci95_{m}']}")

    # --- Update full_results.json ---
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    results["baseline"].update(baseline_sd)
    results["hybrid"].update(hybrid_sd)
    results["baseline"]["per_run_metrics"] = baseline_runs
    results["hybrid"]["per_run_metrics"] = hybrid_runs

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nUpdated {RESULTS_FILE}")
    print("SD/CI fields written for baseline and hybrid.")


if __name__ == "__main__":
    main()
