"""
Ablation study: decompose the hybrid pipeline's FPR reduction into its
three components. Answers the attribution question the paper must support:
does the Isolation Forest blend contribute to the FPR reduction, or does
cost-sensitive weighting + CV threshold optimisation account for it alone?

Arms (all on the same temporal split, 5 seeds each):
  A. baseline          — plain XGBoost, threshold 0.5 (replicates Table 2 baseline)
  B. cost_weight_only  — + scale_pos_weight, threshold 0.5
  C. threshold_no_blend— + CV threshold optimisation, blend OFF
  D. full_hybrid       — + IF blend inside CV folds (replicates Table 2 hybrid)

Writes results/ablation_blend.json (never touches full_results.json).

Run from: financial-llm-governance/aml-detection/
  python ablation_blend.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))

from src.baseline import run_baseline, full_metrics
from src.data_loader import load_elliptic, temporal_split, feature_columns
from src.hybrid_pipeline import run_hybrid_pipeline

DATA_DIR = Path(__file__).parent / "data" / "elliptic" / "dataset"
OUT_FILE = Path(__file__).parent / "results" / "ablation_blend.json"
N_RUNS = 5
RECALL_FLOOR = 0.65
BLEND_WEIGHT = 0.15
N_CV_SPLITS = 5


def summarize(per_run: list[dict]) -> dict:
    keys = [k for k, v in per_run[0].items() if isinstance(v, (int, float))]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in per_run], dtype=float)
        out[k] = round(float(vals.mean()), 4)
        out[f"sd_{k}"] = round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else 0.0
    return out


def run_cost_weight_only(X_train, y_train, X_test, y_test) -> dict:
    feat_cols = feature_columns(X_train)
    X_tr, X_te = X_train[feat_cols].values, X_test[feat_cols].values
    y_tr, y_te = y_train.values, y_test.values
    spw = round(int((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1), 2)

    per_run = []
    for seed in range(N_RUNS):
        clf = XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=seed, verbosity=0,
        )
        clf.fit(X_tr, y_tr)
        y_score = clf.predict_proba(X_te)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        per_run.append(full_metrics(y_te, y_pred, y_score))
    return {"avg_metrics": summarize(per_run), "per_run_metrics": per_run,
            "scale_pos_weight": spw, "threshold": 0.5}


def main():
    print("Loading Elliptic dataset...")
    X, y, _ = load_elliptic(DATA_DIR)
    X_train, X_test, y_train, y_test = temporal_split(X, y)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    results = {"config": {
        "n_runs": N_RUNS, "recall_floor": RECALL_FLOOR,
        "blend_weight": BLEND_WEIGHT, "n_cv_splits": N_CV_SPLITS,
        "split": "temporal 1-34 / 35-49",
    }}

    print("\n[A] baseline (plain XGB, t=0.5)...")
    base = run_baseline(X_train, y_train, X_test, y_test, n_runs=N_RUNS, threshold=0.5)
    results["A_baseline"] = {"avg_metrics": base["avg_metrics"],
                             "per_run_metrics": base["per_run_metrics"]}

    print("[B] cost-weight only (spw, t=0.5)...")
    results["B_cost_weight_only"] = run_cost_weight_only(X_train, y_train, X_test, y_test)

    print("[C] cost-weight + CV threshold, blend OFF...")
    no_blend = run_hybrid_pipeline(
        X_train, y_train, X_test, y_test,
        recall_floor=RECALL_FLOOR, blend_isolation_forest=False,
        n_cv_splits=N_CV_SPLITS, n_runs=N_RUNS,
    )
    results["C_threshold_no_blend"] = {
        "avg_metrics": no_blend["avg_metrics"],
        "per_run_metrics": no_blend["per_run_metrics"],
        "optimal_thresholds": no_blend["optimal_thresholds"],
    }

    print("[D] full hybrid (blend ON)...")
    full = run_hybrid_pipeline(
        X_train, y_train, X_test, y_test,
        recall_floor=RECALL_FLOOR, blend_isolation_forest=True,
        blend_weight=BLEND_WEIGHT, n_cv_splits=N_CV_SPLITS, n_runs=N_RUNS,
    )
    results["D_full_hybrid"] = {
        "avg_metrics": full["avg_metrics"],
        "per_run_metrics": full["per_run_metrics"],
        "optimal_thresholds": full["optimal_thresholds"],
    }

    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {OUT_FILE}\n")
    hdr = f"{'arm':<24}{'FPR':>8}{'recall':>8}{'prec':>8}{'F1':>8}{'AUC':>8}{'FP':>6}"
    print(hdr)
    for arm in ["A_baseline", "B_cost_weight_only", "C_threshold_no_blend", "D_full_hybrid"]:
        m = results[arm]["avg_metrics"]
        print(f"{arm:<24}{m['false_positive_rate']:>8.4f}{m['recall']:>8.4f}"
              f"{m['precision']:>8.4f}{m['f1_illicit']:>8.4f}{m['auc_roc']:>8.4f}"
              f"{m.get('FP', float('nan')):>6}")


if __name__ == "__main__":
    main()
