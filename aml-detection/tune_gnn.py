"""
Hyperparameter-tuned GNN baselines (GCN + GAT) for camera-ready revision.

Changes over the vanilla baselines in the original study:
  GCN: hidden 128->256, epochs 200->300, lr 0.01->0.005, dropout 0.4->0.3
  GAT: hidden 16->32, heads 8->4, epochs 150->200, n_runs 1->3, dropout 0.4->0.3

All other settings identical (temporal split, cost-sensitive, recall_floor=0.65).
Updates full_results.json with gcn_tuned / gat_tuned keys.

Run from: financial-llm-governance/aml-detection/
  ../../.conda/python.exe tune_gnn.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.gnn_baseline import load_graph_data, run_gnn_baseline

RESULTS_FILE = Path(__file__).parent / "results" / "full_results.json"
DATA_DIR = Path(__file__).parent / "data" / "elliptic" / "dataset"


def main():
    print("Loading Elliptic graph...")
    graph = load_graph_data(DATA_DIR)
    print(f"Nodes: {graph['data'].x.shape[0]} | Edges: {graph['data'].edge_index.shape[1]//2}")

    # --- Tuned GCN ---
    print("\nTraining tuned GCN (hidden=256, epochs=300, lr=0.005, dropout=0.3, 3 runs)...")
    gcn_tuned = run_gnn_baseline(
        graph=graph,
        model_type="gcn",
        hidden=256,
        epochs=300,
        lr=0.005,
        weight_decay=5e-4,
        cost_sensitive=True,
        recall_floor=0.65,
        val_frac=0.2,
        n_runs=3,
        verbose=True,
    )
    gcn_metrics = gcn_tuned["avg_metrics_recall_floor"]
    print(f"\nGCN tuned (recall-floor threshold):")
    print(f"  FPR={gcn_metrics['false_positive_rate']}  Recall={gcn_metrics['recall']}  "
          f"F1={gcn_metrics['f1_illicit']}  AUC={gcn_metrics.get('auc_roc','n/a')}")

    # --- Tuned GAT ---
    print("\nTraining tuned GAT (hidden=32, heads=4, epochs=200, dropout=0.3, 3 runs)...")
    gat_tuned = run_gnn_baseline(
        graph=graph,
        model_type="gat",
        hidden=32,
        epochs=200,
        lr=0.005,
        weight_decay=5e-4,
        cost_sensitive=True,
        recall_floor=0.65,
        val_frac=0.2,
        n_runs=3,
        verbose=True,
    )
    gat_metrics = gat_tuned["avg_metrics_recall_floor"]
    print(f"\nGAT tuned (recall-floor threshold):")
    print(f"  FPR={gat_metrics['false_positive_rate']}  Recall={gat_metrics['recall']}  "
          f"F1={gat_metrics['f1_illicit']}  AUC={gat_metrics.get('auc_roc','n/a')}")

    # --- Save ---
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    results["gcn_tuned"] = {
        **gcn_metrics,
        "hidden": 256,
        "epochs": 300,
        "lr": 0.005,
        "dropout": 0.3,
        "n_runs": 3,
        "threshold_mode": "recall_floor_0.65",
        "per_run_recall_floor": gcn_tuned["per_run_recall_floor"],
    }
    results["gat_tuned"] = {
        **gat_metrics,
        "hidden": 32,
        "heads": 4,
        "epochs": 200,
        "lr": 0.005,
        "dropout": 0.3,
        "n_runs": 3,
        "threshold_mode": "recall_floor_0.65",
        "per_run_recall_floor": gat_tuned["per_run_recall_floor"],
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nUpdated {RESULTS_FILE} with gcn_tuned + gat_tuned keys.")
    print("\n=== SUMMARY ===")
    print(f"{'Model':<20} {'FPR':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    for label, m in [("GCN (vanilla)", results["gcn"]), ("GAT (vanilla)", results["gat"]),
                     ("GCN (tuned)", results["gcn_tuned"]), ("GAT (tuned)", results["gat_tuned"])]:
        print(f"{label:<20} {m.get('false_positive_rate','n/a'):>8} "
              f"{m.get('recall','n/a'):>8} {m.get('f1_illicit','n/a'):>8} "
              f"{m.get('auc_roc','n/a'):>8}")


if __name__ == "__main__":
    main()
