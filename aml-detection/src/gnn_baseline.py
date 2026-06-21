"""
Graph Neural Network baseline for the Elliptic AML dataset.

The Elliptic dataset is fundamentally a transaction graph, and the original
Weber et al. (2019) paper as well as the Feedzai (2021) study both benchmark a
Graph Convolutional Network. This module adds a GNN comparison point to our
study, evaluated under the *same* rules as the tabular models so the comparison
is apples-to-apples:

  - Strict temporal split: train on time steps 1-34, test on 35-49.
  - Illicit-class-specific metrics (precision/recall/F1 on the illicit class)
    plus the operationally binding false-positive rate (FPR).
  - The same recall-floor threshold optimisation used by the hybrid pipeline,
    so we can ask: does graph structure alone reduce FPR, or is cost-sensitive
    thresholding the thing that actually drives it down?

This deliberately contrasts with common public Elliptic notebooks that report
inflated numbers via random train/test splits (temporal leakage) or
majority-class-weighted / flipped-label metrics. See the "Common Evaluation
Pitfalls" section of the study notebook.

Regulatory alignment:
  FS AI RMF (Feb 2026) - Pillar 2 (Risk Identification): comparing model
  families under a common, leakage-free protocol is part of documenting model
  error rates honestly.

Models implemented: GCN (Kipf & Welling, 2017), GAT (Velickovic et al., 2018).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv

from .baseline import full_metrics
from .data_loader import _DEFAULT_DATA_DIR, _CLASS_MAP, feature_columns
from .hybrid_pipeline import _optimal_threshold


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def load_graph_data(data_dir: Path | str | None = None) -> dict:
    """Load the Elliptic dataset as a PyTorch Geometric graph.

    Returns a dict with:
      data        : torch_geometric.data.Data (x, edge_index, y)
      time_step   : np.ndarray of per-node time steps (1-49)
      labeled     : np.ndarray bool mask of nodes with a known label
      feat_cols   : list of feature column names (165, excludes time_step)
    Labels: 1 = illicit, 0 = licit, -1 = unknown (masked out of train/test).
    Node features are the same 165 columns the tabular models use (time_step
    is used only for splitting, not as a feature).
    """
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR

    df_features = pd.read_csv(data_dir / "elliptic_txs_features.csv", header=None)
    df_classes = pd.read_csv(data_dir / "elliptic_txs_classes.csv")
    df_edges = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")

    df_features.columns = (
        ["id", "time_step"]
        + [f"trans_feat_{i}" for i in range(93)]
        + [f"agg_feat_{i}" for i in range(72)]
    )
    df_classes["class"] = df_classes["class"].map(_CLASS_MAP)

    df = df_features.merge(df_classes, left_on="id", right_on="txId", how="left")
    df.drop(columns=["txId"], inplace=True)

    feat_cols = [c for c in df.columns if c not in ("id", "time_step", "class")]
    assert len(feat_cols) == 165, f"expected 165 features, got {len(feat_cols)}"

    # Node feature matrix and labels (unknown -> -1).
    x = torch.tensor(df[feat_cols].values, dtype=torch.float)
    y_raw = df["class"].values
    y = torch.tensor(np.where(np.isnan(y_raw), -1, y_raw), dtype=torch.long)
    time_step = df["time_step"].values.astype(int)

    # Map txId -> node index, build (undirected) edge_index.
    id_to_idx = {tx: i for i, tx in enumerate(df["id"].values)}
    src = df_edges["txId1"].map(id_to_idx)
    dst = df_edges["txId2"].map(id_to_idx)
    valid = src.notna() & dst.notna()
    src = src[valid].astype(int).values
    dst = dst[valid].astype(int).values
    # Undirected: add reverse edges for symmetric message passing.
    edge_index = torch.tensor(
        np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]),
        dtype=torch.long,
    )

    data = Data(x=x, edge_index=edge_index, y=y)
    return {
        "data": data,
        "time_step": time_step,
        "labeled": (y_raw == 0) | (y_raw == 1),
        "feat_cols": feat_cols,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class GCN(torch.nn.Module):
    """Two-layer Graph Convolutional Network (Kipf & Welling, 2017)."""

    def __init__(self, in_features: int, hidden: int = 128, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden)
        self.conv2 = GCNConv(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # logits


class GAT(torch.nn.Module):
    """Two-layer Graph Attention Network (Velickovic et al., 2018)."""

    def __init__(self, in_features: int, hidden: int = 16, num_classes: int = 2, heads: int = 8, dropout: float = 0.4):
        super().__init__()
        self.conv1 = GATConv(in_features, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, num_classes, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # logits


_MODELS = {"gcn": GCN, "gat": GAT}


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def _make_masks(time_step: np.ndarray, labeled: np.ndarray, val_frac: float, seed: int):
    """Train (loss) / val (threshold tuning) / test masks.

    train+val are labeled nodes in time steps 1-34; test is labeled nodes in
    35-49. val is a random hold-out within the training period, used ONLY to
    pick the recall-floor threshold (never in the loss) to avoid leakage.
    """
    n = len(time_step)
    rng = np.random.default_rng(seed)

    train_period = labeled & (time_step >= 1) & (time_step <= 34)
    test_period = labeled & (time_step >= 35) & (time_step <= 49)

    train_idx = np.where(train_period)[0]
    val_count = int(len(train_idx) * val_frac)
    val_sel = rng.choice(train_idx, size=val_count, replace=False)

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    train_mask[val_sel] = False  # carve val out of the loss set
    val_mask[val_sel] = True
    test_mask[test_period] = True

    return (
        torch.tensor(train_mask),
        torch.tensor(val_mask),
        torch.tensor(test_mask),
    )


def run_gnn_baseline(
    data_dir: Path | str | None = None,
    model_type: str = "gcn",
    hidden: int | None = None,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    cost_sensitive: bool = True,
    recall_floor: float = 0.65,
    val_frac: float = 0.2,
    n_runs: int = 3,
    graph: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Train a GNN under the study's temporal + FPR/recall-floor protocol.

    Returns a dict with averaged metrics at the default 0.5 threshold and at
    the recall-floor-optimised threshold, plus per-run detail and mean test
    scores for downstream plots.
    """
    if model_type not in _MODELS:
        raise ValueError(f"model_type must be one of {list(_MODELS)}")

    graph = graph or load_graph_data(data_dir)
    data = graph["data"]
    time_step = graph["time_step"]
    labeled = graph["labeled"]
    in_features = data.x.shape[1]

    if hidden is None:
        hidden = 128 if model_type == "gcn" else 16

    # Cost-sensitive class weights (inverse frequency) on the training labels.
    default_metrics_runs, floor_metrics_runs, thresholds = [], [], []
    test_scores_runs = []
    test_mask_ref = None
    y_test_ref = None

    for seed in range(n_runs):
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_mask, val_mask, test_mask = _make_masks(time_step, labeled, val_frac, seed)
        test_mask_ref = test_mask

        y_tr = data.y[train_mask]
        if cost_sensitive:
            n_pos = int((y_tr == 1).sum())
            n_neg = int((y_tr == 0).sum())
            # weight illicit (class 1) up by the licit/illicit ratio
            w = torch.tensor([1.0, max(n_neg / max(n_pos, 1), 1.0)], dtype=torch.float)
        else:
            w = None

        model = _MODELS[model_type](in_features, hidden=hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        model.train()
        for ep in range(epochs):
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = F.cross_entropy(logits[train_mask], data.y[train_mask], weight=w)
            loss.backward()
            optimizer.step()

        # Inference (transductive: one forward pass scores every node).
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probs = F.softmax(logits, dim=1)[:, 1].numpy()  # illicit-class prob

        y_val = data.y[val_mask].numpy()
        y_test = data.y[test_mask].numpy()
        val_scores = probs[val_mask.numpy()]
        test_scores = probs[test_mask.numpy()]
        y_test_ref = y_test
        test_scores_runs.append(test_scores)

        # Default 0.5 threshold.
        y_pred_default = (test_scores >= 0.5).astype(int)
        default_metrics_runs.append(full_metrics(y_test, y_pred_default, test_scores))

        # Recall-floor threshold tuned on the held-out validation slice.
        t_opt = _optimal_threshold(y_val, val_scores, recall_floor)
        thresholds.append(t_opt)
        y_pred_floor = (test_scores >= t_opt).astype(int)
        floor_metrics_runs.append(full_metrics(y_test, y_pred_floor, test_scores))

        if verbose:
            dm = default_metrics_runs[-1]
            fm = floor_metrics_runs[-1]
            print(
                f"  [{model_type}] run {seed}: "
                f"default FPR={dm['false_positive_rate']:.4f} F1={dm['f1_illicit']:.4f} | "
                f"recall-floor(t={t_opt:.3f}) FPR={fm['false_positive_rate']:.4f} "
                f"recall={fm['recall']:.4f} F1={fm['f1_illicit']:.4f}"
            )

    def _avg(runs):
        return {
            k: round(float(np.mean([m[k] for m in runs])), 4)
            for k in runs[0]
            if isinstance(runs[0][k], (int, float))
        }

    avg_default = _avg(default_metrics_runs)
    avg_floor = _avg(floor_metrics_runs)
    avg_floor["avg_optimal_threshold"] = round(float(np.mean(thresholds)), 4)
    avg_floor["recall_floor_constraint"] = recall_floor

    return {
        "model_type": model_type,
        "cost_sensitive": cost_sensitive,
        "avg_metrics_default": avg_default,
        "avg_metrics_recall_floor": avg_floor,
        "per_run_default": default_metrics_runs,
        "per_run_recall_floor": floor_metrics_runs,
        "optimal_thresholds": thresholds,
        "y_test": y_test_ref,
        "mean_test_scores": np.mean(test_scores_runs, axis=0),
        "test_time_step": time_step[test_mask_ref.numpy()],
    }
