"""
Data loading and preprocessing for the Elliptic Bitcoin dataset.

Dataset source: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
Place the three CSV files in: data/elliptic/dataset/
  - elliptic_txs_features.csv
  - elliptic_txs_classes.csv
  - elliptic_txs_edgelist.csv

Dataset schema follows the format described in:
  Weber et al. (2019). Anti-Money Laundering in Bitcoin: Experimenting with
  Graph Convolutional Networks for Financial Forensics. KDD Workshop.
  https://arxiv.org/abs/1908.02591

Preprocessing approach adapted from:
  Lorenz, Silva & Aparício (2021). Machine learning methods to detect money
  laundering in the Bitcoin blockchain in the presence of label scarcity.
  https://arxiv.org/abs/2005.14635
  GitHub: https://github.com/feedzai/research-aml-elliptic
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Canonical dataset location relative to this repo's root
_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "elliptic" / "dataset"

# Encoding: 1=illicit, 2=licit, unknown → kept as NaN or filtered
_CLASS_MAP = {"1": 1, "2": 0, "unknown": np.nan}

# Temporal train/test boundary matching Feedzai (section 4.1)
LAST_TRAIN_TIMESTEP: int = 34
LAST_TIMESTEP: int = 49


def _load_csvs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    if not (data_dir / "elliptic_txs_features.csv").exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}.\n"
            "Download from https://www.kaggle.com/datasets/ellipticco/elliptic-data-set "
            "and place the three CSV files in data/elliptic/dataset/"
        )
    df_features = pd.read_csv(data_dir / "elliptic_txs_features.csv", header=None)
    df_classes = pd.read_csv(data_dir / "elliptic_txs_classes.csv")
    df_edges = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")
    return df_features, df_classes, df_edges


def _rename_columns(df_features: pd.DataFrame, df_classes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign human-readable column names.

    Feature layout (167 columns):
      col 0: txId
      col 1: time_step
      cols 2–94: 93 local transaction features (trans_feat_0 … trans_feat_92)
      cols 95–166: 72 aggregated neighbourhood features (agg_feat_0 … agg_feat_71)
    """
    df_features.columns = (
        ["id", "time_step"]
        + [f"trans_feat_{i}" for i in range(93)]
        + [f"agg_feat_{i}" for i in range(72)]
    )
    df_classes["class"] = df_classes["class"].map(_CLASS_MAP)
    return df_features, df_classes


def load_elliptic(
    data_dir: Path | str | None = None,
    only_labeled: bool = True,
    include_unknown_for_unsupervised: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None]:
    """Load and preprocess the Elliptic dataset.

    Parameters
    ----------
    data_dir:
        Directory containing the three CSV files.
    only_labeled:
        If True (default), drop rows with unknown label — used for supervised training.
    include_unknown_for_unsupervised:
        If True, also return the full feature matrix (including unknowns) for
        Isolation Forest training. Ignored when only_labeled=False.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (time_step column retained for temporal splitting).
    y : pd.Series
        Binary labels (1=illicit, 0=licit).
    X_full : pd.DataFrame or None
        Full feature matrix including unknown-class rows, returned when
        include_unknown_for_unsupervised=True.
    """
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    df_features, df_classes, _ = _load_csvs(data_dir)
    df_features, df_classes = _rename_columns(df_features, df_classes)

    df = df_features.merge(df_classes, left_on="id", right_on="txId", how="left")
    df.drop(columns=["txId"], inplace=True)

    X_full = None
    if include_unknown_for_unsupervised:
        X_full = df.drop(columns=["id", "class"]).copy()

    if only_labeled:
        df = df.dropna(subset=["class"]).reset_index(drop=True)

    X = df.drop(columns=["id", "class"])
    y = df["class"].astype(int)
    return X, y, X_full


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    last_train_timestep: int = LAST_TRAIN_TIMESTEP,
    last_timestep: int = LAST_TIMESTEP,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Temporal train/test split — no future leakage.

    Splits on the time_step column. Test set covers time steps
    (last_train_timestep+1) … last_timestep, matching Feedzai section 4.1.
    """
    mask_train = (X["time_step"] >= 1) & (X["time_step"] <= last_train_timestep)
    mask_test = (X["time_step"] > last_train_timestep) & (X["time_step"] <= last_timestep)

    X_train = X[mask_train].reset_index(drop=True)
    X_test = X[mask_test].reset_index(drop=True)
    y_train = y[mask_train].reset_index(drop=True)
    y_test = y[mask_test].reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def feature_columns(X: pd.DataFrame) -> list[str]:
    """Return the 165 numeric feature columns (drops time_step)."""
    return [c for c in X.columns if c != "time_step"]


def class_balance(y: pd.Series) -> dict[str, float]:
    """Report illicit/licit counts and illicit prevalence."""
    counts = y.value_counts()
    return {
        "n_illicit": int(counts.get(1, 0)),
        "n_licit": int(counts.get(0, 0)),
        "illicit_prevalence": float(counts.get(1, 0) / len(y)),
    }
