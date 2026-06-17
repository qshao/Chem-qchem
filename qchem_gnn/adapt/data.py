from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..graph import GraphData, build_graph_from_smiles


@dataclass
class AdaptData:
    smiles: list[str]
    graphs: list[GraphData]
    targets: np.ndarray          # [N, T] float32
    target_names: list[str]
    valid_idx: list[int]
    task: str


def detect_smiles_column(df: pd.DataFrame, smiles_col: str | None) -> str:
    if smiles_col and smiles_col != "auto":
        if smiles_col not in df.columns:
            raise ValueError(f"smiles_col '{smiles_col}' not in columns {list(df.columns)}")
        return smiles_col
    for c in df.columns:
        if c.lower() == "smiles":
            return c
    raise ValueError(f"Could not auto-detect a SMILES column in {list(df.columns)}")


def detect_target_columns(df: pd.DataFrame, targets, smiles_col: str) -> list[str]:
    if isinstance(targets, list) and len(targets) == 0:
        raise ValueError("targets list must not be empty")
    if targets and targets != "auto":
        cols = [targets] if isinstance(targets, str) else list(targets)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"target columns not found: {missing}")
        return cols
    candidates = [
        c for c in df.columns
        if "solubility" in c.lower() and "sd" not in c.lower() and "esol" not in c.lower()
    ]
    if candidates:
        return [candidates[0]]
    numeric = [c for c in df.columns if c != smiles_col and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        raise ValueError("Could not auto-detect a target column")
    return [numeric[-1]]


def load_dataset(csv, smiles_col, targets, task: str) -> AdaptData:
    df = pd.read_csv(csv)
    smiles_col = detect_smiles_column(df, smiles_col)
    target_cols = detect_target_columns(df, targets, smiles_col)
    df = df[[smiles_col, *target_cols]].dropna().reset_index(drop=True)

    raw_smiles = df[smiles_col].tolist()
    raw_y = df[target_cols].to_numpy(dtype=np.float32)

    smiles: list[str] = []
    graphs: list[GraphData] = []
    valid_idx: list[int] = []
    keep_rows: list[int] = []
    for i, smi in enumerate(raw_smiles):
        try:
            graphs.append(build_graph_from_smiles(smi))
        except Exception:
            continue
        smiles.append(smi)
        valid_idx.append(i)
        keep_rows.append(i)

    targets_arr = raw_y[keep_rows] if keep_rows else np.empty((0, len(target_cols)), dtype=np.float32)
    if task == "classification" and targets_arr.size:
        uniq = np.unique(targets_arr)
        if not np.all(np.isin(uniq, [0.0, 1.0])):
            raise ValueError("classification targets must be binary 0/1 per column")
    return AdaptData(
        smiles=smiles, graphs=graphs, targets=targets_arr,
        target_names=target_cols, valid_idx=valid_idx, task=task,
    )


def stratified_split(targets, task, test_frac, val_frac, seed, stratify, n_bins=5):
    rng = np.random.default_rng(seed)
    n = targets.shape[0]

    if stratify:
        if task == "classification":
            strat = targets[:, 0].astype(int)
        else:
            col = targets[:, 0]
            bounds = np.quantile(col, np.linspace(0, 1, n_bins + 1)[1:-1])
            strat = np.digitize(col, bounds)
        groups = [np.where(strat == g)[0] for g in np.unique(strat)]
    else:
        groups = [np.arange(n)]

    train_val_idx: list[int] = []
    test_idx: list[int] = []
    for mask in groups:
        mask = mask.copy()
        rng.shuffle(mask)
        cut = max(1, int(len(mask) * test_frac)) if len(mask) > 1 else 0
        test_idx.extend(mask[:cut].tolist())
        train_val_idx.extend(mask[cut:].tolist())

    tv = np.array(train_val_idx)
    rng.shuffle(tv)
    val_cut = max(1, int(len(tv) * val_frac)) if len(tv) > 1 else 0
    val_idx = tv[:val_cut].tolist()
    train_idx = tv[val_cut:].tolist()
    return train_idx, val_idx, test_idx


class LabelNormalizer:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = np.asarray(mu, dtype=np.float32)
        self.sigma = np.asarray(sigma, dtype=np.float32)

    @classmethod
    def fit(cls, y_train: np.ndarray) -> "LabelNormalizer":
        mu = y_train.mean(axis=0)
        sigma = y_train.std(axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        return cls(mu, sigma)

    def transform(self, y):
        return (y - self.mu) / self.sigma

    def inverse(self, y):
        return y * self.sigma + self.mu

    def to_dict(self):
        return {"mu": self.mu.tolist(), "sigma": self.sigma.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["mu"], dtype=np.float32), np.array(d["sigma"], dtype=np.float32))
