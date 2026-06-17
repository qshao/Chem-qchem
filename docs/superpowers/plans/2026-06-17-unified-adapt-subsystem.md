# Unified Adaptation Subsystem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three standalone `train_*.py` adapter scripts with one config-driven `adapt` command that is easy to point at a new dataset, robust (single source of truth + tests), and supports built-in hyperparameter sweeps and multi-target regression/classification.

**Architecture:** A new `qchem_gnn/adapt/` package holds all downstream-adaptation logic behind a small method protocol (`mlp_head`, `finetune`, `engine`), with one place for data loading/column-detection/splitting and one place for backbone embedding. A self-contained `adapt/config.py` (separate from the legacy `config.py`) parses and validates the `adapt` YAML schema. A `runner` executes one run; a `sweep` module expands a grid of dotted-key overrides and writes a comparison report. The legacy `config.py`/`downstream` command and existing imports stay working via thin re-export shims.

**Tech Stack:** Python 3.13, PyTorch, NumPy, pandas, RDKit (via existing `graph.py`), PyYAML, pytest.

## Global Constraints

- Backbone GNN reconstruction uses `model_config` keys exactly: `atom_vocab_size`, `bond_vocab_size`, `hidden_dim`, `num_message_passing_steps`, `graph_targets`.
- Backbone is loaded with `qchem_gnn.checkpoint.load_checkpoint(path)` → dict with `model_state_dict` and `model_config`.
- Graph construction: `qchem_gnn.graph.build_graph_from_smiles(smiles)` raises `ValueError` on unparseable SMILES; batching via `GraphBatch.from_graphs(list[GraphData])`.
- Embeddings: `model.encode_graph_embeddings(batch)` → `[B, hidden_dim]` final embedding. Per-layer pooled states are obtained by iterating `model.encoder.message_passing_blocks` and calling `model.encoder._mean_pool(node_states, batch.batch, batch.num_graphs)` (see `engine_adapter.extract_intermediate_embeddings` for the exact loop being ported).
- Saved adapters MUST carry an `adapter_type` key (`mlp_head` | `finetune` | `engine`) so inference dispatches purely on it.
- CLI entry point is `python -m qchem_gnn.cli` (no `qchem` console script exists). Tests call `qchem_gnn.cli.main(argv_list)`.
- New code targets single-target regression in Phase 1; multi-target + classification arrive in Phase 3. Heads are built with an explicit `output_dim` from the start so Phase 3 needs no head rewrite.
- All randomness seeded; splits reproducible from `split.seed`.
- Unparseable SMILES are skipped with a counted report, never crash a run.

---

## Phase 1 — Package + single-run regression

### Task 1: `adapt/data.py` — data loading, column detection, split, label normalization

**Files:**
- Create: `qchem_gnn/adapt/__init__.py` (empty for now)
- Create: `qchem_gnn/adapt/data.py`
- Test: `tests/test_adapt_data.py`

**Interfaces:**
- Consumes: `qchem_gnn.graph.build_graph_from_smiles`, `GraphData`.
- Produces:
  - `@dataclass AdaptData{ smiles: list[str]; graphs: list[GraphData]; targets: np.ndarray (shape [N, T], float32); target_names: list[str]; valid_idx: list[int]; task: str }`
  - `detect_smiles_column(df: pd.DataFrame, smiles_col: str | None) -> str`
  - `detect_target_columns(df: pd.DataFrame, targets: list[str] | str | None, smiles_col: str) -> list[str]`
  - `load_dataset(csv: str | Path, smiles_col: str | None, targets, task: str) -> AdaptData`
  - `stratified_split(targets: np.ndarray, task: str, test_frac: float, val_frac: float, seed: int, stratify: bool, n_bins: int = 5) -> tuple[list[int], list[int], list[int]]` (returns train, val, test index lists into `0..N-1`)
  - `class LabelNormalizer` with `mu: np.ndarray [T]`, `sigma: np.ndarray [T]`; classmethod `fit(y_train: np.ndarray) -> LabelNormalizer`; `transform(y)`, `inverse(y)`, `to_dict()`, classmethod `from_dict(d)`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_adapt_data.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qchem_gnn.adapt.data import (
    AdaptData,
    LabelNormalizer,
    detect_smiles_column,
    detect_target_columns,
    load_dataset,
    stratified_split,
)


def _write_csv(tmp_path, rows):
    path = tmp_path / "mini.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_detect_smiles_column_auto():
    df = pd.DataFrame({"Compound ID": ["a"], "smiles": ["CCO"], "y": [1.0]})
    assert detect_smiles_column(df, None) == "smiles"
    assert detect_smiles_column(df, "smiles") == "smiles"


def test_detect_target_columns_auto_solubility_excludes_esol_and_sd():
    df = pd.DataFrame(
        {"smiles": ["CCO"], "measured log solubility in mols per litre": [1.0],
         "ESOL predicted log solubility in mols per litre": [0.9]}
    )
    cols = detect_target_columns(df, "auto", "smiles")
    assert cols == ["measured log solubility in mols per litre"]


def test_detect_target_columns_explicit_list():
    df = pd.DataFrame({"smiles": ["CCO"], "a": [1.0], "b": [2.0]})
    assert detect_target_columns(df, ["a", "b"], "smiles") == ["a", "b"]


def test_load_dataset_drops_unparseable_smiles(tmp_path):
    csv = _write_csv(tmp_path, {"smiles": ["CCO", "not_a_smiles", "c1ccccc1"], "y": [1.0, 2.0, 3.0]})
    data = load_dataset(csv, None, ["y"], "regression")
    assert data.smiles == ["CCO", "c1ccccc1"]
    assert data.valid_idx == [0, 2]
    assert data.targets.shape == (2, 1)
    assert len(data.graphs) == 2
    assert data.target_names == ["y"]


def test_stratified_split_is_disjoint_and_covers_all(tmp_path):
    y = np.linspace(-10, 2, 200).reshape(-1, 1).astype(np.float32)
    tr, va, te = stratified_split(y, "regression", 0.2, 0.25, seed=42, stratify=True)
    allidx = sorted(tr + va + te)
    assert allidx == list(range(200))
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    assert abs(len(te) / 200 - 0.2) < 0.05
    assert abs(len(va) / 200 - 0.2) < 0.05


def test_stratified_split_reproducible():
    y = np.random.RandomState(0).randn(100, 1).astype(np.float32)
    a = stratified_split(y, "regression", 0.2, 0.25, seed=7, stratify=True)
    b = stratified_split(y, "regression", 0.2, 0.25, seed=7, stratify=True)
    assert a == b


def test_label_normalizer_round_trip():
    y = np.array([[1.0], [3.0], [5.0]], dtype=np.float32)
    norm = LabelNormalizer.fit(y)
    z = norm.transform(y)
    assert np.allclose(norm.inverse(z), y, atol=1e-5)
    restored = LabelNormalizer.from_dict(norm.to_dict())
    assert np.allclose(restored.inverse(z), y, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_adapt_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qchem_gnn.adapt'`

- [ ] **Step 3: Create the package and implement `data.py`**

```python
# qchem_gnn/adapt/__init__.py
```
(leave empty)

```python
# qchem_gnn/adapt/data.py
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
    if targets and targets != "auto":
        cols = list(targets)
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
    df = df[[smiles_col, *target_cols]].dropna()

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_adapt_data.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/adapt/__init__.py qchem_gnn/adapt/data.py tests/test_adapt_data.py
git commit -m "feat(adapt): data loading, column detection, stratified split, label norm"
```

---

### Task 2: `adapt/backbone.py` — backbone loading & embedding helpers

**Files:**
- Create: `qchem_gnn/adapt/backbone.py`
- Test: `tests/test_adapt_backbone.py`

**Interfaces:**
- Consumes: `load_checkpoint`, `MolecularQuantumGNN`, `GraphBatch`, `GraphData`, `build_graph_from_smiles`.
- Produces:
  - `load_backbone(path: str | Path) -> tuple[MolecularQuantumGNN, dict]` (model in eval mode, plus `model_config`)
  - `embed_final(graphs: list[GraphData], model, batch_size: int = 256) -> np.ndarray` → `[N, H]`
  - `embed_per_layer(graphs: list[GraphData], model, batch_size: int = 256) -> list[np.ndarray]` → list length `num_message_passing_steps`, each `[N, H]`
  - `build_graphs(smiles: list[str]) -> tuple[list[GraphData], list[int]]` (valid graphs + indices into input)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_adapt_backbone.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from qchem_gnn.adapt.backbone import build_graphs, embed_final, embed_per_layer, load_backbone
from qchem_gnn.checkpoint import save_checkpoint
from qchem_gnn.model import MolecularQuantumGNN

MODEL_CONFIG = dict(
    atom_vocab_size=128, bond_vocab_size=8, hidden_dim=16,
    num_message_passing_steps=3, graph_targets=2,
)


def _make_ckpt(tmp_path: Path) -> Path:
    model = MolecularQuantumGNN(**MODEL_CONFIG)
    path = tmp_path / "backbone.pt"
    save_checkpoint(path, {"model_state_dict": model.state_dict(), "model_config": MODEL_CONFIG})
    return path


def test_load_backbone_returns_model_and_config(tmp_path):
    path = _make_ckpt(tmp_path)
    model, cfg = load_backbone(path)
    assert cfg["hidden_dim"] == 16
    assert not model.training


def test_build_graphs_skips_invalid():
    graphs, idx = build_graphs(["CCO", "xxx", "c1ccccc1"])
    assert idx == [0, 2]
    assert len(graphs) == 2


def test_embed_final_shape(tmp_path):
    model, _ = load_backbone(_make_ckpt(tmp_path))
    graphs, _ = build_graphs(["CCO", "c1ccccc1", "CC(=O)O"])
    emb = embed_final(graphs, model, batch_size=2)
    assert emb.shape == (3, 16)
    assert np.isfinite(emb).all()


def test_embed_per_layer_shape(tmp_path):
    model, _ = load_backbone(_make_ckpt(tmp_path))
    graphs, _ = build_graphs(["CCO", "c1ccccc1", "CC(=O)O"])
    layers = embed_per_layer(graphs, model, batch_size=2)
    assert len(layers) == 3
    for layer in layers:
        assert layer.shape == (3, 16)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_adapt_backbone.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qchem_gnn.adapt.backbone'`

- [ ] **Step 3: Implement `backbone.py`**

The `embed_per_layer` loop is ported from `qchem_gnn/engine_adapter.py::extract_intermediate_embeddings` (the inner per-layer pooling loop), but takes pre-built graphs instead of SMILES.

```python
# qchem_gnn/adapt/backbone.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..checkpoint import load_checkpoint
from ..graph import GraphBatch, GraphData, build_graph_from_smiles
from ..model import MolecularQuantumGNN


def load_backbone(path) -> tuple[MolecularQuantumGNN, dict]:
    ckpt = load_checkpoint(Path(path))
    model = MolecularQuantumGNN(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["model_config"]


def build_graphs(smiles: list[str]) -> tuple[list[GraphData], list[int]]:
    graphs: list[GraphData] = []
    valid_idx: list[int] = []
    for i, smi in enumerate(smiles):
        try:
            graphs.append(build_graph_from_smiles(smi))
            valid_idx.append(i)
        except Exception:
            pass
    return graphs, valid_idx


def embed_final(graphs: list[GraphData], model: MolecularQuantumGNN, batch_size: int = 256) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for s in range(0, len(graphs), batch_size):
        batch = GraphBatch.from_graphs(graphs[s : s + batch_size])
        with torch.no_grad():
            chunks.append(model.encode_graph_embeddings(batch).cpu().numpy())
    if not chunks:
        return np.empty((0, model.encoder.atom_encoder.embedding_dim), dtype=np.float32)
    return np.concatenate(chunks)


def embed_per_layer(graphs: list[GraphData], model: MolecularQuantumGNN, batch_size: int = 256) -> list[np.ndarray]:
    encoder = model.encoder
    num_layers = len(encoder.message_passing_blocks)
    per_layer: list[list[np.ndarray]] = [[] for _ in range(num_layers)]

    model.eval()
    for s in range(0, len(graphs), batch_size):
        batch = GraphBatch.from_graphs(graphs[s : s + batch_size])
        with torch.no_grad():
            node_states = encoder.atom_encoder(batch.atomic_numbers)
            edge_states = encoder.bond_encoder(batch.edge_attr.squeeze(-1))
            for i, block in enumerate(encoder.message_passing_blocks):
                node_states = block(node_states, batch.edge_index, edge_states)
                pooled = encoder._mean_pool(node_states, batch.batch, batch.num_graphs)
                per_layer[i].append(pooled.cpu().numpy())
    return [np.concatenate(c) for c in per_layer]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_adapt_backbone.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/adapt/backbone.py tests/test_adapt_backbone.py
git commit -m "feat(adapt): backbone loading and final/per-layer embedding helpers"
```

---

### Task 3: `adapt/metrics.py` — regression & classification metrics

**Files:**
- Create: `qchem_gnn/adapt/metrics.py`
- Test: `tests/test_adapt_metrics.py`

**Interfaces:**
- Produces:
  - `regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict` — accepts `[N]` or `[N, T]`; returns `{"per_target": [ {mae,rmse,r2}, ... ], "mae": macro, "rmse": macro, "r2": macro}`.
  - `classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict` — returns `{"per_target": [ {auc, accuracy, f1}, ... ], "auc": macro, "accuracy": macro, "f1": macro}`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_adapt_metrics.py
from __future__ import annotations

import numpy as np

from qchem_gnn.adapt.metrics import classification_metrics, regression_metrics


def test_regression_metrics_perfect():
    y = np.array([[1.0], [2.0], [3.0]])
    m = regression_metrics(y, y.copy())
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0
    assert len(m["per_target"]) == 1


def test_regression_metrics_multitarget():
    y = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    pred = y + 1.0
    m = regression_metrics(y, pred)
    assert len(m["per_target"]) == 2
    assert abs(m["mae"] - 1.0) < 1e-6


def test_classification_metrics_perfect():
    y = np.array([[0], [1], [1], [0]])
    prob = np.array([[0.1], [0.9], [0.8], [0.2]])
    m = classification_metrics(y, prob)
    assert abs(m["accuracy"] - 1.0) < 1e-6
    assert abs(m["auc"] - 1.0) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_adapt_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `metrics.py`**

```python
# qchem_gnn/adapt/metrics.py
from __future__ import annotations

import numpy as np


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt, yp = _as_2d(y_true), _as_2d(y_pred)
    per_target = []
    for t in range(yt.shape[1]):
        diff = yp[:, t] - yt[:, t]
        ss_res = float(np.sum(diff ** 2))
        ss_tot = float(np.sum((yt[:, t] - yt[:, t].mean()) ** 2))
        per_target.append({
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        })
    return {
        "per_target": per_target,
        "mae": float(np.mean([m["mae"] for m in per_target])),
        "rmse": float(np.mean([m["rmse"] for m in per_target])),
        "r2": float(np.mean([m["r2"] for m in per_target])),
    }


def _auc_binary(y: np.ndarray, p: np.ndarray) -> float:
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Mann-Whitney U statistic / (n_pos * n_neg)
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    auc = (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    yt, yp = _as_2d(y_true), _as_2d(y_prob)
    per_target = []
    for t in range(yt.shape[1]):
        y = yt[:, t].astype(int)
        p = yp[:, t]
        pred = (p >= 0.5).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_target.append({
            "auc": _auc_binary(y, p),
            "accuracy": float(np.mean(pred == y)),
            "f1": float(f1),
        })
    return {
        "per_target": per_target,
        "auc": float(np.mean([m["auc"] for m in per_target])),
        "accuracy": float(np.mean([m["accuracy"] for m in per_target])),
        "f1": float(np.mean([m["f1"] for m in per_target])),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_adapt_metrics.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/adapt/metrics.py tests/test_adapt_metrics.py
git commit -m "feat(adapt): regression and classification metrics"
```

---

### Task 4: `adapt/methods/base.py` — protocol, dataclasses, shared `MLPHead`

**Files:**
- Create: `qchem_gnn/adapt/methods/__init__.py` (empty)
- Create: `qchem_gnn/adapt/methods/base.py`
- Test: `tests/test_adapt_methods_base.py`

**Interfaces:**
- Produces:
  - `class MLPHead(nn.Module)` with `__init__(self, input_dim: int, output_dim: int = 1, hidden_dims: tuple[int, ...] = (128, 64), dropout: float = 0.1)`; `forward(x) -> Tensor [B, output_dim]`; `config() -> dict` (keys: `input_dim`, `output_dim`, `hidden_dims`, `dropout`).
  - `@dataclass TrainResult{ payload: dict; test_metrics: dict; log: dict }`
  - `@dataclass LoadedAdapter{ adapter_type: str; payload: dict }`
  - `class AdaptMethod(Protocol)` with attribute `name: str` and methods `train(self, backbone, model_config, data, train_idx, val_idx, test_idx, cfg) -> TrainResult`; `save(self, path, result, meta) -> None`; staticmethod `load(path) -> LoadedAdapter`; staticmethod `predict(loaded, smiles, **kw) -> tuple[np.ndarray, list[int]]`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_adapt_methods_base.py
from __future__ import annotations

import torch

from qchem_gnn.adapt.methods.base import MLPHead


def test_mlphead_output_dim():
    head = MLPHead(input_dim=16, output_dim=3, hidden_dims=(8,), dropout=0.0)
    out = head(torch.zeros(5, 16))
    assert out.shape == (5, 3)


def test_mlphead_config_round_trip():
    head = MLPHead(input_dim=16, output_dim=2, hidden_dims=(8, 4), dropout=0.1)
    cfg = head.config()
    clone = MLPHead(**cfg)
    clone.load_state_dict(head.state_dict())
    assert cfg["output_dim"] == 2 and cfg["hidden_dims"] == [8, 4]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapt_methods_base.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `base.py`**

`MLPHead` is adapted from `qchem_gnn/adapters.py::MLPHead`, changed to emit `output_dim` units (no `squeeze`).

```python
# qchem_gnn/adapt/methods/base.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1,
                 hidden_dims: tuple[int, ...] = (128, 64), dropout: float = 0.1):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def config(self) -> dict[str, Any]:
        return {"input_dim": self.input_dim, "output_dim": self.output_dim,
                "hidden_dims": list(self.hidden_dims), "dropout": self.dropout}


@dataclass
class TrainResult:
    payload: dict
    test_metrics: dict
    log: dict


@dataclass
class LoadedAdapter:
    adapter_type: str
    payload: dict


class AdaptMethod(Protocol):
    name: str

    def train(self, backbone, model_config: dict, data, train_idx, val_idx, test_idx, cfg) -> TrainResult: ...
    def save(self, path: Path, result: TrainResult, meta: dict) -> None: ...
    @staticmethod
    def load(path: Path) -> LoadedAdapter: ...
    @staticmethod
    def predict(loaded: LoadedAdapter, smiles: list[str], **kw) -> tuple[np.ndarray, list[int]]: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapt_methods_base.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/adapt/methods/__init__.py qchem_gnn/adapt/methods/base.py tests/test_adapt_methods_base.py
git commit -m "feat(adapt): method protocol, result dataclasses, multi-output MLPHead"
```

---

### Task 5: `adapt/methods/mlp_head.py` — frozen-backbone MLP method

**Files:**
- Create: `qchem_gnn/adapt/methods/mlp_head.py`
- Test: `tests/test_adapt_mlp_head.py`

**Interfaces:**
- Consumes: `MLPHead`, `TrainResult`, `LoadedAdapter` (Task 4); `embed_final`, `build_graphs`, `load_backbone` (Task 2); `LabelNormalizer` (Task 1); `regression_metrics` (Task 3).
- Produces: `class MlpHeadMethod` implementing `AdaptMethod` with `name = "mlp_head"`. Saved payload keys: `adapter_type="mlp_head"`, `head_state`, `head_config`, `feat_mu`, `feat_sig`, `label_norm` (dict), `backbone_ckpt`, `target_names`, `task`, `training_info`.
  - `train(self, backbone, model_config, data, train_idx, val_idx, test_idx, cfg) -> TrainResult` — `cfg` is the resolved `AdaptConfig` (Task 8); reads `cfg.adapter` (`hidden_dims`, `dropout`) and `cfg.training` (`epochs`, `head_lr`, `batch_size`, `patience`, `seed`).
  - `predict(loaded, smiles, **kw) -> (preds [Nvalid, T], valid_idx)`.

- [ ] **Step 1: Write failing test** (uses a tiny in-memory backbone + synthetic linear target so a frozen MLP can actually fit)

```python
# tests/test_adapt_mlp_head.py
from __future__ import annotations

import numpy as np
import torch

from qchem_gnn.adapt.backbone import build_graphs, load_backbone
from qchem_gnn.adapt.data import AdaptData
from qchem_gnn.adapt.methods.mlp_head import MlpHeadMethod
from qchem_gnn.checkpoint import save_checkpoint
from qchem_gnn.model import MolecularQuantumGNN

MODEL_CONFIG = dict(atom_vocab_size=128, bond_vocab_size=8, hidden_dim=16,
                    num_message_passing_steps=2, graph_targets=2)


def _ckpt(tmp_path):
    model = MolecularQuantumGNN(**MODEL_CONFIG)
    p = tmp_path / "bb.pt"
    save_checkpoint(p, {"model_state_dict": model.state_dict(), "model_config": MODEL_CONFIG})
    return p


class _Cfg:
    def __init__(self):
        self.adapter = {"hidden_dims": [8], "dropout": 0.0}
        self.training = {"epochs": 30, "head_lr": 1e-2, "batch_size": 8, "patience": 50, "seed": 0}


def _data():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "CCN", "CCCCO", "c1ccncc1", "CCCl"]
    graphs, idx = build_graphs(smiles)
    y = np.linspace(-3, 3, len(graphs)).reshape(-1, 1).astype(np.float32)
    return AdaptData(smiles=[smiles[i] for i in idx], graphs=graphs, targets=y,
                     target_names=["y"], valid_idx=idx, task="regression")


def test_mlp_head_train_save_load_predict(tmp_path):
    ckpt = _ckpt(tmp_path)
    backbone, cfg_model = load_backbone(ckpt)
    data = _data()
    n = len(data.graphs)
    method = MlpHeadMethod()
    result = method.train(backbone, cfg_model, data,
                          train_idx=list(range(n - 4)), val_idx=[n - 4, n - 3],
                          test_idx=[n - 2, n - 1], cfg=_Cfg())
    assert "mae" in result.test_metrics
    assert result.payload["adapter_type"] == "mlp_head"

    out = tmp_path / "mlp.pt"
    method.save(out, result, meta={"backbone_ckpt": str(ckpt), "target_names": ["y"], "task": "regression"})
    loaded = MlpHeadMethod.load(out)
    preds, valid = MlpHeadMethod.predict(loaded, ["CCO", "bad", "c1ccccc1"])
    assert valid == [0, 2]
    assert preds.shape == (2, 1)
    assert np.isfinite(preds).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapt_mlp_head.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `mlp_head.py`** (training loop adapted from `adapters.py::train_mlp_head`, generalized to `output_dim = T` and to consume pre-computed split indices + `AdaptData`)

```python
# qchem_gnn/adapt/methods/mlp_head.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..backbone import build_graphs, embed_final, load_backbone
from ..data import LabelNormalizer
from ..metrics import regression_metrics
from .base import LoadedAdapter, MLPHead, TrainResult


class MlpHeadMethod:
    name = "mlp_head"

    def train(self, backbone, model_config, data, train_idx, val_idx, test_idx, cfg) -> TrainResult:
        seed = int(cfg.training.get("seed", 42))
        torch.manual_seed(seed)

        emb = embed_final(data.graphs, backbone, batch_size=int(cfg.training.get("batch_size", 128)) * 2)
        y = data.targets
        T = y.shape[1]

        feat_mu = emb[train_idx].mean(0)
        feat_sig = emb[train_idx].std(0).clip(1e-8)
        norm = LabelNormalizer.fit(y[train_idx])

        def nf(X): return (X - feat_mu) / feat_sig

        Xtr = torch.as_tensor(nf(emb[train_idx]), dtype=torch.float32)
        Xva = torch.as_tensor(nf(emb[val_idx]), dtype=torch.float32) if val_idx else None
        Xte = torch.as_tensor(nf(emb[test_idx]), dtype=torch.float32) if test_idx else None
        ytr = torch.as_tensor(norm.transform(y[train_idx]), dtype=torch.float32)

        head = MLPHead(emb.shape[1], output_dim=T,
                       hidden_dims=tuple(cfg.adapter.get("hidden_dims", (128, 64))),
                       dropout=float(cfg.adapter.get("dropout", 0.1)))
        epochs = int(cfg.training.get("epochs", 300))
        lr = float(cfg.training.get("head_lr", 1e-3))
        bs = int(cfg.training.get("batch_size", 128))
        patience = int(cfg.training.get("patience", 40))
        opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
        crit = nn.MSELoss()

        best_val, best_state, wait = float("inf"), {k: v.clone() for k, v in head.state_dict().items()}, 0
        n = len(Xtr)
        for _ in range(epochs):
            head.train()
            perm = torch.randperm(n)
            for s in range(0, n, bs):
                idx = perm[s : s + bs]
                opt.zero_grad()
                loss = crit(head(Xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()
            sched.step()
            if Xva is not None:
                head.eval()
                with torch.no_grad():
                    vp = norm.inverse(head(Xva).numpy())
                vmae = float(np.mean(np.abs(vp - y[val_idx])))
                if vmae < best_val:
                    best_val, wait = vmae, 0
                    best_state = {k: v.clone() for k, v in head.state_dict().items()}
                else:
                    wait += 1
                    if wait >= patience:
                        break

        head.load_state_dict(best_state)
        head.eval()
        test_metrics = {}
        if Xte is not None:
            with torch.no_grad():
                tp = norm.inverse(head(Xte).numpy())
            test_metrics = regression_metrics(y[test_idx], tp)

        payload = {
            "adapter_type": "mlp_head",
            "head_state": head.state_dict(),
            "head_config": head.config(),
            "feat_mu": feat_mu.tolist(),
            "feat_sig": feat_sig.tolist(),
            "label_norm": norm.to_dict(),
        }
        return TrainResult(payload=payload, test_metrics=test_metrics,
                           log={"best_val_mae": round(best_val, 4)})

    def save(self, path: Path, result: TrainResult, meta: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**result.payload, **meta, "training_info": {**meta.get("training_info", {}),
                                                                "test_metrics": result.test_metrics,
                                                                **result.log}}, path)

    @staticmethod
    def load(path: Path) -> LoadedAdapter:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        return LoadedAdapter(adapter_type="mlp_head", payload=state)

    @staticmethod
    def predict(loaded: LoadedAdapter, smiles: list[str], **kw) -> tuple[np.ndarray, list[int]]:
        s = loaded.payload
        model, _ = load_backbone(s["backbone_ckpt"])
        graphs, valid_idx = build_graphs(smiles)
        emb = embed_final(graphs, model, batch_size=kw.get("batch_size", 256))
        cfg = s["head_config"]
        head = MLPHead(**cfg)
        head.load_state_dict(s["head_state"])
        head.eval()
        feat_mu = np.array(s["feat_mu"]); feat_sig = np.array(s["feat_sig"])
        norm = LabelNormalizer.from_dict(s["label_norm"])
        with torch.no_grad():
            X = torch.as_tensor((emb - feat_mu) / feat_sig, dtype=torch.float32)
            preds = norm.inverse(head(X).numpy())
        return preds, valid_idx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapt_mlp_head.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/adapt/methods/mlp_head.py tests/test_adapt_mlp_head.py
git commit -m "feat(adapt): mlp_head method (frozen backbone) with save/load/predict"
```

---

### Task 6: `adapt/methods/finetune.py` — joint backbone+head method

**Files:**
- Create: `qchem_gnn/adapt/methods/finetune.py`
- Test: `tests/test_adapt_finetune.py`

**Interfaces:**
- Consumes: same helpers as Task 5 plus `GraphBatch`, `MolecularQuantumGNN`, `load_backbone`.
- Produces: `class FinetuneMethod` (`name = "finetune"`). Saved payload keys: `adapter_type="finetune"`, `model_state`, `model_config`, `head_state`, `head_config`, `label_norm`, `backbone_ckpt`, plus meta. Reads `cfg.training`: `epochs`, `head_lr`, `backbone_lr`, `batch_size`, `patience`, `grad_clip`, `seed`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_adapt_finetune.py
from __future__ import annotations

import numpy as np
import torch

from qchem_gnn.adapt.backbone import build_graphs, load_backbone
from qchem_gnn.adapt.data import AdaptData
from qchem_gnn.adapt.methods.finetune import FinetuneMethod
from qchem_gnn.checkpoint import save_checkpoint
from qchem_gnn.model import MolecularQuantumGNN

MODEL_CONFIG = dict(atom_vocab_size=128, bond_vocab_size=8, hidden_dim=16,
                    num_message_passing_steps=2, graph_targets=2)


def _ckpt(tmp_path):
    model = MolecularQuantumGNN(**MODEL_CONFIG)
    p = tmp_path / "bb.pt"
    save_checkpoint(p, {"model_state_dict": model.state_dict(), "model_config": MODEL_CONFIG})
    return p


class _Cfg:
    def __init__(self):
        self.adapter = {"hidden_dims": [8], "dropout": 0.0}
        self.training = {"epochs": 5, "head_lr": 1e-2, "backbone_lr": 1e-3,
                         "batch_size": 4, "patience": 50, "grad_clip": 1.0, "seed": 0}


def _data():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "CCN", "CCCCO", "c1ccncc1", "CCCl"]
    graphs, idx = build_graphs(smiles)
    y = np.linspace(-3, 3, len(graphs)).reshape(-1, 1).astype(np.float32)
    return AdaptData(smiles=[smiles[i] for i in idx], graphs=graphs, targets=y,
                     target_names=["y"], valid_idx=idx, task="regression")


def test_finetune_train_save_load_predict(tmp_path):
    ckpt = _ckpt(tmp_path)
    backbone, cfg_model = load_backbone(ckpt)
    data = _data(); n = len(data.graphs)
    method = FinetuneMethod()
    result = method.train(backbone, cfg_model, data,
                          train_idx=list(range(n - 4)), val_idx=[n - 4, n - 3],
                          test_idx=[n - 2, n - 1], cfg=_Cfg())
    assert "mae" in result.test_metrics
    out = tmp_path / "ft.pt"
    method.save(out, result, meta={"backbone_ckpt": str(ckpt), "target_names": ["y"], "task": "regression"})
    loaded = FinetuneMethod.load(out)
    preds, valid = FinetuneMethod.predict(loaded, ["CCO", "bad", "c1ccccc1"])
    assert valid == [0, 2] and preds.shape == (2, 1) and np.isfinite(preds).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapt_finetune.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `finetune.py`** (training loop adapted from `adapters.py::train_finetune`, generalized to `output_dim = T` and consuming split indices)

```python
# qchem_gnn/adapt/methods/finetune.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ...graph import GraphBatch
from ...model import MolecularQuantumGNN
from ..backbone import build_graphs, load_backbone
from ..data import LabelNormalizer
from ..metrics import regression_metrics
from .base import LoadedAdapter, MLPHead, TrainResult


class FinetuneMethod:
    name = "finetune"

    def train(self, backbone, model_config, data, train_idx, val_idx, test_idx, cfg) -> TrainResult:
        seed = int(cfg.training.get("seed", 42))
        torch.manual_seed(seed)
        model = backbone
        h_dim = model.encoder.atom_encoder.embedding_dim
        y = data.targets
        T = y.shape[1]
        norm = LabelNormalizer.fit(y[train_idx])

        head = MLPHead(h_dim, output_dim=T,
                       hidden_dims=tuple(cfg.adapter.get("hidden_dims", (128, 64))),
                       dropout=float(cfg.adapter.get("dropout", 0.1)))

        epochs = int(cfg.training.get("epochs", 200))
        head_lr = float(cfg.training.get("head_lr", 1e-3))
        bb_lr = float(cfg.training.get("backbone_lr", 5e-5))
        bs = int(cfg.training.get("batch_size", 64))
        patience = int(cfg.training.get("patience", 30))
        grad_clip = float(cfg.training.get("grad_clip", 1.0))

        opt = torch.optim.Adam(
            [{"params": model.parameters(), "lr": bb_lr},
             {"params": head.parameters(), "lr": head_lr}], weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        crit = nn.MSELoss()

        graphs_tr = [data.graphs[i] for i in train_idx]
        ytr = torch.as_tensor(norm.transform(y[train_idx]), dtype=torch.float32)
        val_batch = GraphBatch.from_graphs([data.graphs[i] for i in val_idx]) if val_idx else None
        te_batch = GraphBatch.from_graphs([data.graphs[i] for i in test_idx]) if test_idx else None

        best_val = float("inf")
        best_model = {k: v.clone() for k, v in model.state_dict().items()}
        best_head = {k: v.clone() for k, v in head.state_dict().items()}
        wait = 0
        n = len(graphs_tr)
        for _ in range(epochs):
            model.train(); head.train()
            perm = torch.randperm(n)
            for s in range(0, n, bs):
                bidx = perm[s : s + bs]
                batch = GraphBatch.from_graphs([graphs_tr[i] for i in bidx.tolist()])
                opt.zero_grad()
                loss = crit(head(model.encode_graph_embeddings(batch)), ytr[bidx])
                loss.backward()
                nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), grad_clip)
                opt.step()
            sched.step()
            if val_batch is not None:
                model.eval(); head.eval()
                with torch.no_grad():
                    vp = norm.inverse(head(model.encode_graph_embeddings(val_batch)).cpu().numpy())
                vmae = float(np.mean(np.abs(vp - y[val_idx])))
                if vmae < best_val:
                    best_val, wait = vmae, 0
                    best_model = {k: v.clone() for k, v in model.state_dict().items()}
                    best_head = {k: v.clone() for k, v in head.state_dict().items()}
                else:
                    wait += 1
                    if wait >= patience:
                        break

        model.load_state_dict(best_model); head.load_state_dict(best_head)
        model.eval(); head.eval()
        test_metrics = {}
        if te_batch is not None:
            with torch.no_grad():
                tp = norm.inverse(head(model.encode_graph_embeddings(te_batch)).cpu().numpy())
            test_metrics = regression_metrics(y[test_idx], tp)

        payload = {
            "adapter_type": "finetune",
            "model_state": model.state_dict(),
            "model_config": model_config,
            "head_state": head.state_dict(),
            "head_config": head.config(),
            "label_norm": norm.to_dict(),
        }
        return TrainResult(payload=payload, test_metrics=test_metrics,
                           log={"best_val_mae": round(best_val, 4)})

    def save(self, path: Path, result: TrainResult, meta: dict) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**result.payload, **meta,
                    "training_info": {**meta.get("training_info", {}),
                                      "test_metrics": result.test_metrics, **result.log}}, path)

    @staticmethod
    def load(path: Path) -> LoadedAdapter:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        return LoadedAdapter(adapter_type="finetune", payload=state)

    @staticmethod
    def predict(loaded: LoadedAdapter, smiles: list[str], **kw) -> tuple[np.ndarray, list[int]]:
        s = loaded.payload
        model = MolecularQuantumGNN(**s["model_config"])
        model.load_state_dict(s["model_state"]); model.eval()
        head = MLPHead(**s["head_config"]); head.load_state_dict(s["head_state"]); head.eval()
        norm = LabelNormalizer.from_dict(s["label_norm"])
        graphs, valid_idx = build_graphs(smiles)
        bs = kw.get("batch_size", 256)
        chunks = []
        for st in range(0, len(graphs), bs):
            batch = GraphBatch.from_graphs(graphs[st : st + bs])
            with torch.no_grad():
                chunks.append(head(model.encode_graph_embeddings(batch)).cpu().numpy())
        preds = norm.inverse(np.concatenate(chunks)) if chunks else np.empty((0, len(s["label_norm"]["mu"])))
        return preds, valid_idx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapt_finetune.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/adapt/methods/finetune.py tests/test_adapt_finetune.py
git commit -m "feat(adapt): finetune method (joint backbone+head) with save/load/predict"
```

---

### Task 7: `adapt/methods/engine.py` + `registry.py` — ENGINE method & method registry

**Files:**
- Create: `qchem_gnn/adapt/methods/engine.py`
- Create: `qchem_gnn/adapt/registry.py`
- Test: `tests/test_adapt_engine.py`, `tests/test_adapt_registry.py`

**Interfaces:**
- Produces:
  - `class EngineAdapterHead(nn.Module)` — ported from `qchem_gnn/engine_adapter.py::EngineAdapterHead`, generalized: exit heads are `nn.Linear(hidden_dim, output_dim)`; `forward` returns list of `[B, output_dim]`; `predict_ensemble(layer_tensors) -> np.ndarray [B, output_dim]`; `predict_early_exit(layer_tensors, tolerance, min_layers=1) -> (preds [B, output_dim], exit_layers [B])` (early-exit std computed on the mean across targets).
  - `class EngineMethod` (`name = "engine"`). Saved payload keys: `adapter_type="engine"`, `adapter_state`, `hidden_dim`, `num_layers`, `output_dim`, `layer_scalers`, `label_norm`, `backbone_ckpt`, plus meta. Reads `cfg.training`: `epochs`, `head_lr` (alias `lr`), `seed`.
  - `registry.get_method(name: str) -> AdaptMethod`; `registry.METHODS: dict[str, type]`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_adapt_registry.py
from qchem_gnn.adapt.registry import METHODS, get_method


def test_registry_has_three_methods():
    assert set(METHODS) == {"mlp_head", "finetune", "engine"}
    assert get_method("engine").name == "engine"


def test_registry_unknown_raises():
    import pytest
    with pytest.raises(KeyError):
        get_method("nope")
```

```python
# tests/test_adapt_engine.py
from __future__ import annotations

import numpy as np

from qchem_gnn.adapt.backbone import build_graphs, load_backbone
from qchem_gnn.adapt.data import AdaptData
from qchem_gnn.adapt.methods.engine import EngineMethod
from qchem_gnn.checkpoint import save_checkpoint
from qchem_gnn.model import MolecularQuantumGNN

MODEL_CONFIG = dict(atom_vocab_size=128, bond_vocab_size=8, hidden_dim=16,
                    num_message_passing_steps=3, graph_targets=2)


def _ckpt(tmp_path):
    model = MolecularQuantumGNN(**MODEL_CONFIG)
    p = tmp_path / "bb.pt"
    save_checkpoint(p, {"model_state_dict": model.state_dict(), "model_config": MODEL_CONFIG})
    return p


class _Cfg:
    def __init__(self):
        self.adapter = {}
        self.training = {"epochs": 20, "head_lr": 5e-3, "seed": 0}


def _data():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "CCN", "CCCCO", "c1ccncc1", "CCCl"]
    graphs, idx = build_graphs(smiles)
    y = np.linspace(-3, 3, len(graphs)).reshape(-1, 1).astype(np.float32)
    return AdaptData(smiles=[smiles[i] for i in idx], graphs=graphs, targets=y,
                     target_names=["y"], valid_idx=idx, task="regression")


def test_engine_train_save_load_predict_modes(tmp_path):
    ckpt = _ckpt(tmp_path)
    backbone, cfg_model = load_backbone(ckpt)
    data = _data(); n = len(data.graphs)
    method = EngineMethod()
    result = method.train(backbone, cfg_model, data,
                          train_idx=list(range(n - 4)), val_idx=[n - 4, n - 3],
                          test_idx=[n - 2, n - 1], cfg=_Cfg())
    assert result.payload["adapter_type"] == "engine"
    out = tmp_path / "eng.pt"
    method.save(out, result, meta={"backbone_ckpt": str(ckpt), "target_names": ["y"], "task": "regression"})
    loaded = EngineMethod.load(out)
    preds, valid = EngineMethod.predict(loaded, ["CCO", "bad", "c1ccccc1"], mode="ensemble")
    assert valid == [0, 2] and preds.shape == (2, 1)
    preds_ee, _ = EngineMethod.predict(loaded, ["CCO", "c1ccccc1"], mode="early_exit", exit_tolerance=0.1)
    assert preds_ee.shape == (2, 1) and np.isfinite(preds_ee).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_adapt_engine.py tests/test_adapt_registry.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `engine.py` and `registry.py`**

`EngineAdapterHead` is ported from `qchem_gnn/engine_adapter.py` (lines for `__init__`, `forward`, `predict_ensemble`, `predict_early_exit`) with these exact changes: `exit_heads` use `nn.Linear(hidden_dim, output_dim)`; `predict_ensemble` returns `torch.stack(preds).mean(0).cpu().numpy()` (shape `[B, output_dim]`, no `squeeze`); in `predict_early_exit`, replace the per-molecule scalar `pred` with the target-mean `pred.mean(dim=1)` for the std/exit test while storing the full `[B, output_dim]` prediction in `final_pred`.

```python
# qchem_gnn/adapt/methods/engine.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..backbone import build_graphs, embed_per_layer, load_backbone
from ..data import LabelNormalizer
from ..metrics import regression_metrics
from .base import LoadedAdapter, TrainResult


class EngineAdapterHead(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, output_dim: int = 1):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                          nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_layers)
        ])
        self.alphas = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)])
        self.exit_heads = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.output_dim = output_dim

    def forward(self, layer_tensors):
        h = torch.zeros_like(layer_tensors[0])
        preds = []
        for proj, alpha, head, x in zip(self.projections, self.alphas, self.exit_heads, layer_tensors):
            gate = torch.sigmoid(alpha)
            h = proj(x) * gate + h * (1.0 - gate)
            preds.append(head(h))
        return preds

    @torch.no_grad()
    def predict_ensemble(self, layer_tensors):
        preds = self.forward(layer_tensors)
        return torch.stack(preds, dim=0).mean(0).cpu().numpy()

    @torch.no_grad()
    def predict_early_exit(self, layer_tensors, tolerance=0.05, min_layers=1):
        B = layer_tensors[0].shape[0]
        h = torch.zeros_like(layer_tensors[0])
        accumulated = []
        active = torch.ones(B, dtype=torch.bool)
        final_pred = torch.zeros(B, self.output_dim)
        exit_layer = torch.full((B,), self.num_layers - 1, dtype=torch.long)
        for i, (proj, alpha, head, x) in enumerate(
            zip(self.projections, self.alphas, self.exit_heads, layer_tensors)
        ):
            gate = torch.sigmoid(alpha)
            h = proj(x) * gate + h * (1.0 - gate)
            pred = head(h)                     # [B, output_dim]
            accumulated.append(pred.mean(dim=1))
            if i >= min_layers and len(accumulated) >= 2:
                std = torch.stack(accumulated, dim=0).std(dim=0)
                exiting = active & (std < tolerance)
                if exiting.any():
                    exit_layer[exiting] = i
                    final_pred[exiting] = pred[exiting]
                    active[exiting] = False
        if active.any():
            final_pred[active] = self.forward(layer_tensors)[-1][active]
        return final_pred.cpu().numpy(), exit_layer.cpu().numpy()


class EngineMethod:
    name = "engine"

    def train(self, backbone, model_config, data, train_idx, val_idx, test_idx, cfg) -> TrainResult:
        seed = int(cfg.training.get("seed", 42))
        torch.manual_seed(seed)
        h_dim = model_config["hidden_dim"]
        n_steps = model_config["num_message_passing_steps"]
        y = data.targets
        T = y.shape[1]

        layer_embs = embed_per_layer(data.graphs, backbone)
        norm = LabelNormalizer.fit(y[train_idx])

        scalers, tr, va, te = [], [], [], []
        for emb in layer_embs:
            mu = emb[train_idx].mean(0); sig = emb[train_idx].std(0).clip(1e-8)
            scalers.append((mu, sig))
            tr.append(torch.as_tensor((emb[train_idx] - mu) / sig, dtype=torch.float32))
            va.append(torch.as_tensor((emb[val_idx] - mu) / sig, dtype=torch.float32) if val_idx else None)
            te.append(torch.as_tensor((emb[test_idx] - mu) / sig, dtype=torch.float32) if test_idx else None)
        ytr = torch.as_tensor(norm.transform(y[train_idx]), dtype=torch.float32)

        adapter = EngineAdapterHead(h_dim, n_steps, output_dim=T)
        epochs = int(cfg.training.get("epochs", 400))
        lr = float(cfg.training.get("head_lr", cfg.training.get("lr", 3e-3)))
        opt = torch.optim.Adam(adapter.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        crit = nn.MSELoss()

        best_val = float("inf")
        best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
        for _ in range(epochs):
            adapter.train()
            opt.zero_grad()
            loss = sum(crit(p, ytr) for p in adapter(tr))
            loss.backward(); opt.step(); sched.step()
            if val_idx:
                adapter.eval()
                vp = norm.inverse(adapter.predict_ensemble(va))
                vmae = float(np.mean(np.abs(vp - y[val_idx])))
                if vmae < best_val:
                    best_val = vmae
                    best_state = {k: v.clone() for k, v in adapter.state_dict().items()}

        adapter.load_state_dict(best_state); adapter.eval()
        test_metrics = {}
        if test_idx:
            tp = norm.inverse(adapter.predict_ensemble(te))
            test_metrics = regression_metrics(y[test_idx], tp)

        payload = {
            "adapter_type": "engine",
            "adapter_state": adapter.state_dict(),
            "hidden_dim": h_dim, "num_layers": n_steps, "output_dim": T,
            "layer_scalers": [(mu.tolist(), sig.tolist()) for mu, sig in scalers],
            "label_norm": norm.to_dict(),
        }
        return TrainResult(payload=payload, test_metrics=test_metrics,
                           log={"best_val_mae": round(best_val, 4)})

    def save(self, path: Path, result: TrainResult, meta: dict) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**result.payload, **meta,
                    "training_info": {**meta.get("training_info", {}),
                                      "test_metrics": result.test_metrics, **result.log}}, path)

    @staticmethod
    def load(path: Path) -> LoadedAdapter:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        return LoadedAdapter(adapter_type="engine", payload=state)

    @staticmethod
    def predict(loaded: LoadedAdapter, smiles: list[str], **kw) -> tuple[np.ndarray, list[int]]:
        s = loaded.payload
        model, _ = load_backbone(s["backbone_ckpt"])
        adapter = EngineAdapterHead(s["hidden_dim"], s["num_layers"], output_dim=s["output_dim"])
        adapter.load_state_dict(s["adapter_state"]); adapter.eval()
        norm = LabelNormalizer.from_dict(s["label_norm"])
        graphs, valid_idx = build_graphs(smiles)
        layer_embs = embed_per_layer(graphs, model, batch_size=kw.get("batch_size", 256))
        scalers = [(np.array(mu), np.array(sig)) for mu, sig in s["layer_scalers"]]
        tensors = [torch.as_tensor((emb - mu) / sig, dtype=torch.float32)
                   for emb, (mu, sig) in zip(layer_embs, scalers)]
        mode = kw.get("mode", "ensemble")
        if mode == "early_exit":
            tol = kw.get("exit_tolerance", 0.05) / float(np.mean(norm.sigma))
            pred_norm, _ = adapter.predict_early_exit(tensors, tolerance=tol)
        else:
            pred_norm = adapter.predict_ensemble(tensors)
        return norm.inverse(pred_norm), valid_idx
```

```python
# qchem_gnn/adapt/registry.py
from __future__ import annotations

from .methods.engine import EngineMethod
from .methods.finetune import FinetuneMethod
from .methods.mlp_head import MlpHeadMethod

METHODS = {
    "mlp_head": MlpHeadMethod,
    "finetune": FinetuneMethod,
    "engine": EngineMethod,
}


def get_method(name: str):
    if name not in METHODS:
        raise KeyError(f"Unknown adapt method '{name}'. Valid: {sorted(METHODS)}")
    return METHODS[name]()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_adapt_engine.py tests/test_adapt_registry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/adapt/methods/engine.py qchem_gnn/adapt/registry.py tests/test_adapt_engine.py tests/test_adapt_registry.py
git commit -m "feat(adapt): engine method (multi-output) and method registry"
```

---

### Task 8: `adapt/config.py` — AdaptConfig parse + validation

**Files:**
- Create: `qchem_gnn/adapt/config.py`
- Test: `tests/test_adapt_config.py`

**Interfaces:**
- Produces:
  - `class AdaptConfigError(ValueError)`
  - `@dataclass AdaptConfig{ method: str; backbone: str; task: str; csv: str; smiles_col: str | None; targets; adapter: dict; training: dict; split: dict; outputs: dict; sweep: dict | None }`
  - `resolve_adapt_config(raw: dict, overrides: dict | None = None) -> AdaptConfig` — deep-merges over defaults, validates, returns dataclass.
  - Module constant `DEFAULTS: dict`.
- Validation: `method ∈ {mlp_head,finetune,engine}`; `task ∈ {regression,classification}`; `backbone`, `csv` non-empty strings; `targets` is `"auto"` or non-empty list of strings; `split` fractions in `(0,1)`, `seed` int; `finetune`-only training keys (`backbone_lr`, `grad_clip`) rejected for other methods; unknown top-level keys rejected.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_adapt_config.py
from __future__ import annotations

import pytest

from qchem_gnn.adapt.config import AdaptConfig, AdaptConfigError, resolve_adapt_config


def _base(**over):
    cfg = {"command": "adapt", "method": "finetune", "backbone": "bb.pt",
           "task": "regression", "dataset": {"csv": "d.csv", "targets": ["y"]},
           "outputs": {"adapter": "out.pt"}}
    cfg.update(over)
    return cfg


def test_resolve_fills_defaults():
    cfg = resolve_adapt_config(_base())
    assert isinstance(cfg, AdaptConfig)
    assert cfg.split["test_frac"] == 0.2
    assert cfg.training["epochs"] >= 1
    assert cfg.targets == ["y"]


def test_unknown_method_rejected():
    with pytest.raises(AdaptConfigError):
        resolve_adapt_config(_base(method="lora"))


def test_unknown_task_rejected():
    with pytest.raises(AdaptConfigError):
        resolve_adapt_config(_base(task="ranking"))


def test_backbone_lr_rejected_for_mlp_head():
    cfg = _base(method="mlp_head")
    cfg["training"] = {"backbone_lr": 1e-4}
    with pytest.raises(AdaptConfigError):
        resolve_adapt_config(cfg)


def test_overrides_applied():
    cfg = resolve_adapt_config(_base(), overrides={"training": {"epochs": 999}})
    assert cfg.training["epochs"] == 999


def test_missing_backbone_rejected():
    cfg = _base(); del cfg["backbone"]
    with pytest.raises(AdaptConfigError):
        resolve_adapt_config(cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_adapt_config.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `config.py`**

```python
# qchem_gnn/adapt/config.py
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

VALID_METHODS = {"mlp_head", "finetune", "engine"}
VALID_TASKS = {"regression", "classification"}
VALID_TOP = {"command", "method", "backbone", "task", "dataset", "adapter",
             "training", "split", "outputs", "sweep"}
FINETUNE_ONLY_TRAINING = {"backbone_lr", "grad_clip"}

DEFAULTS = {
    "adapter": {"hidden_dims": [128, 64], "dropout": 0.1},
    "training": {"epochs": 200, "head_lr": 1.0e-3, "backbone_lr": 5.0e-5,
                 "batch_size": 64, "patience": 30, "grad_clip": 1.0, "seed": 42},
    "split": {"test_frac": 0.2, "val_frac": 0.25, "seed": 42, "stratify": True},
    "outputs": {"adapter": None, "report": None},
}


class AdaptConfigError(ValueError):
    pass


@dataclass
class AdaptConfig:
    method: str
    backbone: str
    task: str
    csv: str
    smiles_col: str | None
    targets: object
    adapter: dict
    training: dict
    split: dict
    outputs: dict
    sweep: dict | None


def _deep_merge(base, override):
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def resolve_adapt_config(raw: dict, overrides: dict | None = None) -> AdaptConfig:
    cfg = _deep_merge(raw, overrides or {})

    extra = set(cfg) - VALID_TOP
    if extra:
        raise AdaptConfigError(f"unknown top-level keys: {sorted(extra)}")

    method = cfg.get("method")
    if method not in VALID_METHODS:
        raise AdaptConfigError(f"method must be one of {sorted(VALID_METHODS)}")
    task = cfg.get("task", "regression")
    if task not in VALID_TASKS:
        raise AdaptConfigError(f"task must be one of {sorted(VALID_TASKS)}")

    backbone = cfg.get("backbone")
    if not isinstance(backbone, str) or not backbone.strip():
        raise AdaptConfigError("backbone must be a non-empty path string")

    dataset = cfg.get("dataset") or {}
    csv = dataset.get("csv")
    if not isinstance(csv, str) or not csv.strip():
        raise AdaptConfigError("dataset.csv must be a non-empty path string")
    targets = dataset.get("targets", "auto")
    if targets != "auto":
        if not isinstance(targets, list) or not targets or not all(isinstance(t, str) for t in targets):
            raise AdaptConfigError("dataset.targets must be 'auto' or a non-empty list of strings")
    smiles_col = dataset.get("smiles_col")

    adapter = _deep_merge(DEFAULTS["adapter"], cfg.get("adapter"))
    training = _deep_merge(DEFAULTS["training"], cfg.get("training"))
    split = _deep_merge(DEFAULTS["split"], cfg.get("split"))
    outputs = _deep_merge(DEFAULTS["outputs"], cfg.get("outputs"))

    if method != "finetune":
        bad = FINETUNE_ONLY_TRAINING & set((cfg.get("training") or {}))
        if bad:
            raise AdaptConfigError(f"training keys {sorted(bad)} only valid for method 'finetune'")

    for key in ("test_frac", "val_frac"):
        v = split[key]
        if not (isinstance(v, (int, float)) and 0.0 < float(v) < 1.0):
            raise AdaptConfigError(f"split.{key} must be in (0, 1)")
    if not isinstance(split["seed"], int):
        raise AdaptConfigError("split.seed must be an integer")

    sweep = cfg.get("sweep")
    if sweep is not None:
        _validate_sweep(sweep)

    return AdaptConfig(method=method, backbone=backbone, task=task, csv=csv,
                       smiles_col=smiles_col, targets=targets, adapter=adapter,
                       training=training, split=split, outputs=outputs, sweep=sweep)


def _validate_sweep(sweep: dict) -> None:
    if not isinstance(sweep, dict) or "grid" not in sweep:
        raise AdaptConfigError("sweep must be a mapping with a 'grid' key")
    grid = sweep["grid"]
    if not isinstance(grid, dict) or not grid:
        raise AdaptConfigError("sweep.grid must be a non-empty mapping")
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise AdaptConfigError(f"sweep.grid['{key}'] must be a non-empty list")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_adapt_config.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/adapt/config.py tests/test_adapt_config.py
git commit -m "feat(adapt): self-contained AdaptConfig schema with validation"
```

---

### Task 9: `adapt/runner.py` — single-run orchestration

**Files:**
- Create: `qchem_gnn/adapt/runner.py`
- Test: `tests/test_adapt_runner.py`

**Interfaces:**
- Consumes: `resolve_adapt_config`/`AdaptConfig` (Task 8), `load_dataset`/`stratified_split` (Task 1), `load_backbone` (Task 2), `get_method` (Task 7).
- Produces: `run(cfg: AdaptConfig) -> dict` — executes one adaptation; writes the adapter to `cfg.outputs["adapter"]` if set; writes a JSON report to `cfg.outputs["report"]` if set; returns `{"test_metrics": ..., "split": {n_train,n_val,n_test}, "log": ...}`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_adapt_runner.py
from __future__ import annotations

import json

import pandas as pd

from qchem_gnn.adapt.config import resolve_adapt_config
from qchem_gnn.adapt.runner import run
from qchem_gnn.checkpoint import save_checkpoint
from qchem_gnn.model import MolecularQuantumGNN

MODEL_CONFIG = dict(atom_vocab_size=128, bond_vocab_size=8, hidden_dim=16,
                    num_message_passing_steps=2, graph_targets=2)


def _setup(tmp_path):
    model = MolecularQuantumGNN(**MODEL_CONFIG)
    bb = tmp_path / "bb.pt"
    save_checkpoint(bb, {"model_state_dict": model.state_dict(), "model_config": MODEL_CONFIG})
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "CCN", "CCCCO", "c1ccncc1", "CCCl",
              "CCBr", "CCCCC", "CCOC", "CC#N", "CCS", "CCCCCC", "c1ccccc1O", "CCCN"]
    y = [float(i) for i in range(len(smiles))]
    csv = tmp_path / "d.csv"
    pd.DataFrame({"smiles": smiles, "y": y}).to_csv(csv, index=False)
    return bb, csv


def test_run_writes_adapter_and_report(tmp_path):
    bb, csv = _setup(tmp_path)
    out = tmp_path / "adapter.pt"
    report = tmp_path / "report.json"
    raw = {"command": "adapt", "method": "mlp_head", "backbone": str(bb),
           "task": "regression", "dataset": {"csv": str(csv), "targets": ["y"]},
           "training": {"epochs": 5}, "outputs": {"adapter": str(out), "report": str(report)}}
    cfg = resolve_adapt_config(raw)
    result = run(cfg)
    assert out.exists() and report.exists()
    assert "test_metrics" in result
    payload = json.loads(report.read_text())
    assert payload["split"]["n_train"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapt_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `runner.py`**

```python
# qchem_gnn/adapt/runner.py
from __future__ import annotations

import json
from pathlib import Path

from .backbone import load_backbone
from .config import AdaptConfig
from .data import load_dataset, stratified_split
from .registry import get_method


def run(cfg: AdaptConfig) -> dict:
    data = load_dataset(cfg.csv, cfg.smiles_col, cfg.targets, cfg.task)
    train_idx, val_idx, test_idx = stratified_split(
        data.targets, cfg.task,
        test_frac=cfg.split["test_frac"], val_frac=cfg.split["val_frac"],
        seed=int(cfg.split["seed"]), stratify=bool(cfg.split["stratify"]),
    )
    backbone, model_config = load_backbone(cfg.backbone)
    method = get_method(cfg.method)

    result = method.train(backbone, model_config, data, train_idx, val_idx, test_idx, cfg)

    meta = {
        "backbone_ckpt": str(Path(cfg.backbone).resolve()),
        "target_names": data.target_names,
        "task": cfg.task,
        "training_info": {
            "dataset": Path(cfg.csv).name,
            "method": cfg.method,
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "target_col": ", ".join(data.target_names),
        },
    }
    if cfg.outputs.get("adapter"):
        method.save(cfg.outputs["adapter"], result, meta)

    summary = {
        "method": cfg.method,
        "test_metrics": result.test_metrics,
        "split": {"n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx)},
        "log": result.log,
    }
    if cfg.outputs.get("report"):
        rp = Path(cfg.outputs["report"]); rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(summary, indent=2))
    return summary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapt_runner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/adapt/runner.py tests/test_adapt_runner.py
git commit -m "feat(adapt): single-run orchestration (runner)"
```

---

### Task 10: CLI `adapt` command + back-compat shims + `predict_property.py`

**Files:**
- Modify: `qchem_gnn/cli.py` (add `adapt` subparser near line 105 after the `downstream` parser; add dispatch in `main` near line 751)
- Create: `qchem_gnn/adapt/__init__.py` content (re-exports)
- Rewrite: `qchem_gnn/adapters.py` (re-export shim), `qchem_gnn/engine_adapter.py` (re-export shim)
- Modify: `scripts/predict_property.py` (import from `qchem_gnn.adapt`)
- Test: `tests/test_adapt_cli.py`

**Interfaces:**
- Consumes: `run` (Task 9), `resolve_adapt_config`/`load_yaml_config`.
- Produces:
  - `qchem_gnn.adapt.__init__` re-exports: `AdaptConfig`, `resolve_adapt_config`, `run`, `get_method`, `predict_smiles`.
  - `predict_smiles(smiles: list[str], adapter_path, **kw) -> tuple[np.ndarray, list[int]]` (added to `adapt/__init__.py`): reads `adapter_type` from the saved file and dispatches to that method's `load`/`predict`.
  - `cli.main(["adapt", "config.yaml"])` returns `0` and writes the configured adapter.
  - Back-compat: `from qchem_gnn.engine_adapter import EngineAdapterHead, extract_intermediate_embeddings, load_adapter, predict` and `from qchem_gnn.adapters import MLPHead, predict` still import.

- [ ] **Step 1: Write failing test**

```python
# tests/test_adapt_cli.py
from __future__ import annotations

import pandas as pd
import yaml

from qchem_gnn.cli import main
from qchem_gnn.checkpoint import save_checkpoint
from qchem_gnn.model import MolecularQuantumGNN

MODEL_CONFIG = dict(atom_vocab_size=128, bond_vocab_size=8, hidden_dim=16,
                    num_message_passing_steps=2, graph_targets=2)


def test_adapt_cli_runs_and_writes_adapter(tmp_path):
    model = MolecularQuantumGNN(**MODEL_CONFIG)
    bb = tmp_path / "bb.pt"
    save_checkpoint(bb, {"model_state_dict": model.state_dict(), "model_config": MODEL_CONFIG})
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "CCN", "CCCCO", "c1ccncc1", "CCCl"]
    csv = tmp_path / "d.csv"
    pd.DataFrame({"smiles": smiles, "y": [float(i) for i in range(len(smiles))]}).to_csv(csv, index=False)
    out = tmp_path / "adapter.pt"
    cfg = {"command": "adapt", "method": "mlp_head", "backbone": str(bb),
           "task": "regression", "dataset": {"csv": str(csv), "targets": ["y"]},
           "training": {"epochs": 3}, "outputs": {"adapter": str(out)}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    assert main(["adapt", str(cfg_path)]) == 0
    assert out.exists()


def test_back_compat_imports():
    from qchem_gnn.engine_adapter import EngineAdapterHead  # noqa: F401
    from qchem_gnn.adapters import MLPHead  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapt_cli.py -v`
Expected: FAIL (no `adapt` subcommand → `SystemExit`/error)

- [ ] **Step 3: Implement CLI wiring, shims, and `predict_property.py`**

Add the subparser after the `downstream` parser block in `qchem_gnn/cli.py` (around line 121):

```python
    adapt_cmd = subparsers.add_parser("adapt", help="Adapt a frozen/fine-tuned backbone to a property CSV")
    adapt_cmd.add_argument("config", help="YAML config path")
```

Add the dispatch branch in `main` before the final `parser.error` (around line 751):

```python
    if args.command == "adapt":
        return run_adapt(args)
```

Add the handler function in `qchem_gnn/cli.py` (near the other `run_*` functions):

```python
def run_adapt(args) -> int:
    from .adapt.config import resolve_adapt_config
    from .adapt.runner import run as run_adapt_single
    from .adapt.sweep import run_sweep
    raw = load_yaml_config(args.config)
    cfg = resolve_adapt_config(raw)
    if cfg.sweep is not None:
        run_sweep(cfg)
    else:
        summary = run_adapt_single(cfg)
        print(summary)
    return 0
```

> NOTE: `run_sweep` is created in Task 11. To keep Task 10 self-contained and tests green, add a temporary stub module so the import resolves: create `qchem_gnn/adapt/sweep.py` containing only:
> ```python
> def run_sweep(cfg):
>     raise NotImplementedError("sweep arrives in Task 11")
> ```
> Task 11 replaces this file with the real implementation.

`qchem_gnn/adapt/__init__.py` (re-exports + unified predict dispatch):

```python
# qchem_gnn/adapt/__init__.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import AdaptConfig, AdaptConfigError, resolve_adapt_config
from .registry import METHODS, get_method
from .runner import run


def predict_smiles(smiles: list[str], adapter_path, **kw) -> tuple[np.ndarray, list[int]]:
    header = torch.load(Path(adapter_path), map_location="cpu", weights_only=False)
    adapter_type = header.get("adapter_type", "engine")
    method_cls = METHODS[adapter_type]
    loaded = method_cls.load(adapter_path)
    return method_cls.predict(loaded, smiles, **kw)


__all__ = ["AdaptConfig", "AdaptConfigError", "resolve_adapt_config",
           "run", "get_method", "METHODS", "predict_smiles"]
```

Rewrite `qchem_gnn/engine_adapter.py` as a shim (replace entire file):

```python
# qchem_gnn/engine_adapter.py
"""Back-compat shim. Implementation moved to qchem_gnn.adapt.methods.engine."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .adapt.backbone import build_graphs, embed_per_layer  # noqa: F401
from .adapt.methods.engine import EngineAdapterHead, EngineMethod


def extract_intermediate_embeddings(smiles_list, model, batch_size=256):
    graphs, valid_idx = build_graphs(smiles_list)
    layer_embs = embed_per_layer(graphs, model, batch_size=batch_size)
    final = np.empty((len(graphs), 0))
    return layer_embs, final, valid_idx


def load_adapter(path):
    loaded = EngineMethod.load(path)
    meta = {
        "label_mu": loaded.payload["label_norm"]["mu"][0],
        "label_sig": loaded.payload["label_norm"]["sigma"][0],
        "layer_scalers": [(np.array(mu), np.array(sig)) for mu, sig in loaded.payload["layer_scalers"]],
        "backbone_checkpoint": loaded.payload["backbone_ckpt"],
        "training_info": loaded.payload.get("training_info", {}),
    }
    adapter = EngineAdapterHead(loaded.payload["hidden_dim"], loaded.payload["num_layers"],
                                output_dim=loaded.payload["output_dim"])
    adapter.load_state_dict(loaded.payload["adapter_state"])
    adapter.eval()
    return adapter, meta


def predict(smiles_list, backbone_ckpt, adapter_path, mode="ensemble", exit_tolerance=0.05, batch_size=256):
    loaded = EngineMethod.load(adapter_path)
    preds, valid_idx = EngineMethod.predict(loaded, smiles_list, mode=mode,
                                            exit_tolerance=exit_tolerance, batch_size=batch_size)
    return preds[:, 0], valid_idx
```

Rewrite `qchem_gnn/adapters.py` as a shim (replace entire file):

```python
# qchem_gnn/adapters.py
"""Back-compat shim. Implementation moved to qchem_gnn.adapt."""
from __future__ import annotations

from .adapt import predict_smiles
from .adapt.backbone import build_graphs, embed_final as _embed_final
from .adapt.methods.base import MLPHead


def embed_smiles(smiles_list, model, batch_size=256):
    graphs, valid_idx = build_graphs(smiles_list)
    return _embed_final(graphs, model, batch_size=batch_size), valid_idx


def predict(smiles_list, adapter_path, batch_size=256):
    return predict_smiles(smiles_list, adapter_path, batch_size=batch_size)
```

Update `scripts/predict_property.py` — replace its import line and prediction call so it uses the unified dispatcher:

```python
# near the top, replace the qchem_gnn.adapters import with:
from qchem_gnn.adapt import predict_smiles
```
and replace the body's predict call with:
```python
    preds, valid_idx = predict_smiles(
        smiles_list, adapter_path,
        mode=args.mode, exit_tolerance=args.exit_tol, batch_size=args.batch,
    )
    # preds is [N_valid, T]; for single-target take column 0 for the display table
    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds[:, 0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_adapt_cli.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full suite to confirm no regressions from the shims**

Run: `pytest -q`
Expected: PASS (all existing + new tests)

- [ ] **Step 6: Commit**

```bash
git add qchem_gnn/cli.py qchem_gnn/adapt/__init__.py qchem_gnn/adapt/sweep.py qchem_gnn/adapters.py qchem_gnn/engine_adapter.py scripts/predict_property.py tests/test_adapt_cli.py
git commit -m "feat(adapt): wire 'adapt' CLI command, back-compat shims, unified predict"
```

---

## Phase 2 — Sweep

### Task 11: `adapt/sweep.py` — grid expansion + comparison report

**Files:**
- Rewrite: `qchem_gnn/adapt/sweep.py` (replace the Task 10 stub)
- Modify: `qchem_gnn/adapt/config.py` (set-by-dotted-path helper — add `apply_dotted_overrides`)
- Test: `tests/test_adapt_sweep.py`

**Interfaces:**
- Produces:
  - `expand_grid(grid: dict[str, list]) -> list[dict]` — Cartesian product; each element is a nested override dict reconstructed from dotted keys (e.g. `{"training.epochs": 10}` → `{"training": {"epochs": 10}}`).
  - `run_sweep(cfg: AdaptConfig) -> list[dict]` — for each grid cell: rebuild a raw config from `cfg`, apply the override, re-resolve, run, collect a flat row of `{**override_values, **test_metrics_flat}`; write `cfg.sweep["report"]` CSV if set; print a comparison table.
- In `config.py`: `to_raw_dict(cfg: AdaptConfig) -> dict` (round-trips an `AdaptConfig` back to a raw mapping so overrides can be re-applied and re-validated).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_adapt_sweep.py
from __future__ import annotations

import pandas as pd

from qchem_gnn.adapt.config import resolve_adapt_config
from qchem_gnn.adapt.sweep import expand_grid, run_sweep
from qchem_gnn.checkpoint import save_checkpoint
from qchem_gnn.model import MolecularQuantumGNN

MODEL_CONFIG = dict(atom_vocab_size=128, bond_vocab_size=8, hidden_dim=16,
                    num_message_passing_steps=2, graph_targets=2)


def test_expand_grid_cartesian():
    cells = expand_grid({"training.epochs": [1, 2], "adapter.dropout": [0.0, 0.1]})
    assert len(cells) == 4
    assert {"training": {"epochs": 1}, "adapter": {"dropout": 0.0}} in cells


def test_run_sweep_writes_report(tmp_path):
    model = MolecularQuantumGNN(**MODEL_CONFIG)
    bb = tmp_path / "bb.pt"
    save_checkpoint(bb, {"model_state_dict": model.state_dict(), "model_config": MODEL_CONFIG})
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "CCN", "CCCCO", "c1ccncc1", "CCCl",
              "CCBr", "CCCCC", "CCOC", "CC#N", "CCS", "CCCCCC", "c1ccccc1O", "CCCN"]
    csv = tmp_path / "d.csv"
    pd.DataFrame({"smiles": smiles, "y": [float(i) for i in range(len(smiles))]}).to_csv(csv, index=False)
    report = tmp_path / "sweep.csv"
    raw = {"command": "adapt", "method": "mlp_head", "backbone": str(bb), "task": "regression",
           "dataset": {"csv": str(csv), "targets": ["y"]}, "training": {"epochs": 2},
           "outputs": {"adapter": str(tmp_path / "a.pt")},
           "sweep": {"grid": {"training.epochs": [2, 3]}, "report": str(report)}}
    cfg = resolve_adapt_config(raw)
    rows = run_sweep(cfg)
    assert len(rows) == 2 and report.exists()
    df = pd.read_csv(report)
    assert "test_mae" in df.columns and len(df) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_adapt_sweep.py -v`
Expected: FAIL (`expand_grid` undefined / stub raises `NotImplementedError`)

- [ ] **Step 3: Implement `to_raw_dict`/`apply_dotted_overrides` in `config.py` and the real `sweep.py`**

Append to `qchem_gnn/adapt/config.py`:

```python
def to_raw_dict(cfg: AdaptConfig) -> dict:
    raw = {
        "command": "adapt", "method": cfg.method, "backbone": cfg.backbone, "task": cfg.task,
        "dataset": {"csv": cfg.csv, "targets": cfg.targets},
        "adapter": deepcopy(cfg.adapter), "training": deepcopy(cfg.training),
        "split": deepcopy(cfg.split), "outputs": deepcopy(cfg.outputs),
    }
    if cfg.smiles_col is not None:
        raw["dataset"]["smiles_col"] = cfg.smiles_col
    return raw
```

```python
# qchem_gnn/adapt/sweep.py
from __future__ import annotations

import csv
import itertools
from pathlib import Path

from .config import AdaptConfig, resolve_adapt_config, to_raw_dict


def _nest(dotted: str, value):
    parts = dotted.split(".")
    out = cur = {}
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
    return out


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def expand_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    cells = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        override: dict = {}
        for k, v in zip(keys, combo):
            override = _merge(override, _nest(k, v))
        cells.append(override)
    return cells


def _flatten_metrics(metrics: dict) -> dict:
    flat = {}
    for k in ("mae", "rmse", "r2", "auc", "accuracy", "f1"):
        if k in metrics:
            flat[f"test_{k}"] = round(metrics[k], 4)
    return flat


def run_sweep(cfg: AdaptConfig) -> list[dict]:
    from .runner import run  # local import avoids cycle
    grid = cfg.sweep["grid"]
    report_path = cfg.sweep.get("report")
    base_raw = to_raw_dict(cfg)
    cells = expand_grid(grid)

    rows: list[dict] = []
    for i, override in enumerate(cells):
        raw = _merge(base_raw, override)
        if base_raw["outputs"].get("adapter"):
            stem = Path(base_raw["outputs"]["adapter"])
            raw = _merge(raw, {"outputs": {"adapter": str(stem.with_name(f"{stem.stem}_cell{i}{stem.suffix}"))}})
        cell_cfg = resolve_adapt_config(raw)
        summary = run(cell_cfg)
        flat_over = {k: v for k, v in _flat_items(override)}
        rows.append({**flat_over, **_flatten_metrics(summary["test_metrics"])})

    if rows:
        cols = list(rows[0].keys())
        if report_path:
            rp = Path(report_path); rp.parent.mkdir(parents=True, exist_ok=True)
            with rp.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader(); w.writerows(rows)
        _print_table(cols, rows)
    return rows


def _flat_items(d: dict, prefix: str = ""):
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from _flat_items(v, key)
        else:
            yield key, v


def _print_table(cols, rows):
    widths = {c: max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(str(c).rjust(widths[c]) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).rjust(widths[c]) for c in cols))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_adapt_sweep.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/adapt/sweep.py qchem_gnn/adapt/config.py tests/test_adapt_sweep.py
git commit -m "feat(adapt): grid sweep with CSV comparison report and printed table"
```

---

### Task 12: Example configs + remove old scripts + update tutorial

**Files:**
- Create: `configs/adapt_mlp_head_solubility.yaml`, `configs/adapt_finetune_solubility.yaml`, `configs/adapt_engine_solubility.yaml`, `configs/adapt_engine_epoch_sweep.yaml`
- Delete: `scripts/train_mlp_head.py`, `scripts/train_finetune.py`, `scripts/train_engine_adapter.py`
- Modify: `docs/tutorials/engine_solubility_tutorial.md` (commands → `python -m qchem_gnn.cli adapt ...`)
- Test: `tests/test_adapt_example_configs.py`

**Interfaces:**
- Consumes: `resolve_adapt_config`, `load_yaml_config`.
- Produces: example YAML files that parse and validate.

- [ ] **Step 1: Write failing test**

```python
# tests/test_adapt_example_configs.py
from __future__ import annotations

from pathlib import Path

import pytest

from qchem_gnn.adapt.config import resolve_adapt_config
from qchem_gnn.config import load_yaml_config

CONFIGS = [
    "configs/adapt_mlp_head_solubility.yaml",
    "configs/adapt_finetune_solubility.yaml",
    "configs/adapt_engine_solubility.yaml",
    "configs/adapt_engine_epoch_sweep.yaml",
]


@pytest.mark.parametrize("path", CONFIGS)
def test_example_config_resolves(path):
    cfg = resolve_adapt_config(load_yaml_config(Path(path)))
    assert cfg.method in {"mlp_head", "finetune", "engine"}


def test_sweep_config_has_grid():
    cfg = resolve_adapt_config(load_yaml_config(Path("configs/adapt_engine_epoch_sweep.yaml")))
    assert cfg.sweep is not None and "grid" in cfg.sweep
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapt_example_configs.py -v`
Expected: FAIL (config files do not exist → `ConfigError`/`FileNotFoundError`)

- [ ] **Step 3: Create configs, delete old scripts, update tutorial**

`configs/adapt_mlp_head_solubility.yaml`:
```yaml
command: adapt
method: mlp_head
backbone: runs/example_contrastive.pt
task: regression
dataset:
  csv: data/delaney-processed.csv
  targets: [measured log solubility in mols per litre]
adapter: {hidden_dims: [128, 64], dropout: 0.1}
training: {epochs: 300, head_lr: 1.0e-3, batch_size: 128, patience: 40, seed: 42}
split: {test_frac: 0.2, val_frac: 0.25, seed: 42, stratify: true}
outputs: {adapter: runs/mlp_head_solubility.pt, report: runs/mlp_head_metrics.json}
```

`configs/adapt_finetune_solubility.yaml`:
```yaml
command: adapt
method: finetune
backbone: runs/example_contrastive.pt
task: regression
dataset:
  csv: data/delaney-processed.csv
  targets: [measured log solubility in mols per litre]
adapter: {hidden_dims: [128, 64], dropout: 0.1}
training: {epochs: 200, head_lr: 1.0e-3, backbone_lr: 5.0e-5, batch_size: 64, patience: 30, grad_clip: 1.0, seed: 42}
split: {test_frac: 0.2, val_frac: 0.25, seed: 42, stratify: true}
outputs: {adapter: runs/finetune_solubility.pt, report: runs/finetune_metrics.json}
```

`configs/adapt_engine_solubility.yaml`:
```yaml
command: adapt
method: engine
backbone: runs/example_contrastive.pt
task: regression
dataset:
  csv: data/delaney-processed.csv
  targets: [measured log solubility in mols per litre]
training: {epochs: 400, head_lr: 3.0e-3, seed: 42}
split: {test_frac: 0.2, val_frac: 0.25, seed: 42, stratify: true}
outputs: {adapter: runs/engine_solubility.pt, report: runs/engine_metrics.json}
```

`configs/adapt_engine_epoch_sweep.yaml`:
```yaml
command: adapt
method: engine
backbone: runs/example_contrastive.pt
task: regression
dataset:
  csv: data/delaney-processed.csv
  targets: [measured log solubility in mols per litre]
training: {epochs: 400, head_lr: 3.0e-3, seed: 42}
split: {test_frac: 0.2, val_frac: 0.25, seed: 42, stratify: true}
outputs: {adapter: runs/engine_sweep.pt}
sweep:
  grid:
    training.epochs: [10, 50, 100, 200, 400]
  report: runs/engine_epoch_sweep.csv
```

Delete the three old scripts:
```bash
git rm scripts/train_mlp_head.py scripts/train_finetune.py scripts/train_engine_adapter.py
```

In `docs/tutorials/engine_solubility_tutorial.md`, replace every `python scripts/train_engine_adapter.py ...` invocation with:
```bash
python -m qchem_gnn.cli adapt configs/adapt_engine_solubility.yaml
```
and replace the by-hand epoch-sweep description with:
```bash
python -m qchem_gnn.cli adapt configs/adapt_engine_epoch_sweep.yaml
# writes runs/engine_epoch_sweep.csv and prints the comparison table
```
and replace `python scripts/predict_solubility.py --adapter ...` with:
```bash
python scripts/predict_property.py --adapter runs/engine_solubility.pt "CCO" "c1ccccc1"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapt_example_configs.py -v`
Expected: PASS (5 parametrized + 1)

- [ ] **Step 5: Commit**

```bash
git add configs/adapt_*.yaml docs/tutorials/engine_solubility_tutorial.md tests/test_adapt_example_configs.py
git rm scripts/train_mlp_head.py scripts/train_finetune.py scripts/train_engine_adapter.py
git commit -m "feat(adapt): example configs, remove standalone scripts, update tutorial"
```

---

## Phase 3 — Multi-target + classification

### Task 13: Classification training path (loss + metrics dispatch)

**Files:**
- Modify: `qchem_gnn/adapt/methods/mlp_head.py`, `finetune.py`, `engine.py` (select loss + metrics by `data.task`; skip label normalization for classification; apply sigmoid at predict time for classification)
- Modify: `qchem_gnn/adapt/data.py` (`load_dataset` for classification keeps targets as 0/1 floats; no change needed beyond task pass-through, but add a guard that classification targets are within `{0,1}` per column)
- Test: `tests/test_adapt_classification.py`

**Interfaces:**
- Consumes: `classification_metrics` (Task 3).
- Produces: methods honor `data.task == "classification"`: loss `nn.BCEWithLogitsLoss`; no `LabelNormalizer`; saved payload records `task`; `predict` returns probabilities (`sigmoid`) for classification. A saved `label_norm` is replaced by `label_norm: None` for classification; `predict`/`save` branch on `payload["task"]`.

Design note to implementers: introduce two small shared helpers in `methods/base.py` to avoid divergence across the three methods —
```python
def make_loss(task: str) -> nn.Module:
    return nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()

def postprocess(task: str, raw: np.ndarray, norm) -> np.ndarray:
    if task == "classification":
        return 1.0 / (1.0 + np.exp(-raw))
    return norm.inverse(raw)
```
Each method: when `task == "classification"`, set `norm = None`, train targets are the raw 0/1 (no transform), evaluation uses `classification_metrics` on `postprocess(...)`, and `predict` applies `postprocess`. When `task == "regression"`, behavior is unchanged.

- [ ] **Step 1: Write failing test**

```python
# tests/test_adapt_classification.py
from __future__ import annotations

import numpy as np
import pandas as pd

from qchem_gnn.adapt.config import resolve_adapt_config
from qchem_gnn.adapt.runner import run
from qchem_gnn.adapt import predict_smiles
from qchem_gnn.checkpoint import save_checkpoint
from qchem_gnn.model import MolecularQuantumGNN

MODEL_CONFIG = dict(atom_vocab_size=128, bond_vocab_size=8, hidden_dim=16,
                    num_message_passing_steps=2, graph_targets=2)


def test_classification_run_reports_auc_and_predicts_probabilities(tmp_path):
    model = MolecularQuantumGNN(**MODEL_CONFIG)
    bb = tmp_path / "bb.pt"
    save_checkpoint(bb, {"model_state_dict": model.state_dict(), "model_config": MODEL_CONFIG})
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "CCN", "CCCCO", "c1ccncc1", "CCCl",
              "CCBr", "CCCCC", "CCOC", "CC#N", "CCS", "CCCCCC", "c1ccccc1O", "CCCN"]
    label = [i % 2 for i in range(len(smiles))]
    csv = tmp_path / "d.csv"
    pd.DataFrame({"smiles": smiles, "active": label}).to_csv(csv, index=False)
    out = tmp_path / "clf.pt"
    raw = {"command": "adapt", "method": "mlp_head", "backbone": str(bb), "task": "classification",
           "dataset": {"csv": str(csv), "targets": ["active"]}, "training": {"epochs": 5},
           "outputs": {"adapter": str(out)}}
    cfg = resolve_adapt_config(raw)
    summary = run(cfg)
    assert "auc" in summary["test_metrics"]
    preds, valid = predict_smiles(["CCO", "c1ccccc1"], out)
    assert preds.shape == (2, 1)
    assert ((preds >= 0.0) & (preds <= 1.0)).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapt_classification.py -v`
Expected: FAIL (regression path normalizes labels and returns `classification_metrics`-less summary; AUC absent)

- [ ] **Step 3: Implement task branching**

Add the two helpers to `qchem_gnn/adapt/methods/base.py`:

```python
import numpy as np
from torch import nn

def make_loss(task: str) -> nn.Module:
    return nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()

def postprocess(task: str, raw, norm):
    if task == "classification":
        return 1.0 / (1.0 + np.exp(-raw))
    return norm.inverse(raw)
```

In each of `mlp_head.py`, `finetune.py`, `engine.py`, apply this pattern (shown for `mlp_head.py`; mirror in the other two):

```python
from ..metrics import classification_metrics, regression_metrics
from .base import make_loss, postprocess

# inside train():
task = data.task
norm = None if task == "classification" else LabelNormalizer.fit(y[train_idx])

def prep_y(arr):
    return arr if task == "classification" else norm.transform(arr)

# build the target tensors with prep_y(...)
crit = make_loss(task)

# validation / test predictions:
def eval_pred(raw):  # raw model output as numpy
    return postprocess(task, raw, norm)

# choose metric fn:
metric_fn = classification_metrics if task == "classification" else regression_metrics
# ... test_metrics = metric_fn(y[test_idx], eval_pred(raw_test))

# payload:
payload["task"] = task
payload["label_norm"] = None if norm is None else norm.to_dict()
```

In `predict` of each method, branch:
```python
norm = None if s.get("task") == "classification" else LabelNormalizer.from_dict(s["label_norm"])
preds = postprocess(s.get("task", "regression"), raw_preds, norm)
```

For the **engine** method's validation/early-exit during classification: validation monitors negative AUC (lower is better → track `-auc`) or simply uses BCE-consistent ranking; to keep selection logic uniform, monitor `mae`-style proxy by treating `1 - auc` as the "val score" (lower better). Concretely in `engine.py`:
```python
if task == "classification":
    val_score = 1.0 - classification_metrics(y[val_idx], eval_pred(adapter.predict_ensemble(va)))["auc"]
else:
    val_score = float(np.mean(np.abs(eval_pred(adapter.predict_ensemble(va)) - y[val_idx])))
# keep best on min(val_score)
```
Apply the analogous `val_score` selection in `mlp_head.py` and `finetune.py`.

In `qchem_gnn/adapt/data.py::load_dataset`, after building `targets_arr`, add the classification guard:
```python
    if task == "classification" and targets_arr.size:
        uniq = np.unique(targets_arr)
        if not np.all(np.isin(uniq, [0.0, 1.0])):
            raise ValueError("classification targets must be binary 0/1 per column")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapt_classification.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/adapt/methods/*.py qchem_gnn/adapt/data.py tests/test_adapt_classification.py
git commit -m "feat(adapt): classification task path (BCE loss, AUC/acc/F1, prob outputs)"
```

---

### Task 14: Multi-target regression end-to-end + report formatting

**Files:**
- Test: `tests/test_adapt_multitarget.py`
- Modify (if needed): `qchem_gnn/adapt/runner.py` (ensure per-target metrics surface in the JSON report — they already do via `test_metrics["per_target"]`; add `target_names` to the summary)

**Interfaces:**
- Produces: `run` summary includes `"target_names"`; multi-target regression produces `targets.shape[1] == T` predictions end-to-end across all three methods.

- [ ] **Step 1: Write failing test**

```python
# tests/test_adapt_multitarget.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qchem_gnn.adapt import predict_smiles
from qchem_gnn.adapt.config import resolve_adapt_config
from qchem_gnn.adapt.runner import run
from qchem_gnn.checkpoint import save_checkpoint
from qchem_gnn.model import MolecularQuantumGNN

MODEL_CONFIG = dict(atom_vocab_size=128, bond_vocab_size=8, hidden_dim=16,
                    num_message_passing_steps=3, graph_targets=2)


@pytest.mark.parametrize("method", ["mlp_head", "finetune", "engine"])
def test_multitarget_regression(tmp_path, method):
    model = MolecularQuantumGNN(**MODEL_CONFIG)
    bb = tmp_path / "bb.pt"
    save_checkpoint(bb, {"model_state_dict": model.state_dict(), "model_config": MODEL_CONFIG})
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "CCN", "CCCCO", "c1ccncc1", "CCCl",
              "CCBr", "CCCCC", "CCOC", "CC#N", "CCS", "CCCCCC", "c1ccccc1O", "CCCN"]
    n = len(smiles)
    csv = tmp_path / "d.csv"
    pd.DataFrame({"smiles": smiles,
                  "a": np.linspace(-3, 3, n),
                  "b": np.linspace(0, 10, n)}).to_csv(csv, index=False)
    out = tmp_path / f"{method}.pt"
    raw = {"command": "adapt", "method": method, "backbone": str(bb), "task": "regression",
           "dataset": {"csv": str(csv), "targets": ["a", "b"]},
           "training": {"epochs": 5}, "outputs": {"adapter": str(out)}}
    cfg = resolve_adapt_config(raw)
    summary = run(cfg)
    assert len(summary["test_metrics"]["per_target"]) == 2
    assert summary["target_names"] == ["a", "b"]
    preds, valid = predict_smiles(["CCO", "c1ccccc1"], out)
    assert preds.shape == (2, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapt_multitarget.py -v`
Expected: FAIL on `summary["target_names"]` KeyError (and possibly engine early-exit shape if not handled)

- [ ] **Step 3: Add `target_names` to the runner summary**

In `qchem_gnn/adapt/runner.py`, add to the `summary` dict:
```python
        "target_names": data.target_names,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapt_multitarget.py -v`
Expected: PASS (3 parametrized)

- [ ] **Step 5: Run the whole suite**

Run: `pytest -q`
Expected: PASS (all tests, no regressions)

- [ ] **Step 6: Commit**

```bash
git add qchem_gnn/adapt/runner.py tests/test_adapt_multitarget.py
git commit -m "feat(adapt): multi-target regression end-to-end across all methods"
```

---

## Self-Review (completed during planning)

**Spec coverage:**
- `adapt/` package + method protocol → Tasks 1–9.
- Single source of truth for split/column detection → Task 1 (consumed everywhere).
- Config schema + validation (own module, not legacy `config.py`) → Task 8.
- CLI `adapt` command → Task 10.
- Back-compat shims for `adapters.py`/`engine_adapter.py` → Task 10.
- Replace 3 scripts with example configs; `downstream` untouched; tutorial updated → Task 12.
- Built-in sweep with comparison report → Task 11.
- Multi-target + classification → Tasks 13–14.
- Tests (data/split/registry/config/sweep + per-method smoke + classification + multi-target) → every task.
- `predict_property.py` dispatches on `adapter_type` → Task 10.

**Placeholder scan:** No "TBD"/"add error handling"/"similar to" left. The Task 10 `sweep.py` stub is intentional and explicitly replaced in Task 11.

**Type consistency:** `AdaptData` fields, `MLPHead(input_dim, output_dim, hidden_dims, dropout)`, `TrainResult{payload,test_metrics,log}`, `LoadedAdapter{adapter_type,payload}`, method `train(self, backbone, model_config, data, train_idx, val_idx, test_idx, cfg)` and static `load`/`predict` signatures are consistent across Tasks 4–9 and the registry/runner. Saved `adapter_type` strings (`mlp_head`/`finetune`/`engine`) match the registry keys used by `predict_smiles`.
