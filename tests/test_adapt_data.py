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
