from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qchem_gnn.dataset_index import iter_dataset_indices
from qchem_gnn.quantum_data import load_quantum_zinc_dataset


def _write_shard(tmp_path: Path, smiles: list[str]) -> tuple[Path, Path, Path]:
    dataset_root = tmp_path / "zinc-250k"
    subsets_dir = dataset_root / "subsets"
    geometries_dir = dataset_root / "geometries"
    results_dir = dataset_root / "results"
    subsets_dir.mkdir(parents=True)
    geometries_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)

    csv_path = subsets_dir / "subset_000.csv"
    geo_path = geometries_dir / "coords_000.pkl"
    results_path = results_dir / "results_000.h5"

    pd.DataFrame({"smiles": smiles, "logP": [0.1] * len(smiles), "qed": [0.2] * len(smiles), "SAS": [0.3] * len(smiles)}).to_csv(csv_path, index=False)
    with geo_path.open("wb") as handle:
        pickle.dump(
            {
                f"subset_0_idx_{idx}": {
                    "smiles": smile,
                    "charge": 0,
                    "symbols": ["C", "H", "H", "H", "H"],
                    "atomic_nums": [6, 1, 1, 1, 1],
                    "conformers": [np.zeros((5, 3), dtype=np.float32)],
                }
                for idx, smile in enumerate(smiles)
            },
            handle,
        )
    results_path.touch()
    return dataset_root, csv_path, geo_path


def test_iter_dataset_indices_uses_persistent_cache(tmp_path: Path, monkeypatch):
    dataset_root, csv_path, geo_path = _write_shard(tmp_path, ["C", "CC"])

    first = list(
        iter_dataset_indices(
            dataset_root,
            subset_ids=[0],
            limit_per_shard=2,
            use_results=True,
            results_root=dataset_root / "results",
        )
    )
    cache_path = dataset_root / ".qchem_gnn_index_cache.json"
    assert cache_path.exists()
    assert first[0].csv_path == csv_path
    assert first[0].geometry_path == geo_path

    def fail_read_csv(*args, **kwargs):
        raise AssertionError("cache should avoid re-reading shard CSV")

    monkeypatch.setattr("qchem_gnn.dataset_index.pd.read_csv", fail_read_csv)

    second = list(
        iter_dataset_indices(
            dataset_root,
            subset_ids=[0],
            limit_per_shard=2,
            use_results=True,
            results_root=dataset_root / "results",
        )
    )
    assert second == first


def test_load_quantum_zinc_dataset_skips_malformed_smiles(tmp_path: Path):
    csv_path = tmp_path / "subset_000.csv"
    geo_path = tmp_path / "coords_000.pkl"

    pd.DataFrame(
        [
            {"smiles": "C", "logP": 0.1, "qed": 0.2, "SAS": 0.3},
            {"smiles": "not-a-smiles", "logP": 0.4, "qed": 0.5, "SAS": 0.6},
            {"smiles": "CC", "logP": 0.7, "qed": 0.8, "SAS": 0.9},
        ]
    ).to_csv(csv_path, index=False)

    with geo_path.open("wb") as handle:
        pickle.dump(
            {
                "subset_0_idx_0": {
                    "smiles": "C",
                    "charge": 0,
                    "symbols": ["C", "H", "H", "H", "H"],
                    "atomic_nums": [6, 1, 1, 1, 1],
                    "conformers": [np.zeros((5, 3), dtype=np.float32)],
                },
                "subset_0_idx_1": {
                    "smiles": "not-a-smiles",
                    "charge": 0,
                    "symbols": ["C"],
                    "atomic_nums": [6],
                    "conformers": [np.zeros((1, 3), dtype=np.float32)],
                },
                "subset_0_idx_2": {
                    "smiles": "CC",
                    "charge": 0,
                    "symbols": ["C", "C"] + ["H"] * 6,
                    "atomic_nums": [6, 6] + [1] * 6,
                    "conformers": [np.zeros((8, 3), dtype=np.float32)],
                },
            },
            handle,
        )

    dataset = load_quantum_zinc_dataset(csv_path, geometry_path=geo_path, limit=3)

    assert len(dataset) == 2
    assert dataset.skipped_mol_ids == ("subset_0_idx_1",)
    assert [example.mol_id for example in dataset.examples] == ["subset_0_idx_0", "subset_0_idx_2"]
