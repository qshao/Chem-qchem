from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qchem_gnn.minimal import load_minimal_zinc_dataset
from qchem_gnn.pretrain import pretrain_on_minimal_dataset


def _write_tiny_subset(tmp_path: Path) -> tuple[Path, Path]:
    csv_path = tmp_path / "subset_000.csv"
    geo_path = tmp_path / "coords_000.pkl"

    pd.DataFrame(
        [
            {"smiles": "C", "logP": 0.1, "qed": 0.2, "SAS": 0.3},
            {"smiles": "CC", "logP": 0.4, "qed": 0.5, "SAS": 0.6},
        ]
    ).to_csv(csv_path, index=False)

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
                "smiles": "CC",
                "charge": 0,
                "symbols": ["C", "C"] + ["H"] * 6,
                "atomic_nums": [6, 6] + [1] * 6,
                "conformers": [np.zeros((8, 3), dtype=np.float32)],
            },
        },
        geo_path.open("wb"),
    )

    return csv_path, geo_path


def test_pretrain_on_minimal_dataset_uses_auxiliary_targets(tmp_path: Path):
    csv_path, geo_path = _write_tiny_subset(tmp_path)
    dataset = load_minimal_zinc_dataset(csv_path, geo_path, limit=2)

    result = pretrain_on_minimal_dataset(
        dataset,
        hidden_dim=32,
        num_message_passing_steps=2,
        epochs=300,
        learning_rate=0.02,
        aux_weight=0.1,
    )

    assert result.loss_history[0] > result.loss_history[-1]
    assert result.loss_history[-1] < result.loss_history[0] * 0.1
    assert result.embeddings.shape == (2, 32)
    assert torch.isfinite(result.embeddings).all()
    assert result.aux_normalization is not None
    assert result.aux_normalization["aux_mean"].shape == (3,)
    assert result.aux_normalization["aux_std"].shape == (3,)
    assert result.aux_head_state_dict
