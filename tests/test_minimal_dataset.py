import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qchem_gnn.minimal import load_minimal_zinc_dataset, train_on_minimal_dataset
from qchem_gnn.quantum_data import compute_target_normalization


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


def test_load_minimal_zinc_dataset_reads_subset_and_geometry(tmp_path):
    csv_path, geo_path = _write_tiny_subset(tmp_path)

    dataset = load_minimal_zinc_dataset(csv_path, geo_path, limit=2)

    assert len(dataset) == 2
    first = dataset[0]
    assert first.mol_id == "subset_0_idx_0"
    assert first.conformer_count == 1
    assert first.charge == 0
    assert first.graph.num_nodes == 5
    assert first.node_target.shape == (5, 2)
    assert first.edge_target.shape[1] == 1
    assert first.graph_target.shape == (2,)
    assert first.aux_target is not None
    assert first.aux_target.shape == (3,)


def test_train_on_minimal_dataset_returns_embeddings_and_history(tmp_path):
    csv_path, geo_path = _write_tiny_subset(tmp_path)
    dataset = load_minimal_zinc_dataset(csv_path, geo_path, limit=2)

    result = train_on_minimal_dataset(
        dataset,
        hidden_dim=32,
        num_message_passing_steps=2,
        epochs=300,
        learning_rate=0.02,
    )

    assert result.loss_history[0] > result.loss_history[-1]
    assert result.loss_history[-1] < result.loss_history[0] * 0.1
    assert result.embeddings.shape == (2, 32)
    assert torch.isfinite(result.embeddings).all()
    assert result.target_normalization["node_mean"].shape == (2,)


def test_compute_target_normalization_returns_positive_scales(tmp_path):
    csv_path, geo_path = _write_tiny_subset(tmp_path)
    dataset = load_minimal_zinc_dataset(csv_path, geo_path, limit=2)

    stats = compute_target_normalization(dataset)

    assert stats["node_mean"].shape == (2,)
    assert stats["edge_mean"].shape == (1,)
    assert stats["graph_mean"].shape == (2,)
    assert torch.all(stats["node_std"] > 0)
    assert torch.all(stats["edge_std"] > 0)
    assert torch.all(stats["graph_std"] > 0)
