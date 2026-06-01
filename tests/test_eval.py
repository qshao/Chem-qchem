from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import torch

from qchem_gnn.eval import run_fine_tuning, run_linear_probe, run_morgan_baseline, run_sample_efficiency
from qchem_gnn.minimal import load_quantum_zinc_dataset, train_on_minimal_dataset
from qchem_gnn.checkpoint import save_checkpoint, build_checkpoint_state


def test_linear_probe_runs_on_exported_embeddings():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=np.float32)
    split = {"train": [0, 1], "val": [2], "test": [3]}

    metrics = run_linear_probe(embeddings, labels, split)
    sample_efficiency = run_sample_efficiency(embeddings, labels, split, fractions=[0.01, 0.05, 0.1, 1.0])

    assert set(metrics) >= {"mae", "rmse", "r2", "train", "val", "test"}
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["rmse"])
    assert np.isfinite(metrics["r2"])
    assert set(sample_efficiency) == {0.01, 0.05, 0.1, 1.0}
    assert all(set(result) >= {"mae", "rmse", "r2"} for result in sample_efficiency.values())


def test_fine_tuning_and_morgan_baseline_run_on_tiny_dataset(tmp_path: Path):
    csv_path = tmp_path / "subset_000.csv"
    geo_path = tmp_path / "coords_000.pkl"

    pd.DataFrame(
        [
            {"smiles": "C", "logP": 0.1, "qed": 0.2, "SAS": 0.3},
            {"smiles": "CC", "logP": 0.4, "qed": 0.5, "SAS": 0.6},
            {"smiles": "CCC", "logP": 0.7, "qed": 0.8, "SAS": 0.9},
        ]
    ).to_csv(csv_path, index=False)

    with geo_path.open("wb") as handle:
        pickle.dump(
            {
                "subset_0_idx_0": {"smiles": "C", "charge": 0, "symbols": ["C", "H", "H", "H", "H"], "atomic_nums": [6, 1, 1, 1, 1], "conformers": [np.zeros((5, 3), dtype=np.float32)]},
                "subset_0_idx_1": {"smiles": "CC", "charge": 0, "symbols": ["C", "C"] + ["H"] * 6, "atomic_nums": [6, 6] + [1] * 6, "conformers": [np.zeros((8, 3), dtype=np.float32)]},
                "subset_0_idx_2": {"smiles": "CCC", "charge": 0, "symbols": ["C", "C", "C"] + ["H"] * 8, "atomic_nums": [6, 6, 6] + [1] * 8, "conformers": [np.zeros((11, 3), dtype=np.float32)]},
            },
            handle,
        )

    dataset = load_quantum_zinc_dataset(csv_path, geometry_path=geo_path, limit=3)
    result = train_on_minimal_dataset(dataset, epochs=2)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        build_checkpoint_state(
            loss_history=result.loss_history,
            embeddings=result.embeddings,
            model_state_dict=result.model.state_dict(),
            optimizer_state_dict=result.optimizer_state_dict,
            epoch=result.epoch,
            global_step=result.global_step,
            target_normalization=result.target_normalization,
            dataset_config={
                "csv": str(csv_path),
                "dataset_root": None,
                "subset_ids": (),
                "geometry": str(geo_path),
                "results": None,
                "use_results": False,
                "limit": 3,
                "limit_per_shard": 16,
            },
            split_metadata={"subset_ids": []},
            model_config={
                "atom_vocab_size": 128,
                "bond_vocab_size": 8,
                "hidden_dim": 32,
                "num_message_passing_steps": 2,
                "graph_targets": 2,
            },
            run_metadata={"num_examples": len(dataset), "limit": 3, "limit_per_shard": 16, "epochs": 2},
        ),
    )

    fine_tune_metrics = run_fine_tuning(dataset, checkpoint_path)
    morgan_metrics = run_morgan_baseline([example.smiles for example in dataset.examples], np.stack([example.graph_target.numpy() for example in dataset.examples]), {"train": [0, 1], "val": [2], "test": []})

    assert set(fine_tune_metrics) >= {"mae", "rmse", "r2"}
    assert np.isfinite(fine_tune_metrics["mae"])
    assert set(morgan_metrics) >= {"mae", "rmse", "r2"}
    assert np.isfinite(morgan_metrics["mae"])
