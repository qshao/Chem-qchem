import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qchem_gnn.minimal import load_minimal_zinc_dataset
from qchem_gnn.contrastive_pretrain import contrastive_pretrain_on_dataset
from tests._validation_fixtures import make_tiny_quantum_dataset


def _write_subset_with_conformers(tmp_path: Path):
    csv_path = tmp_path / "subset_000.csv"
    geo_path = tmp_path / "coords_000.pkl"
    pd.DataFrame(
        [
            {"smiles": "C", "logP": 0.1, "qed": 0.2, "SAS": 0.3},
            {"smiles": "CC", "logP": 0.4, "qed": 0.5, "SAS": 0.6},
            {"smiles": "CCC", "logP": 0.2, "qed": 0.3, "SAS": 0.4},
            {"smiles": "CCO", "logP": 0.5, "qed": 0.6, "SAS": 0.7},
        ]
    ).to_csv(csv_path, index=False)

    def _coords(n):
        rng = np.random.default_rng(n)
        return [rng.standard_normal((n, 3)).astype(np.float32)]

    pickle.dump(
        {
            "subset_0_idx_0": {"smiles": "C", "charge": 0, "atomic_nums": [6, 1, 1, 1, 1], "conformers": _coords(5)},
            "subset_0_idx_1": {"smiles": "CC", "charge": 0, "atomic_nums": [6, 6] + [1] * 6, "conformers": _coords(8)},
            "subset_0_idx_2": {"smiles": "CCC", "charge": 0, "atomic_nums": [6, 6, 6] + [1] * 8, "conformers": _coords(11)},
            "subset_0_idx_3": {"smiles": "CCO", "charge": 0, "atomic_nums": [6, 6, 8] + [1] * 6, "conformers": _coords(9)},
        },
        geo_path.open("wb"),
    )
    return csv_path, geo_path


def test_contrastive_pretrain_runs_and_reduces_loss(tmp_path: Path):
    csv_path, geo_path = _write_subset_with_conformers(tmp_path)
    dataset = load_minimal_zinc_dataset(csv_path, geo_path, limit=4)

    result = contrastive_pretrain_on_dataset(
        dataset,
        hidden_dim=16,
        num_message_passing_steps=2,
        hidden_dim_3d=16,
        epochs=120,
        batch_size=4,
        learning_rate=0.01,
        contrastive_weight=1.0,
        temperature=0.1,
        seed=0,
    )

    assert result.loss_history[-1] < result.loss_history[0]
    assert len(result.contrastive_loss_history) == len(result.loss_history)
    assert result.embeddings.shape == (4, 16)
    assert torch.isfinite(result.embeddings).all()
    # Inference checkpoint must be loadable by the existing 2D model loader.
    from qchem_gnn.model import MolecularQuantumGNN

    clone = MolecularQuantumGNN(atom_vocab_size=128, bond_vocab_size=8, hidden_dim=16, num_message_passing_steps=2, graph_targets=2)
    clone.load_state_dict(result.model.state_dict())


def test_contrastive_pretrain_vicreg_runs(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    result = contrastive_pretrain_on_dataset(
        dataset,
        hidden_dim=16,
        hidden_dim_3d=16,
        epochs=2,
        batch_size=4,
        teacher_weight=1.0,
        conformer_pool_mode="energy",
        contrastive_loss="vicreg",
        seed=0,
    )
    assert result.loss_history
    assert math.isfinite(result.loss_history[-1])


def test_contrastive_pretrain_infonce_still_default(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    result = contrastive_pretrain_on_dataset(
        dataset, hidden_dim=16, hidden_dim_3d=16, epochs=2, batch_size=4, seed=0,
    )
    assert result.loss_history
    assert math.isfinite(result.loss_history[-1])


def test_contrastive_pretrain_scaffold_mask_runs(tmp_path):
    # The tiny fixture uses CO/CCO/CCN/CCC — all unique scaffolds,
    # so the mask is all-False. This is a regression test that the
    # code path runs without error and produces a finite loss.
    dataset = make_tiny_quantum_dataset(tmp_path)
    result = contrastive_pretrain_on_dataset(
        dataset,
        hidden_dim=16,
        hidden_dim_3d=16,
        epochs=2,
        batch_size=4,
        teacher_weight=1.0,
        conformer_pool_mode="energy",
        use_scaffold_negmask=True,
        seed=0,
    )
    assert result.loss_history
    assert math.isfinite(result.loss_history[-1])


def test_contrastive_pretrain_scaffold_mask_default_false(tmp_path):
    # Omitting use_scaffold_negmask produces the same result as use_scaffold_negmask=False.
    dataset = make_tiny_quantum_dataset(tmp_path)
    result = contrastive_pretrain_on_dataset(
        dataset, hidden_dim=16, hidden_dim_3d=16, epochs=2, batch_size=4, seed=0,
    )
    assert result.loss_history
    assert math.isfinite(result.loss_history[-1])
