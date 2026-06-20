from pathlib import Path

import h5py
import torch

from qchem_gnn.contrastive_pretrain import contrastive_pretrain_on_dataset
from qchem_gnn.quantum_data import load_quantum_zinc_dataset
from tests._quantum_fixtures import write_synthetic_results_h5


def _make_dataset(tmp_path):
    (tmp_path / "subset_044.csv").write_text(
        "smiles,logP,qed,SAS\n"
        "CO,0.0,0.5,2.0\nCCO,0.1,0.6,2.1\nCCN,0.2,0.7,2.2\nCCC,0.3,0.4,1.9\n"
    )
    # one molecule per group; reuse the same schema for 4 rows
    h5_path = tmp_path / "results_044.h5"
    with h5py.File(h5_path, "w") as handle:
        for idx, smiles, n in [
            (0, "CO", 6), (1, "CCO", 9), (2, "CCN", 10), (3, "CCC", 11)
        ]:
            tmp_single = tmp_path / f"_tmp_{idx}.h5"
            write_synthetic_results_h5(
                tmp_single, mol_id=f"subset_44_idx_{idx}", smiles=smiles, n_atoms=n, seed=idx
            )
            with h5py.File(tmp_single, "r") as src:
                src.copy(src[f"subset_44_idx_{idx}"], handle, f"subset_44_idx_{idx}")
    with h5py.File(h5_path, "r") as handle:
        return load_quantum_zinc_dataset(
            tmp_path / "subset_044.csv",
            geometry_path=tmp_path / "coords_044.pkl",
            limit=4,
            results_handle=handle,
            use_results=True,
        )


def test_contrastive_with_teacher_runs(tmp_path):
    dataset = _make_dataset(tmp_path)
    assert len(dataset.examples) == 4
    result = contrastive_pretrain_on_dataset(
        dataset,
        hidden_dim=16,
        hidden_dim_3d=16,
        total_steps=3,
        batch_size=4,
        teacher_weight=1.0,
        conformer_pool_mode="energy",
        seed=0,
    )
    assert len(result.loss_history) == 3
    assert all(torch.isfinite(torch.tensor(v)) for v in result.loss_history)
    # the 2D student head matches chelpg-only target dim (1)
    assert result.model.atom_head.out_features == 1
