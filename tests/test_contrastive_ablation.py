import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from qchem_gnn.minimal import load_minimal_zinc_dataset
from qchem_gnn.contrastive_pretrain import run_contrastive_ablation


def _write_inputs(tmp_path: Path):
    csv_path = tmp_path / "subset_000.csv"
    geo_path = tmp_path / "coords_000.pkl"
    rows = []
    geometry = {}
    smiles_list = ["C", "CC", "CCC", "CCO", "CCN", "CCCC", "CCCO", "CCCN"]
    atom_counts = [5, 8, 11, 9, 9, 14, 12, 12]
    for idx, (smiles, n_atoms) in enumerate(zip(smiles_list, atom_counts)):
        rows.append({"smiles": smiles, "logP": 0.1 * idx, "qed": 0.05 * idx, "SAS": 0.2 * idx})
        rng = np.random.default_rng(idx)
        geometry[f"subset_0_idx_{idx}"] = {
            "smiles": smiles,
            "charge": 0,
            "atomic_nums": [6] * n_atoms,
            "conformers": [rng.standard_normal((n_atoms, 3)).astype(np.float32)],
        }
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    pickle.dump(geometry, geo_path.open("wb"))
    return csv_path, geo_path


def test_run_contrastive_ablation_returns_both_arms(tmp_path: Path):
    csv_path, geo_path = _write_inputs(tmp_path)
    dataset = load_minimal_zinc_dataset(csv_path, geo_path, limit=8)

    report = run_contrastive_ablation(
        dataset, hidden_dim=16, epochs=30, batch_size=4, contrastive_weight=1.0, seed=0
    )

    assert set(report) == {"supervised_only", "with_contrastive"}
    for arm in report.values():
        assert isinstance(arm, dict)
        assert arm  # non-empty metrics
