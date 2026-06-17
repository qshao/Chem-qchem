# tests/test_quantum_dataset_loading.py
from pathlib import Path

import torch

from qchem_gnn.boltzmann import boltzmann_average
from qchem_gnn.graph import build_graph_from_smiles
from qchem_gnn.quantum_data import extract_per_conformer_targets, load_quantum_zinc_dataset
from tests._quantum_fixtures import write_synthetic_results_h5
import h5py


def _write_csv(path: Path, smiles: str) -> Path:
    path.write_text("smiles,logP,qed,SAS\n" + f"{smiles},0.0,0.5,2.0\n")
    return path


def test_use_results_populates_per_conformer_fields(tmp_path):
    csv_path = _write_csv(tmp_path / "subset_044.csv", "CO")
    h5_path = write_synthetic_results_h5(tmp_path / "results_044.h5")

    with h5py.File(h5_path, "r") as handle:
        dataset = load_quantum_zinc_dataset(
            csv_path,
            geometry_path=tmp_path / "coords_044.pkl",  # absent -> no pickle coords
            limit=1,
            results_handle=handle,
            use_results=True,
        )

    assert len(dataset.examples) == 1
    ex = dataset.examples[0]
    assert ex.conformer_node_targets is not None
    assert ex.conformer_node_targets.shape[0] == 2  # two conformers
    assert ex.conformer_graph_targets.shape == (2, 2)
    assert ex.conformer_energies.shape == (2,)
    # molecule-level target equals the Boltzmann average of the per-conformer targets
    expected = boltzmann_average(ex.conformer_node_targets, ex.conformer_energies)
    assert torch.allclose(ex.node_target, expected, atol=1e-5)
    # coords used for the 3D path come from the HDF5 (one per conformer)
    assert ex.conformer_coords is not None and len(ex.conformer_coords) == 2


def test_proxy_path_leaves_per_conformer_fields_none(tmp_path):
    csv_path = _write_csv(tmp_path / "subset_044.csv", "CO")
    dataset = load_quantum_zinc_dataset(
        csv_path,
        geometry_path=tmp_path / "coords_044.pkl",
        limit=1,
        use_results=False,
    )
    ex = dataset.examples[0]
    assert ex.conformer_node_targets is None
    assert ex.conformer_graph_targets is None


def test_molecules_without_dft_group_are_skipped(tmp_path):
    # CSV has two rows, HDF5 only has subset_44_idx_0 -> second row skipped
    (tmp_path / "subset_044.csv").write_text(
        "smiles,logP,qed,SAS\nCO,0.0,0.5,2.0\nCCO,0.1,0.6,2.1\n"
    )
    h5_path = write_synthetic_results_h5(tmp_path / "results_044.h5")
    with h5py.File(h5_path, "r") as handle:
        dataset = load_quantum_zinc_dataset(
            tmp_path / "subset_044.csv",
            geometry_path=tmp_path / "coords_044.pkl",
            limit=2,
            results_handle=handle,
            use_results=True,
        )
    assert [ex.mol_id for ex in dataset.examples] == ["subset_44_idx_0"]
    assert "subset_44_idx_1" in dataset.skipped_mol_ids
