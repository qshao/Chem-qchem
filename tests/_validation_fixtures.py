from __future__ import annotations

from pathlib import Path

import h5py

from qchem_gnn.quantum_data import load_quantum_zinc_dataset
from tests._quantum_fixtures import write_synthetic_results_h5


def make_tiny_quantum_dataset(tmp_path, mols=None):
    """Build a small MinimalQuantumDataset from synthetic real-schema HDF5.

    Each (smiles, n_atoms) pair becomes one molecule group with two conformers.
    ``CO`` -> 6 atoms, ``CCO`` -> 9, ``CCN`` -> 10, ``CCC`` -> 11 (explicit H).
    """
    mols = mols or [("CO", 6), ("CCO", 9), ("CCN", 10), ("CCC", 11)]
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "subset_044.csv"
    csv_path.write_text(
        "smiles,logP,qed,SAS\n"
        + "".join(f"{smiles},0.0,0.5,2.0\n" for smiles, _ in mols)
    )
    h5_path = tmp_path / "results_044.h5"
    with h5py.File(h5_path, "w") as handle:
        for idx, (smiles, n_atoms) in enumerate(mols):
            tmp_single = tmp_path / f"_tmp_{idx}.h5"
            write_synthetic_results_h5(
                tmp_single,
                mol_id=f"subset_44_idx_{idx}",
                smiles=smiles,
                n_atoms=n_atoms,
                seed=idx,
            )
            with h5py.File(tmp_single, "r") as src:
                src.copy(src[f"subset_44_idx_{idx}"], handle, f"subset_44_idx_{idx}")
    with h5py.File(h5_path, "r") as handle:
        return load_quantum_zinc_dataset(
            csv_path,
            geometry_path=tmp_path / "coords_044.pkl",
            limit=len(mols),
            results_handle=handle,
            use_results=True,
        )
