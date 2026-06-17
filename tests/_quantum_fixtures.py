from __future__ import annotations

from pathlib import Path

import numpy as np


def write_synthetic_results_h5(
    path,
    *,
    mol_id: str = "subset_44_idx_0",
    smiles: str = "CO",
    n_atoms: int = 6,
    n_basis: int = 10,
    energies=(-115.0, -115.0005),
    seed: int = 0,
) -> Path:
    """Write one molecule group mirroring the real results_044.h5 schema.

    ``CO`` (methanol) parses to 2 heavy atoms + 4 H = 6 atoms, matching n_atoms.
    """
    import h5py

    rng = np.random.default_rng(seed)
    path = Path(path)
    with h5py.File(path, "w") as handle:
        group = handle.create_group(mol_id)
        group.attrs["smiles"] = smiles
        for conf_idx, energy in enumerate(energies):
            conf = group.create_group(f"conf_{conf_idx}")
            conf.attrs["energy"] = float(energy)
            conf.attrs["polarizability"] = (np.eye(3) * (10.0 + conf_idx)).astype(np.float64)
            conf.create_dataset("chelpg", data=rng.standard_normal(n_atoms))
            wbi = rng.standard_normal((n_atoms, n_atoms))
            conf.create_dataset("wbi", data=(wbi + wbi.T) / 2.0)
            conf.create_dataset("coords", data=rng.standard_normal((n_atoms, 3)))
            conf.create_dataset("iao_pops", data=rng.standard_normal(n_basis + 3))
            conf.create_dataset("fukui_p", data=rng.standard_normal(n_basis))
            conf.create_dataset("dm", data=rng.standard_normal((n_basis, n_basis)))
    return path
