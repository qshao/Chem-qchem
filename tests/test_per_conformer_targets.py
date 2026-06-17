from pathlib import Path

import h5py
import pytest
import torch

from qchem_gnn.graph import build_graph_from_smiles
from qchem_gnn.quantum_data import (
    PerConformerTargets,
    _isotropic_polarizability,
    extract_per_conformer_targets,
)
from tests._quantum_fixtures import write_synthetic_results_h5

REAL_SHARD = Path("zinc-250k/results/results_044.h5")


def test_isotropic_polarizability_is_trace_over_three():
    alpha = [[3.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 9.0]]
    assert _isotropic_polarizability(alpha) == pytest.approx(6.0)


def test_isotropic_polarizability_scalar_passthrough():
    assert _isotropic_polarizability(5.0) == pytest.approx(5.0)


def test_extract_shapes_from_synthetic(tmp_path):
    h5_path = write_synthetic_results_h5(tmp_path / "synthetic.h5")
    graph = build_graph_from_smiles("CO")
    with h5py.File(h5_path, "r") as handle:
        pct = extract_per_conformer_targets(graph, handle["subset_44_idx_0"])

    assert isinstance(pct, PerConformerTargets)
    n, e = graph.num_nodes, graph.num_edges
    assert pct.node_targets.shape == (2, n, 1)
    assert pct.edge_targets.shape == (2, e, 1)
    assert pct.graph_targets.shape == (2, 2)
    assert pct.energies.shape == (2,)
    assert len(pct.coords) == 2 and pct.coords[0].shape == (n, 3)
    # graph target column 1 is isotropic polarizability = trace/3 of (10/11 * I)
    assert pct.graph_targets[0, 1].item() == pytest.approx(10.0, rel=1e-5)
    assert pct.graph_targets[1, 1].item() == pytest.approx(11.0, rel=1e-5)


def test_extract_rejects_atom_count_mismatch(tmp_path):
    # n_atoms deliberately wrong for "CO" (6) -> mismatch must raise
    h5_path = write_synthetic_results_h5(tmp_path / "bad.h5", n_atoms=5)
    graph = build_graph_from_smiles("CO")
    with h5py.File(h5_path, "r") as handle:
        with pytest.raises(ValueError):
            extract_per_conformer_targets(graph, handle["subset_44_idx_0"])


@pytest.mark.skipif(not REAL_SHARD.exists(), reason="real DFT shard not present")
def test_extract_from_real_shard():
    with h5py.File(REAL_SHARD, "r") as handle:
        mol_id = next(iter(handle.keys()))
        smiles = str(handle[mol_id].attrs["smiles"])
        graph = build_graph_from_smiles(smiles)
        pct = extract_per_conformer_targets(graph, handle[mol_id])
    assert pct.node_targets.shape[1] == graph.num_nodes
    assert pct.node_targets.shape[2] == 1
    assert pct.graph_targets.shape[1] == 2
    assert pct.energies.numel() == pct.node_targets.shape[0]
