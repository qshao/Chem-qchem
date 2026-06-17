import torch

from qchem_gnn.conformer import ConformerEncoderBatch, pool_conformers_to_molecules
from qchem_gnn.graph import build_graph_from_smiles


def test_from_molecule_conformers_packs_offsets():
    graphs = [build_graph_from_smiles("C"), build_graph_from_smiles("CC")]
    coords = [
        [torch.zeros(graphs[0].num_nodes, 3), torch.ones(graphs[0].num_nodes, 3)],
        [torch.zeros(graphs[1].num_nodes, 3)],
    ]

    batch = ConformerEncoderBatch.from_molecule_conformers(graphs, coords)

    assert batch.num_molecules == 2
    assert batch.num_conformers == 3
    assert batch.conformer_molecule_index.tolist() == [0, 0, 1]
    assert batch.positions.shape[0] == batch.atomic_numbers.shape[0]
    # node_conformer_index has one entry per packed atom, spanning 3 conformers.
    assert int(batch.node_conformer_index.max().item()) == 2
    # Second conformer's edges are offset past the first conformer's atoms.
    first_conf_atoms = graphs[0].num_nodes
    assert int(batch.edge_index[:, graphs[0].num_edges:].min().item()) >= first_conf_atoms


def test_pool_conformers_to_molecules_mean():
    conformer_embeddings = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]], dtype=torch.float32)
    conformer_molecule_index = torch.tensor([0, 0, 1], dtype=torch.long)

    pooled = pool_conformers_to_molecules(
        conformer_embeddings,
        conformer_molecule_index,
        conformer_energy=None,
        num_molecules=2,
        mode="mean",
    )

    assert pooled.shape == (2, 2)
    assert torch.allclose(pooled[0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(pooled[1], torch.tensor([5.0, 5.0]))
