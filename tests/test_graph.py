from qchem_gnn.graph import build_graph_from_smiles


def test_build_graph_from_smiles_adds_explicit_hydrogens():
    graph = build_graph_from_smiles("C")

    assert graph.num_nodes == 5
    assert graph.atomic_numbers.tolist() == [6, 1, 1, 1, 1]
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] == 8
    assert graph.edge_attr.shape[0] == 8
