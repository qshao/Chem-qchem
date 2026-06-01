import torch

from qchem_gnn.graph import GraphBatch, build_graph_from_smiles
from qchem_gnn.model import MolecularQuantumGNN


def test_molecular_quantum_gnn_returns_embedding_and_quantum_heads():
    graphs = [
        build_graph_from_smiles("C"),
        build_graph_from_smiles("CC"),
    ]
    batch = GraphBatch.from_graphs(graphs)
    model = MolecularQuantumGNN(
        atom_vocab_size=128,
        bond_vocab_size=8,
        hidden_dim=32,
        num_message_passing_steps=2,
        graph_targets=2,
    )

    node_pred, edge_pred, graph_pred, embedding = model(batch)
    graph_states = model.encode_graph_states(batch)

    assert node_pred.shape == (batch.num_nodes, 2)
    assert edge_pred.shape == (batch.num_edges, 1)
    assert graph_pred.shape == (batch.num_graphs, 2)
    assert torch.isfinite(node_pred).all()
    assert torch.isfinite(edge_pred).all()
    assert torch.isfinite(graph_pred).all()
    assert embedding.shape == (batch.num_graphs, 32)
    assert torch.allclose(embedding, model.encode_graph_embeddings(batch))
    assert not torch.allclose(embedding, graph_states)
