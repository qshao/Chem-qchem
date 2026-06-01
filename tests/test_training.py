import torch

from qchem_gnn.graph import GraphBatch, build_graph_from_smiles
from qchem_gnn.losses import compute_multitask_loss as compute_weighted_multitask_loss
from qchem_gnn.model import MolecularQuantumGNN
from qchem_gnn.training import compute_multitask_loss


def test_multitask_loss_wrapper_matches_weighted_helper():
    predictions = (
        torch.zeros(4, 2),
        torch.zeros(6, 1),
        torch.zeros(2, 2),
        torch.zeros(2, 32),
    )
    targets = (
        torch.ones(4, 2),
        torch.ones(6, 1),
        torch.ones(2, 2),
    )

    wrapped_loss = compute_multitask_loss(predictions, targets)
    direct_loss = compute_weighted_multitask_loss(predictions, targets, weights={
        "atom": 1.0,
        "edge": 1.0,
        "graph": 0.5,
    })

    assert torch.allclose(wrapped_loss, direct_loss)


def test_multitask_loss_supports_weighted_quantum_and_auxiliary_targets():
    predictions = (
        torch.zeros(4, 2),
        torch.zeros(6, 1),
        torch.zeros(2, 2),
        torch.zeros(2, 32),
    )
    targets = {
        "atom": torch.ones(4, 2),
        "edge": torch.ones(6, 1),
        "graph": torch.ones(2, 2),
        "aux": torch.ones(2, 32),
        "consistency": torch.ones(2, 32),
    }

    loss = compute_weighted_multitask_loss(
        predictions,
        targets,
        weights={
            "atom": 1.0,
            "edge": 1.0,
            "graph": 0.5,
            "aux": 0.1,
            "consistency": 0.1,
        },
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_molecular_quantum_gnn_can_overfit_a_tiny_batch():
    torch.manual_seed(0)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    node_target = torch.stack(
        [batch.atomic_numbers.float() / 10.0, batch.atomic_numbers.float() / 20.0],
        dim=-1,
    )
    edge_target = batch.edge_attr.float() / 4.0
    graph_target = torch.tensor(
        [[graph.num_nodes / 10.0, graph.num_edges / 10.0] for graph in graphs],
        dtype=torch.float32,
    )

    with torch.no_grad():
        initial_loss = compute_multitask_loss(
            model(batch),
            (node_target, edge_target, graph_target),
        ).item()

    for _ in range(300):
        optimizer.zero_grad()
        loss = compute_multitask_loss(
            model(batch),
            (node_target, edge_target, graph_target),
        )
        loss.backward()
        optimizer.step()

    final_loss = compute_multitask_loss(
        model(batch),
        (node_target, edge_target, graph_target),
    ).item()

    assert final_loss < initial_loss * 0.1
