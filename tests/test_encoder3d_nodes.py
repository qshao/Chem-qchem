import torch

from qchem_gnn.encoder3d import Conformer3DEncoder


def _toy_inputs():
    atomic_numbers = torch.tensor([6, 8, 6, 8])  # 2 conformers, 2 atoms each
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    positions = torch.tensor(
        [[0.0, 0, 0], [1.0, 0, 0], [0.0, 0, 0], [1.2, 0, 0]], dtype=torch.float32
    )
    node_conformer_index = torch.tensor([0, 0, 1, 1])
    return atomic_numbers, edge_index, positions, node_conformer_index, 2


def test_forward_with_nodes_shapes_and_consistency():
    enc = Conformer3DEncoder(atom_vocab_size=128, hidden_dim=16, num_message_passing_steps=2)
    enc.eval()
    args = _toy_inputs()
    with torch.no_grad():
        node_states, conf_emb = enc.forward_with_nodes(*args)
        legacy = enc(*args)
    assert node_states.shape == (4, 16)
    assert conf_emb.shape == (2, 16)
    # the pooled embedding from forward_with_nodes matches the legacy forward()
    assert torch.allclose(conf_emb, legacy, atol=1e-6)
