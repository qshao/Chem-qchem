import torch

from qchem_gnn.encoder3d import Conformer3DEncoder, GaussianRBF


def test_gaussian_rbf_shapes_and_range():
    rbf = GaussianRBF(num_rbf=8, cutoff=5.0)
    distances = torch.tensor([0.0, 1.0, 2.5, 5.0], dtype=torch.float32)
    expanded = rbf(distances)

    assert expanded.shape == (4, 8)
    assert torch.isfinite(expanded).all()
    # Gaussian activations are bounded by 1.0 (peak when distance equals a center).
    assert (expanded.max(dim=-1).values <= 1.0 + 1e-6).all()


def _two_atom_conformer():
    atomic_numbers = torch.tensor([6, 8], dtype=torch.long)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float32)
    node_conformer_index = torch.tensor([0, 0], dtype=torch.long)
    return atomic_numbers, edge_index, positions, node_conformer_index


def test_conformer_encoder_output_shape():
    encoder = Conformer3DEncoder(atom_vocab_size=128, hidden_dim=16, num_rbf=8, num_message_passing_steps=2)
    atomic_numbers, edge_index, positions, node_conformer_index = _two_atom_conformer()

    embedding = encoder(atomic_numbers, edge_index, positions, node_conformer_index, num_conformers=1)

    assert embedding.shape == (1, 16)
    assert torch.isfinite(embedding).all()


def test_conformer_encoder_is_rotation_and_translation_invariant():
    encoder = Conformer3DEncoder(atom_vocab_size=128, hidden_dim=16, num_rbf=8, num_message_passing_steps=2)
    encoder.eval()
    atomic_numbers, edge_index, positions, node_conformer_index = _two_atom_conformer()

    theta = torch.tensor(0.7)
    rotation = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta), torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = torch.tensor([3.0, -2.0, 1.0])
    moved_positions = positions @ rotation.t() + translation

    with torch.no_grad():
        base = encoder(atomic_numbers, edge_index, positions, node_conformer_index, num_conformers=1)
        moved = encoder(atomic_numbers, edge_index, moved_positions, node_conformer_index, num_conformers=1)

    assert torch.allclose(base, moved, atol=1e-5)


def test_conformer_encoder_pools_multiple_conformers():
    encoder = Conformer3DEncoder(atom_vocab_size=128, hidden_dim=16, num_rbf=8, num_message_passing_steps=2)
    # Two conformers of the same 2-atom molecule, packed back to back.
    atomic_numbers = torch.tensor([6, 8, 6, 8], dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        dtype=torch.float32,
    )
    node_conformer_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    embeddings = encoder(atomic_numbers, edge_index, positions, node_conformer_index, num_conformers=2)

    assert embeddings.shape == (2, 16)
