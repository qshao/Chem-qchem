import torch

from qchem_gnn.encoder3d import GaussianRBF


def test_gaussian_rbf_shapes_and_range():
    rbf = GaussianRBF(num_rbf=8, cutoff=5.0)
    distances = torch.tensor([0.0, 1.0, 2.5, 5.0], dtype=torch.float32)
    expanded = rbf(distances)

    assert expanded.shape == (4, 8)
    assert torch.isfinite(expanded).all()
    # Gaussian activations are bounded by 1.0 (peak when distance equals a center).
    assert (expanded.max(dim=-1).values <= 1.0 + 1e-6).all()
