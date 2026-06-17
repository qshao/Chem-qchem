import torch

from qchem_gnn.losses import info_nce_contrastive_loss


def test_info_nce_returns_scalar_and_flows_gradient():
    z_a = torch.randn(8, 16, requires_grad=True)
    z_b = torch.randn(8, 16)

    loss = info_nce_contrastive_loss(z_a, z_b, temperature=0.1)

    assert loss.ndim == 0
    loss.backward()
    assert z_a.grad is not None
    assert torch.isfinite(z_a.grad).all()


def test_info_nce_is_symmetric():
    torch.manual_seed(0)
    z_a = torch.randn(6, 16)
    z_b = torch.randn(6, 16)

    forward = info_nce_contrastive_loss(z_a, z_b, temperature=0.2)
    swapped = info_nce_contrastive_loss(z_b, z_a, temperature=0.2)

    assert torch.allclose(forward, swapped, atol=1e-6)


def test_info_nce_rewards_aligned_pairs():
    base = torch.randn(8, 16)
    aligned = info_nce_contrastive_loss(base, base.clone(), temperature=0.1)
    shuffled = info_nce_contrastive_loss(base, base[torch.randperm(8)], temperature=0.1)

    assert aligned < shuffled
