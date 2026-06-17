import pytest
import torch

from qchem_gnn.losses import info_nce_contrastive_loss, vicreg_loss


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


def test_vicreg_identical_views_zero_invariance():
    # With only the invariance term active, identical views give ~0 loss.
    z = torch.randn(8, 4)
    loss = vicreg_loss(z, z.clone(), sim_weight=1.0, var_weight=0.0, cov_weight=0.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_vicreg_penalizes_collapse():
    # A collapsed (constant) embedding incurs a large variance penalty;
    # a high-variance embedding incurs ~none.
    collapsed = torch.ones(8, 4)
    varied = torch.randn(8, 4) * 5.0
    loss_collapsed = vicreg_loss(collapsed, collapsed, sim_weight=0.0, var_weight=1.0, cov_weight=0.0)
    loss_varied = vicreg_loss(varied, varied, sim_weight=0.0, var_weight=1.0, cov_weight=0.0)
    assert float(loss_collapsed) > float(loss_varied)
    assert float(loss_collapsed) > 1.0


def test_vicreg_covariance_penalizes_correlated_dims():
    # Two perfectly correlated dims -> non-zero covariance term;
    # two orthogonal dims -> ~zero covariance term.
    correlated = torch.tensor([[1.0, 2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, -2.0]])
    decorrelated = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    loss_corr = vicreg_loss(correlated, correlated, sim_weight=0.0, var_weight=0.0, cov_weight=1.0)
    loss_dec = vicreg_loss(decorrelated, decorrelated, sim_weight=0.0, var_weight=0.0, cov_weight=1.0)
    assert float(loss_corr) > float(loss_dec)
    assert float(loss_dec) == pytest.approx(0.0, abs=1e-6)


def test_vicreg_requires_two_examples():
    with pytest.raises(ValueError):
        vicreg_loss(torch.randn(1, 4), torch.randn(1, 4))


def test_vicreg_shape_mismatch_raises():
    with pytest.raises(ValueError):
        vicreg_loss(torch.randn(8, 4), torch.randn(8, 5))
