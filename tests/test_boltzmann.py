import math

import pytest
import torch

from qchem_gnn.boltzmann import (
    BOLTZMANN_HARTREE_PER_K,
    boltzmann_average,
    boltzmann_weights,
)


def test_weights_sum_to_one_and_favor_lower_energy():
    energies = torch.tensor([-115.0005, -115.0000])
    w = boltzmann_weights(energies)
    assert w.dtype == torch.float32
    assert math.isclose(float(w.sum()), 1.0, rel_tol=1e-6)
    assert w[0] > w[1]  # lower energy gets more weight


def test_single_conformer_weight_is_one():
    w = boltzmann_weights(torch.tensor([-42.0]))
    assert w.shape == (1,)
    assert math.isclose(float(w[0]), 1.0, rel_tol=1e-6)


def test_degenerate_energies_are_uniform():
    w = boltzmann_weights(torch.tensor([-5.0, -5.0, -5.0]))
    assert torch.allclose(w, torch.full((3,), 1.0 / 3.0), atol=1e-6)


def test_weights_match_manual_softmax():
    energies = torch.tensor([-10.0, -10.001, -9.999])
    kt = BOLTZMANN_HARTREE_PER_K * 298.15
    manual = torch.softmax(-(energies - energies.min()) / kt, dim=0)
    assert torch.allclose(boltzmann_weights(energies), manual.to(torch.float32), atol=1e-6)


def test_average_weights_leading_axis():
    energies = torch.tensor([-1.0, -1.0])  # uniform -> plain mean
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = boltzmann_average(values, energies)
    assert torch.allclose(out, torch.tensor([2.0, 3.0]), atol=1e-6)


def test_average_handles_3d_values():
    energies = torch.tensor([-1.0, -1.0])
    values = torch.ones(2, 4, 1)
    out = boltzmann_average(values, energies)
    assert out.shape == (4, 1)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        boltzmann_weights(torch.empty(0))
    with pytest.raises(ValueError):
        boltzmann_weights(torch.tensor([-1.0]), temperature=0.0)
    with pytest.raises(ValueError):
        boltzmann_average(torch.ones(3, 2), torch.tensor([-1.0, -1.0]))
