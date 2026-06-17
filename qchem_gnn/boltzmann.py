from __future__ import annotations

import torch

# Boltzmann constant in Hartree per Kelvin.
BOLTZMANN_HARTREE_PER_K: float = 3.166811563e-6


def boltzmann_weights(
    energies: torch.Tensor, temperature: float = 298.15
) -> torch.Tensor:
    """Boltzmann weights over conformer energies (Hartree). Returns float32 [C]."""
    if energies.ndim != 1:
        raise ValueError("energies must be a 1D tensor")
    if energies.numel() == 0:
        raise ValueError("energies must contain at least one conformer")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    e = energies.to(torch.float64)
    kt = BOLTZMANN_HARTREE_PER_K * temperature
    shifted = -(e - e.min()) / kt
    return torch.softmax(shifted, dim=0).to(torch.float32)


def boltzmann_average(
    values: torch.Tensor, energies: torch.Tensor, temperature: float = 298.15
) -> torch.Tensor:
    """Boltzmann-weighted average of ``values`` over their leading conformer axis."""
    weights = boltzmann_weights(energies, temperature)
    if values.shape[0] != weights.shape[0]:
        raise ValueError("values and energies must share the leading conformer dimension")
    weight_shape = [weights.shape[0]] + [1] * (values.ndim - 1)
    return (values.to(torch.float32) * weights.reshape(weight_shape)).sum(dim=0)
