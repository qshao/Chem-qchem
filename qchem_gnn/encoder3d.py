from __future__ import annotations

import torch
from torch import nn


class GaussianRBF(nn.Module):
    def __init__(self, num_rbf: int = 16, cutoff: float = 5.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        width = cutoff / max(num_rbf - 1, 1)
        self.coeff = -0.5 / (width ** 2)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(self.coeff * diff.pow(2))
