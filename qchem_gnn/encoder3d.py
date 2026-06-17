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


class Residual3DMessagePassingBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rbf: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        messages = self.message_mlp(torch.cat([node_states[src], edge_rbf], dim=-1))

        aggregated = torch.zeros_like(node_states)
        aggregated.index_add_(0, dst, messages)

        updated = self.update_mlp(torch.cat([node_states, aggregated], dim=-1))
        return self.norm(node_states + updated)


class Conformer3DEncoder(nn.Module):
    def __init__(
        self,
        atom_vocab_size: int,
        hidden_dim: int = 64,
        num_rbf: int = 16,
        cutoff: float = 5.0,
        num_message_passing_steps: int = 3,
    ):
        super().__init__()
        self.atom_encoder = nn.Embedding(atom_vocab_size, hidden_dim)
        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=cutoff)
        self.blocks = nn.ModuleList(
            Residual3DMessagePassingBlock(hidden_dim, num_rbf)
            for _ in range(num_message_passing_steps)
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        positions: torch.Tensor,
        node_conformer_index: torch.Tensor,
        num_conformers: int,
    ) -> torch.Tensor:
        node_states = self.atom_encoder(atomic_numbers)
        src, dst = edge_index
        distances = (positions[src] - positions[dst]).norm(dim=-1)
        edge_rbf = self.rbf(distances)

        for block in self.blocks:
            node_states = block(node_states, edge_index, edge_rbf)

        pooled = torch.zeros(
            num_conformers,
            node_states.shape[-1],
            dtype=node_states.dtype,
            device=node_states.device,
        )
        pooled.index_add_(0, node_conformer_index, node_states)
        counts = torch.bincount(node_conformer_index, minlength=num_conformers).clamp_min(1).unsqueeze(-1).to(pooled.device)
        return self.embedding_head(pooled / counts)
