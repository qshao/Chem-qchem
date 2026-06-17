# qchem_gnn/teacher_heads.py
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class QuantumTeacherHeads(nn.Module):
    """Per-conformer quantum prediction heads on top of the 3D encoder."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.node_head = nn.Linear(hidden_dim, 1)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        conformer_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_pred = self.node_head(node_states)
        src, dst = edge_index
        edge_pred = self.edge_head(
            torch.cat([node_states[src], node_states[dst]], dim=-1)
        )
        graph_pred = self.graph_head(conformer_embedding)
        return node_pred, edge_pred, graph_pred


def teacher_loss(
    node_pred: torch.Tensor,
    edge_pred: torch.Tensor,
    graph_pred: torch.Tensor,
    node_target: torch.Tensor,
    edge_target: torch.Tensor,
    graph_target: torch.Tensor,
) -> torch.Tensor:
    return (
        F.mse_loss(node_pred, node_target)
        + F.mse_loss(edge_pred, edge_target)
        + F.mse_loss(graph_pred, graph_target)
    )


def assemble_conformer_targets(
    examples: list,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate per-conformer targets in molecule-then-conformer order.

    Matches the node/edge/conformer ordering produced by
    ``ConformerEncoderBatch.from_molecule_conformers``.
    """
    node_targets: list[torch.Tensor] = []
    edge_targets: list[torch.Tensor] = []
    graph_targets: list[torch.Tensor] = []
    energies: list[torch.Tensor] = []

    for example in examples:
        node = example.conformer_node_targets  # [C, N, 1]
        edge = example.conformer_edge_targets  # [C, E, 1]
        node_targets.append(node.reshape(-1, node.shape[-1]))
        edge_targets.append(edge.reshape(-1, edge.shape[-1]))
        graph_targets.append(example.conformer_graph_targets)
        energies.append(example.conformer_energies)

    return (
        torch.cat(node_targets, dim=0),
        torch.cat(edge_targets, dim=0),
        torch.cat(graph_targets, dim=0),
        torch.cat(energies, dim=0),
    )
