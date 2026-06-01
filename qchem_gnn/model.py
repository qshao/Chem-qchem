from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from .graph import GraphBatch


class EncoderOutput(NamedTuple):
    node_states: torch.Tensor
    edge_states: torch.Tensor
    graph_states: torch.Tensor
    mol_embedding: torch.Tensor


class ModelOutput(NamedTuple):
    node_pred: torch.Tensor
    edge_pred: torch.Tensor
    graph_pred: torch.Tensor
    mol_embedding: torch.Tensor


class ResidualMessagePassingBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
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
        edge_states: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        messages = self.message_mlp(torch.cat([node_states[src], edge_states], dim=-1))

        aggregated = torch.zeros_like(node_states)
        aggregated.index_add_(0, dst, messages)

        updated = self.update_mlp(torch.cat([node_states, aggregated], dim=-1))
        return self.norm(node_states + updated)


class MolecularGraphEncoder(nn.Module):
    def __init__(
        self,
        atom_vocab_size: int,
        bond_vocab_size: int,
        hidden_dim: int,
        num_message_passing_steps: int,
    ):
        super().__init__()
        self.atom_encoder = nn.Embedding(atom_vocab_size, hidden_dim)
        self.bond_encoder = nn.Embedding(bond_vocab_size, hidden_dim)
        self.message_passing_blocks = nn.ModuleList(
            ResidualMessagePassingBlock(hidden_dim)
            for _ in range(num_message_passing_steps)
        )
        self.molecular_embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch: GraphBatch) -> EncoderOutput:
        node_states = self.atom_encoder(batch.atomic_numbers)
        bond_indices = batch.edge_attr.squeeze(-1)
        edge_states = self.bond_encoder(bond_indices)

        for block in self.message_passing_blocks:
            node_states = block(node_states, batch.edge_index, edge_states)

        graph_states = self._mean_pool(node_states, batch.batch, batch.num_graphs)
        mol_embedding = self.molecular_embedding_head(graph_states)
        return EncoderOutput(node_states, edge_states, graph_states, mol_embedding)

    @staticmethod
    def _mean_pool(
        node_states: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        pooled = torch.zeros(
            num_graphs,
            node_states.shape[-1],
            dtype=node_states.dtype,
            device=node_states.device,
        )
        pooled.index_add_(0, batch_index, node_states)

        counts = torch.bincount(batch_index, minlength=num_graphs).clamp_min(1).unsqueeze(-1)
        return pooled / counts


class MolecularQuantumGNN(nn.Module):
    def __init__(
        self,
        atom_vocab_size: int,
        bond_vocab_size: int,
        hidden_dim: int = 64,
        num_message_passing_steps: int = 3,
        node_targets: int = 2,
        edge_targets: int = 1,
        graph_targets: int = 2,
    ):
        super().__init__()
        self.encoder = MolecularGraphEncoder(
            atom_vocab_size=atom_vocab_size,
            bond_vocab_size=bond_vocab_size,
            hidden_dim=hidden_dim,
            num_message_passing_steps=num_message_passing_steps,
        )
        self.atom_head = nn.Linear(hidden_dim, node_targets)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_targets),
        )
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, graph_targets),
        )

    def _encode(self, batch: GraphBatch) -> EncoderOutput:
        return self.encoder(batch)

    def forward(self, batch: GraphBatch) -> ModelOutput:
        encoded = self._encode(batch)
        node_pred = self.atom_head(encoded.node_states)

        src, dst = batch.edge_index
        edge_pred = self.edge_head(
            torch.cat([encoded.node_states[src], encoded.node_states[dst], encoded.edge_states], dim=-1)
        )

        graph_pred = self.graph_head(encoded.graph_states)
        return ModelOutput(node_pred, edge_pred, graph_pred, encoded.mol_embedding)

    def encode_graph_embeddings(self, batch: GraphBatch) -> torch.Tensor:
        return self._encode(batch).mol_embedding

    def encode_graph_states(self, batch: GraphBatch) -> torch.Tensor:
        return self._encode(batch).graph_states
