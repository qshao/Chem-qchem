from __future__ import annotations

from dataclasses import dataclass

import torch

from .graph import GraphBatch


@dataclass(frozen=True)
class ConformerBatch:
    graph_batch: GraphBatch
    conformer_index: torch.LongTensor
    conformer_energy: torch.Tensor

    def __post_init__(self) -> None:
        if self.conformer_index.ndim != 1:
            raise ValueError("conformer_index must be a 1D tensor")
        if self.conformer_energy.ndim != 1:
            raise ValueError("conformer_energy must be a 1D tensor")
        if self.conformer_index.shape != self.conformer_energy.shape:
            raise ValueError("conformer_index and conformer_energy must have matching shape")
        if self.conformer_index.numel() == 0:
            return
        if self.conformer_index.min().item() < 0:
            raise ValueError("conformer_index cannot contain negative values")
        if self.conformer_index.max().item() >= self.graph_batch.num_graphs:
            raise ValueError("conformer_index contains graph ids outside graph_batch")

    def conformer_counts_per_graph(self) -> torch.LongTensor:
        return torch.bincount(self.conformer_index, minlength=self.graph_batch.num_graphs)


def pool_conformer_embeddings(
    conformer_embeddings: torch.Tensor,
    conformer_energy: torch.Tensor | None = None,
    mode: str = "mean",
) -> torch.Tensor:
    if conformer_embeddings.ndim == 1:
        return conformer_embeddings
    if conformer_embeddings.shape[0] == 0:
        raise ValueError("conformer_embeddings must contain at least one conformer")

    if mode == "mean":
        return conformer_embeddings.mean(dim=0)
    if mode in {"energy", "weighted"}:
        if conformer_energy is None:
            raise ValueError("conformer_energy is required for energy-weighted pooling")
        weights = torch.softmax(-conformer_energy.to(conformer_embeddings.dtype), dim=0)
        return (conformer_embeddings * weights.unsqueeze(-1)).sum(dim=0)

    raise ValueError(f"Unknown pooling mode: {mode}")
