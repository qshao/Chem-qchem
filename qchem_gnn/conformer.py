from __future__ import annotations

from dataclasses import dataclass

import torch

from .graph import GraphBatch, GraphData


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
        # Note: this uses softmax(-energy) without kT scaling, unlike boltzmann_average()
        # in boltzmann.py which applies the physically correct kT denominator.
        # This branch is not used by the contrastive trainer (which uses _boltzmann_pool_molecules).
        weights = torch.softmax(-conformer_energy.to(conformer_embeddings.dtype), dim=0)
        return (conformer_embeddings * weights.unsqueeze(-1)).sum(dim=0)

    raise ValueError(f"Unknown pooling mode: {mode}")


@dataclass(frozen=True)
class ConformerEncoderBatch:
    atomic_numbers: torch.LongTensor
    edge_index: torch.LongTensor
    positions: torch.Tensor
    node_conformer_index: torch.LongTensor
    conformer_molecule_index: torch.LongTensor
    conformer_energy: torch.Tensor | None
    num_conformers: int
    num_molecules: int

    @classmethod
    def from_molecule_conformers(
        cls,
        graphs: list[GraphData],
        conformer_coords: list[list[torch.Tensor]],
        conformer_energies: list[torch.Tensor] | None = None,
    ) -> "ConformerEncoderBatch":
        atomic_numbers: list[torch.Tensor] = []
        edge_indices: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        node_conformer_index: list[torch.Tensor] = []
        conformer_molecule_index: list[int] = []
        energies: list[torch.Tensor] = []

        node_offset = 0
        conformer_id = 0
        for molecule_id, (graph, coords_list) in enumerate(zip(graphs, conformer_coords)):
            for conformer_position, coords in enumerate(coords_list):
                atomic_numbers.append(graph.atomic_numbers)
                edge_indices.append(graph.edge_index + node_offset)
                positions.append(coords.to(torch.float32))
                node_conformer_index.append(
                    torch.full((graph.num_nodes,), conformer_id, dtype=torch.long)
                )
                conformer_molecule_index.append(molecule_id)
                if conformer_energies is not None:
                    energies.append(conformer_energies[molecule_id][conformer_position])
                node_offset += graph.num_nodes
                conformer_id += 1

        energy_tensor = torch.stack(energies, dim=0) if energies else None
        return cls(
            atomic_numbers=torch.cat(atomic_numbers, dim=0),
            edge_index=torch.cat(edge_indices, dim=1),
            positions=torch.cat(positions, dim=0),
            node_conformer_index=torch.cat(node_conformer_index, dim=0),
            conformer_molecule_index=torch.tensor(conformer_molecule_index, dtype=torch.long),
            conformer_energy=energy_tensor,
            num_conformers=conformer_id,
            num_molecules=len(graphs),
        )


def pool_conformers_to_molecules(
    conformer_embeddings: torch.Tensor,
    conformer_molecule_index: torch.Tensor,
    conformer_energy: torch.Tensor | None,
    num_molecules: int,
    mode: str = "mean",
) -> torch.Tensor:
    pooled = []
    for molecule_id in range(num_molecules):
        mask = conformer_molecule_index == molecule_id
        molecule_embeddings = conformer_embeddings[mask]
        molecule_energy = conformer_energy[mask] if conformer_energy is not None else None
        pooled.append(
            pool_conformer_embeddings(molecule_embeddings, conformer_energy=molecule_energy, mode=mode)
        )
    return torch.stack(pooled, dim=0)
