from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .conformer import ConformerEncoderBatch, pool_conformers_to_molecules
from .encoder3d import Conformer3DEncoder
from .graph import GraphBatch
from .losses import compute_multitask_loss, info_nce_contrastive_loss
from .minimal import MinimalQuantumDataset
from .model import MolecularQuantumGNN
from .quantum_data import compute_target_normalization, normalize_targets


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class ContrastivePretrainingResult:
    model: MolecularQuantumGNN
    loss_history: list[float]
    contrastive_loss_history: list[float]
    embeddings: torch.Tensor
    target_normalization: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, object]
    epoch: int
    global_step: int


def _supervised_loss_for_batch(model_output, examples, normalization) -> torch.Tensor:
    node_target = torch.cat([example.node_target for example in examples], dim=0)
    edge_target = torch.cat([example.edge_target for example in examples], dim=0)
    graph_target = torch.stack([example.graph_target for example in examples], dim=0)
    node_target, edge_target, graph_target = normalize_targets(
        node_target, edge_target, graph_target, normalization
    )
    return compute_multitask_loss(model_output, (node_target, edge_target, graph_target))


def contrastive_pretrain_on_dataset(
    dataset: MinimalQuantumDataset,
    *,
    hidden_dim: int = 32,
    num_message_passing_steps: int = 2,
    hidden_dim_3d: int = 32,
    num_rbf: int = 16,
    cutoff: float = 5.0,
    num_message_passing_steps_3d: int = 2,
    epochs: int = 200,
    batch_size: int = 8,
    learning_rate: float = 0.01,
    supervised_weight: float = 1.0,
    contrastive_weight: float = 1.0,
    temperature: float = 0.1,
    conformer_pool_mode: str = "mean",
    seed: int = 0,
) -> ContrastivePretrainingResult:
    torch.manual_seed(seed)
    examples = dataset.examples
    normalization = compute_target_normalization(dataset)

    model = MolecularQuantumGNN(
        atom_vocab_size=128,
        bond_vocab_size=8,
        hidden_dim=hidden_dim,
        num_message_passing_steps=num_message_passing_steps,
        graph_targets=2,
    )
    encoder3d = Conformer3DEncoder(
        atom_vocab_size=128,
        hidden_dim=hidden_dim_3d,
        num_rbf=num_rbf,
        cutoff=cutoff,
        num_message_passing_steps=num_message_passing_steps_3d,
    )
    proj_2d = ProjectionHead(hidden_dim, hidden_dim)
    proj_3d = ProjectionHead(hidden_dim_3d, hidden_dim)

    params = list(model.parameters()) + list(encoder3d.parameters())
    params += list(proj_2d.parameters()) + list(proj_3d.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    loss_history: list[float] = []
    contrastive_loss_history: list[float] = []
    num_examples = len(examples)

    for _ in range(epochs):
        order = torch.randperm(num_examples)
        epoch_total = 0.0
        epoch_contrastive = 0.0
        num_batches = 0

        for start in range(0, num_examples, batch_size):
            batch_indices = order[start : start + batch_size].tolist()
            batch_examples = [examples[i] for i in batch_indices]

            graph_batch = GraphBatch.from_graphs([ex.graph for ex in batch_examples])
            model_output = model(graph_batch)
            supervised = _supervised_loss_for_batch(model_output, batch_examples, normalization)

            contrastive = torch.zeros((), dtype=supervised.dtype)
            with_coords = [ex for ex in batch_examples if ex.conformer_coords]
            if contrastive_weight and len(with_coords) >= 2:
                coords_index = [
                    pos for pos, ex in enumerate(batch_examples) if ex.conformer_coords
                ]
                conformer_batch = ConformerEncoderBatch.from_molecule_conformers(
                    [ex.graph for ex in with_coords],
                    [ex.conformer_coords for ex in with_coords],
                    conformer_energies=None,
                )
                conformer_embeddings = encoder3d(
                    conformer_batch.atomic_numbers,
                    conformer_batch.edge_index,
                    conformer_batch.positions,
                    conformer_batch.node_conformer_index,
                    conformer_batch.num_conformers,
                )
                molecule_3d = pool_conformers_to_molecules(
                    conformer_embeddings,
                    conformer_batch.conformer_molecule_index,
                    conformer_batch.conformer_energy,
                    conformer_batch.num_molecules,
                    mode=conformer_pool_mode,
                )
                molecule_2d = model_output.mol_embedding[coords_index]
                contrastive = info_nce_contrastive_loss(
                    proj_2d(molecule_2d), proj_3d(molecule_3d), temperature=temperature
                )

            total = supervised_weight * supervised + contrastive_weight * contrastive
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            epoch_total += float(total.item())
            epoch_contrastive += float(contrastive.item())
            num_batches += 1

        loss_history.append(epoch_total / max(num_batches, 1))
        contrastive_loss_history.append(epoch_contrastive / max(num_batches, 1))

    with torch.no_grad():
        full_batch = GraphBatch.from_graphs([ex.graph for ex in examples])
        embeddings = model.encode_graph_embeddings(full_batch)

    return ContrastivePretrainingResult(
        model=model,
        loss_history=loss_history,
        contrastive_loss_history=contrastive_loss_history,
        embeddings=embeddings,
        target_normalization=normalization,
        optimizer_state_dict=optimizer.state_dict(),
        epoch=epochs,
        global_step=epochs * ((num_examples + batch_size - 1) // batch_size),
    )
