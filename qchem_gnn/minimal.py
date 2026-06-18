from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .graph import GraphBatch, GraphData
from .model import MolecularQuantumGNN
from .quantum_data import compute_target_normalization, load_quantum_zinc_dataset, normalize_targets
from .training import compute_multitask_loss


@dataclass(frozen=True)
class MinimalQuantumExample:
    mol_id: str
    smiles: str
    graph: GraphData
    charge: int
    conformer_count: int
    node_target: torch.Tensor
    edge_target: torch.Tensor
    graph_target: torch.Tensor
    aux_target: torch.Tensor | None = None
    conformer_coords: list[torch.Tensor] | None = None
    conformer_energies: torch.Tensor | None = None
    conformer_node_targets: torch.Tensor | None = None
    conformer_edge_targets: torch.Tensor | None = None
    conformer_graph_targets: torch.Tensor | None = None
    scaffold_key: int | None = None


@dataclass(frozen=True)
class MinimalQuantumDataset:
    examples: list[MinimalQuantumExample]
    skipped_mol_ids: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> MinimalQuantumExample:
        return self.examples[index]


@dataclass(frozen=True)
class MinimalTrainingResult:
    model: MolecularQuantumGNN
    loss_history: list[float]
    embeddings: torch.Tensor
    target_normalization: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, object]
    scheduler_state_dict: dict[str, object]
    epoch: int
    global_step: int


def load_minimal_zinc_dataset(
    csv_path: str | Path,
    geometry_path: str | Path | None = None,
    limit: int = 16,
) -> MinimalQuantumDataset:
    return load_quantum_zinc_dataset(
        csv_path,
        geometry_path=geometry_path,
        limit=limit,
        use_results=False,
    )


def train_on_minimal_dataset(
    dataset: MinimalQuantumDataset,
    hidden_dim: int = 32,
    num_message_passing_steps: int = 2,
    epochs: int = 200,
    learning_rate: float = 0.02,
    initial_model_state_dict: dict[str, torch.Tensor] | None = None,
    initial_optimizer_state_dict: dict[str, object] | None = None,
    initial_scheduler_state_dict: dict[str, object] | None = None,
    start_epoch: int = 0,
    start_global_step: int = 0,
) -> MinimalTrainingResult:
    batch = GraphBatch.from_graphs([example.graph for example in dataset.examples])
    node_target = torch.cat([example.node_target for example in dataset.examples], dim=0)
    edge_target = torch.cat([example.edge_target for example in dataset.examples], dim=0)
    graph_target = torch.stack([example.graph_target for example in dataset.examples], dim=0)
    target_normalization = compute_target_normalization(dataset)
    node_target, edge_target, graph_target = normalize_targets(
        node_target,
        edge_target,
        graph_target,
        target_normalization,
    )

    examples = dataset.examples
    node_targets = int(examples[0].node_target.shape[-1])
    model = MolecularQuantumGNN(
        atom_vocab_size=128,
        bond_vocab_size=8,
        hidden_dim=hidden_dim,
        num_message_passing_steps=num_message_passing_steps,
        node_targets=node_targets,
        graph_targets=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    if initial_model_state_dict is not None:
        model.load_state_dict(initial_model_state_dict)
    if initial_optimizer_state_dict is not None:
        optimizer.load_state_dict(initial_optimizer_state_dict)
    if initial_scheduler_state_dict is not None:
        scheduler.load_state_dict(initial_scheduler_state_dict)

    loss_history: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = compute_multitask_loss(
            model(batch),
            (node_target, edge_target, graph_target),
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(float(loss.item()))

    embeddings = model.encode_graph_embeddings(batch)
    return MinimalTrainingResult(
        model=model,
        loss_history=loss_history,
        embeddings=embeddings,
        target_normalization=target_normalization,
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        epoch=start_epoch + epochs,
        global_step=start_global_step + epochs,
    )
