from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .graph import GraphBatch
from .losses import compute_multitask_loss
from .minimal import MinimalQuantumDataset
from .model import MolecularQuantumGNN
from .quantum_data import compute_target_normalization, normalize_targets


@dataclass(frozen=True)
class PretrainingResult:
    model: MolecularQuantumGNN
    aux_head_state_dict: dict[str, object]
    loss_history: list[float]
    embeddings: torch.Tensor
    target_normalization: dict[str, torch.Tensor]
    aux_normalization: dict[str, torch.Tensor] | None
    optimizer_state_dict: dict[str, object]
    scheduler_state_dict: dict[str, object]
    epoch: int
    global_step: int


def _stack_aux_targets(dataset: MinimalQuantumDataset) -> torch.Tensor | None:
    aux_targets = [example.aux_target for example in dataset.examples if example.aux_target is not None]
    if not aux_targets:
        return None
    return torch.stack(aux_targets, dim=0)


def _normalize_aux_targets(aux_targets: torch.Tensor) -> dict[str, torch.Tensor]:
    mean = aux_targets.mean(dim=0)
    std = aux_targets.std(dim=0, unbiased=False).clamp_min(1e-6)
    return {"aux_mean": mean, "aux_std": std}


def pretrain_on_minimal_dataset(
    dataset: MinimalQuantumDataset,
    hidden_dim: int = 32,
    num_message_passing_steps: int = 2,
    epochs: int = 200,
    learning_rate: float = 0.02,
    aux_weight: float = 0.1,
    initial_model_state_dict: dict[str, torch.Tensor] | None = None,
    initial_optimizer_state_dict: dict[str, object] | None = None,
    initial_scheduler_state_dict: dict[str, object] | None = None,
    start_epoch: int = 0,
    start_global_step: int = 0,
) -> PretrainingResult:
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

    aux_targets = _stack_aux_targets(dataset)
    aux_normalization = _normalize_aux_targets(aux_targets) if aux_targets is not None else None
    normalized_aux_targets = None
    if aux_targets is not None:
        normalized_aux_targets = (aux_targets - aux_normalization["aux_mean"]) / aux_normalization["aux_std"]

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
    aux_dim = int(aux_targets.shape[-1]) if aux_targets is not None else 0
    aux_head = nn.Linear(hidden_dim, aux_dim) if aux_dim else None
    params = list(model.parameters())
    if aux_head is not None:
        params.extend(aux_head.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
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
        node_pred, edge_pred, graph_pred, embedding = model(batch)
        if aux_head is not None and normalized_aux_targets is not None:
            aux_pred = aux_head(embedding)
            predictions = (node_pred, edge_pred, graph_pred, embedding, aux_pred)
            targets = {
                "atom": node_target,
                "edge": edge_target,
                "graph": graph_target,
                "aux": normalized_aux_targets,
            }
            weights = {"atom": 1.0, "edge": 1.0, "graph": 0.5, "aux": aux_weight}
        else:
            predictions = (node_pred, edge_pred, graph_pred, embedding)
            targets = (node_target, edge_target, graph_target)
            weights = {"atom": 1.0, "edge": 1.0, "graph": 0.5}
        loss = compute_multitask_loss(predictions, targets, weights=weights)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(float(loss.item()))

    embeddings = model.encode_graph_embeddings(batch)
    return PretrainingResult(
        model=model,
        aux_head_state_dict=aux_head.state_dict() if aux_head is not None else {},
        loss_history=loss_history,
        embeddings=embeddings,
        target_normalization=target_normalization,
        aux_normalization=aux_normalization,
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        epoch=start_epoch + epochs,
        global_step=start_global_step + epochs,
    )
