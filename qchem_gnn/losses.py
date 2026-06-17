from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

Predictions = tuple[torch.Tensor, ...]
Targets = tuple[torch.Tensor, ...] | Mapping[str, torch.Tensor | None]

_DEFAULT_WEIGHTS: dict[str, float] = {
    "atom": 1.0,
    "edge": 1.0,
    "graph": 0.5,
    "aux": 0.0,
    "consistency": 0.0,
}


def _merged_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    merged = dict(_DEFAULT_WEIGHTS)
    if weights is not None:
        merged.update(weights)
    return merged


def compute_multitask_loss(
    predictions: Predictions,
    targets: Targets,
    weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    if len(predictions) < 3:
        raise ValueError("predictions must contain atom, edge, and graph outputs")

    atom_pred, edge_pred, graph_pred = predictions[:3]
    mol_embedding = predictions[3] if len(predictions) > 3 else None
    aux_pred = predictions[4] if len(predictions) > 4 else None

    if isinstance(targets, Mapping):
        atom_target = targets.get("atom")
        edge_target = targets.get("edge")
        graph_target = targets.get("graph")
        aux_target = targets.get("aux")
        consistency_target = targets.get("consistency")
    else:
        atom_target, edge_target, graph_target = targets[:3]
        aux_target = None
        consistency_target = None

    weight_map = _merged_weights(weights)
    loss = torch.zeros((), dtype=atom_pred.dtype, device=atom_pred.device)

    if atom_target is not None:
        loss = loss + weight_map["atom"] * F.mse_loss(atom_pred, atom_target)
    if edge_target is not None:
        loss = loss + weight_map["edge"] * F.mse_loss(edge_pred, edge_target)
    if graph_target is not None:
        loss = loss + weight_map["graph"] * F.mse_loss(graph_pred, graph_target)

    if mol_embedding is not None:
        if consistency_target is not None and weight_map["consistency"]:
            loss = loss + weight_map["consistency"] * F.mse_loss(mol_embedding, consistency_target)

    if aux_target is not None and weight_map["aux"]:
        if aux_pred is not None:
            loss = loss + weight_map["aux"] * F.mse_loss(aux_pred, aux_target)
        elif mol_embedding is not None and mol_embedding.shape == aux_target.shape:
            loss = loss + weight_map["aux"] * F.mse_loss(mol_embedding, aux_target)

    return loss


def info_nce_contrastive_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    if z_a.shape != z_b.shape:
        raise ValueError("z_a and z_b must have the same shape")
    if z_a.shape[0] < 2:
        raise ValueError("contrastive loss needs at least 2 examples for in-batch negatives")

    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = (z_a @ z_b.t()) / temperature
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_ab + loss_ba)
