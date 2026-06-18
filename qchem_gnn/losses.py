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
    negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if z_a.shape != z_b.shape:
        raise ValueError("z_a and z_b must have the same shape")
    if z_a.shape[0] < 2:
        raise ValueError("contrastive loss needs at least 2 examples for in-batch negatives")

    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = (z_a @ z_b.t()) / temperature
    if negative_mask is not None:
        if negative_mask.shape != logits.shape or negative_mask.dtype != torch.bool:
            raise ValueError(
                f"negative_mask must be a bool tensor of shape {logits.shape}, "
                f"got shape={negative_mask.shape} dtype={negative_mask.dtype}"
            )
        eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(negative_mask & ~eye, float("-inf"))
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_ab + loss_ba)


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    n = matrix.shape[0]
    return matrix.flatten()[:-1].reshape(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    *,
    sim_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    if z_a.shape != z_b.shape:
        raise ValueError("z_a and z_b must have the same shape")
    if z_a.shape[0] < 2:
        raise ValueError("vicreg loss needs at least 2 examples for variance/covariance statistics")

    batch, dim = z_a.shape

    # Invariance: pull the two views together.
    inv = F.mse_loss(z_a, z_b)

    # Variance: hinge keeps each dimension's batch std above gamma (anti-collapse).
    std_a = torch.sqrt(z_a.var(dim=0) + eps)
    std_b = torch.sqrt(z_b.var(dim=0) + eps)
    var = torch.relu(gamma - std_a).mean() + torch.relu(gamma - std_b).mean()

    # Covariance: decorrelate dimensions via the off-diagonal of the empirical covariance.
    za_c = z_a - z_a.mean(dim=0)
    zb_c = z_b - z_b.mean(dim=0)
    cov_a = (za_c.t() @ za_c) / (batch - 1)
    cov_b = (zb_c.t() @ zb_c) / (batch - 1)
    cov = _off_diagonal(cov_a).pow(2).sum() / dim + _off_diagonal(cov_b).pow(2).sum() / dim

    return sim_weight * inv + var_weight * var + cov_weight * cov
