from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .losses import compute_multitask_loss as _compute_weighted_multitask_loss

Predictions = tuple[torch.Tensor, ...]
Targets = tuple[torch.Tensor, ...] | Mapping[str, Any]


def compute_multitask_loss(
    predictions: Predictions,
    targets: Targets,
    node_weight: float = 1.0,
    edge_weight: float = 1.0,
    graph_weight: float = 0.5,
) -> torch.Tensor:
    return _compute_weighted_multitask_loss(
        predictions,
        targets,
        weights={
            "atom": node_weight,
            "edge": edge_weight,
            "graph": graph_weight,
        },
    )
