# qchem_gnn/adapt/methods/base.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn


def make_loss(task: str) -> nn.Module:
    return nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()


def postprocess(task: str, raw: np.ndarray, norm) -> np.ndarray:
    if task == "classification":
        return 1.0 / (1.0 + np.exp(-raw))   # sigmoid
    return norm.inverse(raw)


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1,
                 hidden_dims: tuple[int, ...] = (128, 64), dropout: float = 0.1):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def config(self) -> dict[str, Any]:
        return {"input_dim": self.input_dim, "output_dim": self.output_dim,
                "hidden_dims": list(self.hidden_dims), "dropout": self.dropout}


@dataclass
class TrainResult:
    payload: dict
    test_metrics: dict
    log: dict


@dataclass
class LoadedAdapter:
    adapter_type: str
    payload: dict


class AdaptMethod(Protocol):
    name: str

    def train(self, backbone, model_config: dict, data, train_idx, val_idx, test_idx, cfg) -> TrainResult: ...
    def save(self, path: Path, result: TrainResult, meta: dict) -> None: ...
    @staticmethod
    def load(path: Path) -> LoadedAdapter: ...
    @staticmethod
    def predict(loaded: LoadedAdapter, smiles: list[str], **kw) -> tuple[np.ndarray, list[int]]: ...
