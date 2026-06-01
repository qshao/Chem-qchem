from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(inner) for inner in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(metrics), indent=2, sort_keys=True))


def load_metrics(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
