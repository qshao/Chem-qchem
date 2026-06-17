# qchem_gnn/adapt/__init__.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import AdaptConfig, AdaptConfigError, resolve_adapt_config
from .registry import METHODS, get_method
from .runner import run


def predict_smiles(smiles: list[str], adapter_path, **kw) -> tuple[np.ndarray, list[int]]:
    header = torch.load(Path(adapter_path), map_location="cpu", weights_only=False)
    adapter_type = header.get("adapter_type", "engine")
    method_cls = METHODS[adapter_type]
    loaded = method_cls.load(adapter_path)
    return method_cls.predict(loaded, smiles, **kw)


__all__ = ["AdaptConfig", "AdaptConfigError", "resolve_adapt_config",
           "run", "get_method", "METHODS", "predict_smiles"]
