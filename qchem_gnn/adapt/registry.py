# qchem_gnn/adapt/registry.py
from __future__ import annotations

from .methods.engine import EngineMethod
from .methods.finetune import FinetuneMethod
from .methods.mlp_head import MlpHeadMethod

METHODS = {
    "mlp_head": MlpHeadMethod,
    "finetune": FinetuneMethod,
    "engine": EngineMethod,
}


def get_method(name: str):
    if name not in METHODS:
        raise KeyError(f"Unknown adapt method '{name}'. Valid: {sorted(METHODS)}")
    return METHODS[name]()
