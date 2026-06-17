# qchem_gnn/adapters.py
"""Back-compat shim. Implementation moved to qchem_gnn.adapt."""
from __future__ import annotations

from .adapt import predict_smiles
from .adapt.backbone import build_graphs, embed_final as _embed_final
from .adapt.methods.base import MLPHead


def embed_smiles(smiles_list, model, batch_size=256):
    graphs, valid_idx = build_graphs(smiles_list)
    return _embed_final(graphs, model, batch_size=batch_size), valid_idx


def predict(smiles_list, adapter_path, batch_size=256):
    return predict_smiles(smiles_list, adapter_path, batch_size=batch_size)
