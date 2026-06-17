# qchem_gnn/engine_adapter.py
"""Back-compat shim. Implementation moved to qchem_gnn.adapt.methods.engine."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .adapt.backbone import build_graphs, embed_per_layer  # noqa: F401
from .adapt.methods.engine import EngineAdapterHead, EngineMethod


def extract_intermediate_embeddings(smiles_list, model, batch_size=256):
    graphs, valid_idx = build_graphs(smiles_list)
    layer_embs = embed_per_layer(graphs, model, batch_size=batch_size)
    final = np.empty((len(graphs), 0))
    return layer_embs, final, valid_idx


def load_adapter(path):
    loaded = EngineMethod.load(path)
    meta = {
        "label_mu": loaded.payload["label_norm"]["mu"][0],
        "label_sig": loaded.payload["label_norm"]["sigma"][0],
        "layer_scalers": [(np.array(mu), np.array(sig)) for mu, sig in loaded.payload.get("layer_scalers", [])],
        "backbone_checkpoint": loaded.payload["backbone_ckpt"],
        "training_info": loaded.payload.get("training_info", {}),
    }
    adapter = EngineAdapterHead(loaded.payload["hidden_dim"], loaded.payload["num_layers"],
                                output_dim=loaded.payload["output_dim"])
    adapter.load_state_dict(loaded.payload["adapter_state"])
    adapter.eval()
    return adapter, meta


def predict(smiles_list, backbone_ckpt, adapter_path, mode="ensemble", exit_tolerance=0.05, batch_size=256):
    loaded = EngineMethod.load(adapter_path)
    preds, valid_idx = EngineMethod.predict(loaded, smiles_list, mode=mode,
                                            exit_tolerance=exit_tolerance, batch_size=batch_size)
    return preds[:, 0], valid_idx


def save_adapter(path, adapter, layer_scalers, label_mu, label_sig, backbone_checkpoint, training_info=None):
    """Back-compat save_adapter — delegates to old serialisation format."""
    import torch
    state = {
        "adapter_type": "engine",
        "adapter_state_dict": adapter.state_dict(),
        "hidden_dim": adapter.exit_heads[0].in_features,
        "num_layers": adapter.num_layers,
        "layer_scalers": [(mu.tolist(), sig.tolist()) for mu, sig in layer_scalers],
        "label_mu": float(label_mu),
        "label_sig": float(label_sig),
        "backbone_checkpoint": str(backbone_checkpoint),
        "training_info": training_info or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
