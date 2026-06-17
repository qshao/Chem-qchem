"""
ENGINE-style tunable side structure for frozen GNN backbones.

Adapts the ENGINE method (Zhu et al., https://github.com/zhuyun97/engine)
from text-graph LLM+GNN tuning to molecular property prediction with a
pretrained message-passing GNN backbone.

The backbone is always frozen. Only the side structure is trained:

    h_0 = zeros
    h_i = proj_i(gnn_state_i) * σ(α_i)  +  h_{i-1} * (1 − σ(α_i))
    exit_pred_i = head_i(h_i)
    L_train = Σ_i  MSE(exit_pred_i, y)

Public API
----------
extract_intermediate_embeddings   — run frozen GNN, collect per-layer states
EngineAdapterHead                 — trainable side structure (nn.Module)
save_adapter                      — serialise adapter + normalisation stats
load_adapter                      — deserialise adapter
predict                           — end-to-end SMILES → property prediction
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .checkpoint import load_checkpoint
from .graph import GraphBatch, GraphData, build_graph_from_smiles
from .model import MolecularQuantumGNN


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_intermediate_embeddings(
    smiles_list: list[str],
    model: MolecularQuantumGNN,
    batch_size: int = 256,
) -> tuple[list[np.ndarray], np.ndarray, list[int]]:
    """
    Run the frozen backbone and collect pooled graph embeddings at every
    message-passing layer, plus the final mol embedding.

    Returns
    -------
    layer_embs  list of [N, H] arrays, one per message-passing step
    final_embs  [N, H] array  — output of molecular_embedding_head
    valid_idx   list of indices into smiles_list that parsed successfully
    """
    valid_idx: list[int] = []
    graphs: list[GraphData] = []
    for i, smi in enumerate(smiles_list):
        try:
            graphs.append(build_graph_from_smiles(smi))
            valid_idx.append(i)
        except Exception:
            pass

    encoder = model.encoder
    num_layers = len(encoder.message_passing_blocks)
    per_layer: list[list[np.ndarray]] = [[] for _ in range(num_layers)]
    final_chunks: list[np.ndarray] = []

    model.eval()
    for s in range(0, len(graphs), batch_size):
        batch = GraphBatch.from_graphs(graphs[s : s + batch_size])
        with torch.no_grad():
            node_states = encoder.atom_encoder(batch.atomic_numbers)
            edge_states = encoder.bond_encoder(batch.edge_attr.squeeze(-1))
            pooled = None
            for i, block in enumerate(encoder.message_passing_blocks):
                node_states = block(node_states, batch.edge_index, edge_states)
                pooled = encoder._mean_pool(node_states, batch.batch, batch.num_graphs)
                per_layer[i].append(pooled.cpu().numpy())
            final_chunks.append(encoder.molecular_embedding_head(pooled).cpu().numpy())

    layer_embs = [np.concatenate(c) for c in per_layer]
    final_embs = np.concatenate(final_chunks) if final_chunks else np.empty((0, 0))
    return layer_embs, final_embs, valid_idx


# ---------------------------------------------------------------------------
# ENGINE side structure
# ---------------------------------------------------------------------------

class EngineAdapterHead(nn.Module):
    """
    Tunable side structure that blends frozen GNN layer states via
    learnable alpha gates into an accumulating representation:

        h_i = proj_i(x_i) * sigmoid(α_i)  +  h_{i-1} * (1 − sigmoid(α_i))

    Each layer gets its own regression head; training sums MSE across all
    exits so every head receives a direct gradient signal.

    Parameters
    ----------
    hidden_dim   dimension of the backbone's hidden states
    num_layers   number of backbone message-passing steps (= number of exits)
    """

    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        # α_i = 0  →  sigmoid(0) = 0.5  (equal blend at init)
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(num_layers)]
        )
        self.exit_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(self, layer_tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        """Return one [B, 1] prediction per exit layer."""
        h = torch.zeros_like(layer_tensors[0])
        preds = []
        for proj, alpha, head, x in zip(
            self.projections, self.alphas, self.exit_heads, layer_tensors
        ):
            gate = torch.sigmoid(alpha)
            h = proj(x) * gate + h * (1.0 - gate)
            preds.append(head(h))
        return preds

    @torch.no_grad()
    def predict_ensemble(self, layer_tensors: list[torch.Tensor]) -> np.ndarray:
        """Average of all exit-head predictions → [B] numpy array."""
        preds = self.forward(layer_tensors)
        return torch.stack(preds, dim=0).mean(0).squeeze(1).cpu().numpy()

    @torch.no_grad()
    def predict_early_exit(
        self,
        layer_tensors: list[torch.Tensor],
        tolerance: float = 0.05,
        min_layers: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Exit each molecule as soon as the standard deviation across all
        accumulated exit-head predictions falls below `tolerance`.

        Returns (predictions [B], exit_layer_indices [B]).
        """
        B = layer_tensors[0].shape[0]
        h = torch.zeros_like(layer_tensors[0])
        accumulated: list[torch.Tensor] = []
        active = torch.ones(B, dtype=torch.bool)
        final_pred = torch.zeros(B)
        exit_layer = torch.full((B,), self.num_layers - 1, dtype=torch.long)

        for i, (proj, alpha, head, x) in enumerate(
            zip(self.projections, self.alphas, self.exit_heads, layer_tensors)
        ):
            gate = torch.sigmoid(alpha)
            h = proj(x) * gate + h * (1.0 - gate)
            pred = head(h).squeeze(1)
            accumulated.append(pred)

            if i >= min_layers and len(accumulated) >= 2:
                std = torch.stack(accumulated, dim=0).std(dim=0)
                exiting = active & (std < tolerance)
                if exiting.any():
                    exit_layer[exiting] = i
                    final_pred[exiting] = pred[exiting]
                    active[exiting] = False

        if active.any():
            final_pred[active] = accumulated[-1][active]

        return final_pred.cpu().numpy(), exit_layer.cpu().numpy()


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_adapter(
    path: Path | str,
    adapter: EngineAdapterHead,
    layer_scalers: list[tuple[np.ndarray, np.ndarray]],
    label_mu: float,
    label_sig: float,
    backbone_checkpoint: str,
    training_info: dict[str, Any] | None = None,
) -> None:
    """Save adapter weights + normalisation statistics to a .pt file."""
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


def load_adapter(path: Path | str) -> tuple[EngineAdapterHead, dict[str, Any]]:
    """
    Load a saved adapter.

    Returns (adapter, meta) where meta contains label_mu, label_sig,
    layer_scalers, backbone_checkpoint, and training_info.
    """
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    adapter = EngineAdapterHead(
        hidden_dim=state["hidden_dim"],
        num_layers=state["num_layers"],
    )
    adapter.load_state_dict(state["adapter_state_dict"])
    adapter.eval()
    meta = {
        "label_mu": state["label_mu"],
        "label_sig": state["label_sig"],
        "layer_scalers": [
            (np.array(mu), np.array(sig))
            for mu, sig in state["layer_scalers"]
        ],
        "backbone_checkpoint": state["backbone_checkpoint"],
        "training_info": state.get("training_info", {}),
    }
    return adapter, meta


# ---------------------------------------------------------------------------
# End-to-end prediction
# ---------------------------------------------------------------------------

def predict(
    smiles_list: list[str],
    backbone_ckpt: Path | str,
    adapter_path: Path | str,
    mode: str = "ensemble",
    exit_tolerance: float = 0.05,
    batch_size: int = 256,
) -> tuple[np.ndarray, list[int]]:
    """
    Predict property values for a list of SMILES strings.

    Parameters
    ----------
    smiles_list      list of SMILES to score
    backbone_ckpt    path to the pretrained GNN checkpoint
    adapter_path     path to the saved ENGINE adapter (.pt)
    mode             "ensemble" or "early_exit"
    exit_tolerance   std threshold for early exit (in normalised units)
    batch_size       number of molecules per embedding batch

    Returns
    -------
    predictions      [N_valid] numpy array in original label units
    valid_idx        indices into smiles_list that were successfully parsed
    """
    # Load backbone
    ckpt = load_checkpoint(Path(backbone_ckpt))
    model = MolecularQuantumGNN(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])

    # Load adapter
    adapter, meta = load_adapter(adapter_path)
    layer_scalers = meta["layer_scalers"]
    label_mu = meta["label_mu"]
    label_sig = meta["label_sig"]

    # Extract layer embeddings
    layer_embs, _, valid_idx = extract_intermediate_embeddings(
        smiles_list, model, batch_size=batch_size
    )

    # Apply per-layer normalisation
    tensors = [
        torch.as_tensor((emb - mu) / sig, dtype=torch.float32)
        for emb, (mu, sig) in zip(layer_embs, layer_scalers)
    ]

    # Predict
    if mode == "early_exit":
        tol_norm = exit_tolerance / label_sig
        pred_norm, _ = adapter.predict_early_exit(tensors, tolerance=tol_norm)
    else:
        pred_norm = adapter.predict_ensemble(tensors)

    predictions = pred_norm * label_sig + label_mu
    return predictions, valid_idx
