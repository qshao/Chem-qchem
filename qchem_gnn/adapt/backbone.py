from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..checkpoint import load_checkpoint
from ..graph import GraphBatch, GraphData, build_graph_from_smiles
from ..model import MolecularQuantumGNN


def load_backbone(path) -> tuple[MolecularQuantumGNN, dict]:
    ckpt = load_checkpoint(Path(path))
    model = MolecularQuantumGNN(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["model_config"]


def build_graphs(smiles: list[str]) -> tuple[list[GraphData], list[int]]:
    graphs: list[GraphData] = []
    valid_idx: list[int] = []
    for i, smi in enumerate(smiles):
        try:
            graphs.append(build_graph_from_smiles(smi))
            valid_idx.append(i)
        except Exception:
            pass
    return graphs, valid_idx


def embed_final(graphs: list[GraphData], model: MolecularQuantumGNN, batch_size: int = 256) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for s in range(0, len(graphs), batch_size):
        batch = GraphBatch.from_graphs(graphs[s : s + batch_size])
        with torch.no_grad():
            chunks.append(model.encode_graph_embeddings(batch).cpu().numpy())
    if not chunks:
        return np.empty((0, model.encoder.atom_encoder.embedding_dim), dtype=np.float32)
    return np.concatenate(chunks)


def embed_per_layer(graphs: list[GraphData], model: MolecularQuantumGNN, batch_size: int = 256) -> list[np.ndarray]:
    encoder = model.encoder
    num_layers = len(encoder.message_passing_blocks)
    per_layer: list[list[np.ndarray]] = [[] for _ in range(num_layers)]

    model.eval()
    for s in range(0, len(graphs), batch_size):
        batch = GraphBatch.from_graphs(graphs[s : s + batch_size])
        with torch.no_grad():
            node_states = encoder.atom_encoder(batch.atomic_numbers)
            edge_states = encoder.bond_encoder(batch.edge_attr.squeeze(-1))
            for i, block in enumerate(encoder.message_passing_blocks):
                node_states = block(node_states, batch.edge_index, edge_states)
                pooled = encoder._mean_pool(node_states, batch.batch, batch.num_graphs)
                per_layer[i].append(pooled.cpu().numpy())
    return [np.concatenate(c) for c in per_layer]
