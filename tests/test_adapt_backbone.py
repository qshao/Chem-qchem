from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from qchem_gnn.adapt.backbone import build_graphs, embed_final, embed_per_layer, load_backbone
from qchem_gnn.checkpoint import save_checkpoint
from qchem_gnn.model import MolecularQuantumGNN

MODEL_CONFIG = dict(
    atom_vocab_size=128, bond_vocab_size=8, hidden_dim=16,
    num_message_passing_steps=3, graph_targets=2,
)


def _make_ckpt(tmp_path: Path) -> Path:
    model = MolecularQuantumGNN(**MODEL_CONFIG)
    path = tmp_path / "backbone.pt"
    save_checkpoint(path, {"model_state_dict": model.state_dict(), "model_config": MODEL_CONFIG})
    return path


def test_load_backbone_returns_model_and_config(tmp_path):
    path = _make_ckpt(tmp_path)
    model, cfg = load_backbone(path)
    assert cfg["hidden_dim"] == 16
    assert not model.training


def test_build_graphs_skips_invalid():
    graphs, idx = build_graphs(["CCO", "xxx", "c1ccccc1"])
    assert idx == [0, 2]
    assert len(graphs) == 2


def test_embed_final_shape(tmp_path):
    model, _ = load_backbone(_make_ckpt(tmp_path))
    graphs, _ = build_graphs(["CCO", "c1ccccc1", "CC(=O)O"])
    emb = embed_final(graphs, model, batch_size=2)
    assert emb.shape == (3, 16)
    assert np.isfinite(emb).all()


def test_embed_per_layer_shape(tmp_path):
    model, _ = load_backbone(_make_ckpt(tmp_path))
    graphs, _ = build_graphs(["CCO", "c1ccccc1", "CC(=O)O"])
    layers = embed_per_layer(graphs, model, batch_size=2)
    assert len(layers) == 3
    for layer in layers:
        assert layer.shape == (3, 16)
