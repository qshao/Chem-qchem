# tests/test_adapt_mlp_head.py
from __future__ import annotations

import numpy as np
import torch

from qchem_gnn.adapt.backbone import build_graphs, load_backbone
from qchem_gnn.adapt.data import AdaptData
from qchem_gnn.adapt.methods.mlp_head import MlpHeadMethod
from qchem_gnn.checkpoint import save_checkpoint
from qchem_gnn.model import MolecularQuantumGNN

MODEL_CONFIG = dict(atom_vocab_size=128, bond_vocab_size=8, hidden_dim=16,
                    num_message_passing_steps=2, graph_targets=2)


def _ckpt(tmp_path):
    model = MolecularQuantumGNN(**MODEL_CONFIG)
    p = tmp_path / "bb.pt"
    save_checkpoint(p, {"model_state_dict": model.state_dict(), "model_config": MODEL_CONFIG})
    return p


class _Cfg:
    def __init__(self):
        self.adapter = {"hidden_dims": [8], "dropout": 0.0}
        self.training = {"epochs": 30, "head_lr": 1e-2, "batch_size": 8, "patience": 50, "seed": 0}


def _data():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "CCN", "CCCCO", "c1ccncc1", "CCCl"]
    graphs, idx = build_graphs(smiles)
    y = np.linspace(-3, 3, len(graphs)).reshape(-1, 1).astype(np.float32)
    return AdaptData(smiles=[smiles[i] for i in idx], graphs=graphs, targets=y,
                     target_names=["y"], valid_idx=idx, task="regression")


def test_mlp_head_train_save_load_predict(tmp_path):
    ckpt = _ckpt(tmp_path)
    backbone, cfg_model = load_backbone(ckpt)
    data = _data()
    n = len(data.graphs)
    method = MlpHeadMethod()
    result = method.train(backbone, cfg_model, data,
                          train_idx=list(range(n - 4)), val_idx=[n - 4, n - 3],
                          test_idx=[n - 2, n - 1], cfg=_Cfg())
    assert "mae" in result.test_metrics
    assert result.payload["adapter_type"] == "mlp_head"

    out = tmp_path / "mlp.pt"
    method.save(out, result, meta={"backbone_ckpt": str(ckpt), "target_names": ["y"], "task": "regression"})
    loaded = MlpHeadMethod.load(out)
    preds, valid = MlpHeadMethod.predict(loaded, ["CCO", "bad", "c1ccccc1"])
    assert valid == [0, 2]
    assert preds.shape == (2, 1)
    assert np.isfinite(preds).all()
