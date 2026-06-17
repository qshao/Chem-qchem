from __future__ import annotations

import numpy as np

from qchem_gnn.adapt.backbone import build_graphs, load_backbone
from qchem_gnn.adapt.data import AdaptData
from qchem_gnn.adapt.methods.engine import EngineMethod
from qchem_gnn.checkpoint import save_checkpoint
from qchem_gnn.model import MolecularQuantumGNN

MODEL_CONFIG = dict(atom_vocab_size=128, bond_vocab_size=8, hidden_dim=16,
                    num_message_passing_steps=3, graph_targets=2)


def _ckpt(tmp_path):
    model = MolecularQuantumGNN(**MODEL_CONFIG)
    p = tmp_path / "bb.pt"
    save_checkpoint(p, {"model_state_dict": model.state_dict(), "model_config": MODEL_CONFIG})
    return p


class _Cfg:
    def __init__(self):
        self.adapter = {}
        self.training = {"epochs": 20, "head_lr": 5e-3, "seed": 0}


def _data():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "CCN", "CCCCO", "c1ccncc1", "CCCl"]
    graphs, idx = build_graphs(smiles)
    y = np.linspace(-3, 3, len(graphs)).reshape(-1, 1).astype(np.float32)
    return AdaptData(smiles=[smiles[i] for i in idx], graphs=graphs, targets=y,
                     target_names=["y"], valid_idx=idx, task="regression")


def test_engine_train_save_load_predict_modes(tmp_path):
    ckpt = _ckpt(tmp_path)
    backbone, cfg_model = load_backbone(ckpt)
    data = _data(); n = len(data.graphs)
    method = EngineMethod()
    result = method.train(backbone, cfg_model, data,
                          train_idx=list(range(n - 4)), val_idx=[n - 4, n - 3],
                          test_idx=[n - 2, n - 1], cfg=_Cfg())
    assert result.payload["adapter_type"] == "engine"
    out = tmp_path / "eng.pt"
    method.save(out, result, meta={"backbone_ckpt": str(ckpt), "target_names": ["y"], "task": "regression"})
    loaded = EngineMethod.load(out)
    preds, valid = EngineMethod.predict(loaded, ["CCO", "bad", "c1ccccc1"], mode="ensemble")
    assert valid == [0, 2] and preds.shape == (2, 1)
    preds_ee, _ = EngineMethod.predict(loaded, ["CCO", "c1ccccc1"], mode="early_exit", exit_tolerance=0.1)
    assert preds_ee.shape == (2, 1) and np.isfinite(preds_ee).all()
