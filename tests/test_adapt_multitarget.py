# tests/test_adapt_multitarget.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qchem_gnn.adapt import predict_smiles
from qchem_gnn.adapt.config import resolve_adapt_config
from qchem_gnn.adapt.runner import run
from qchem_gnn.checkpoint import save_checkpoint
from qchem_gnn.model import MolecularQuantumGNN

MODEL_CONFIG = dict(atom_vocab_size=128, bond_vocab_size=8, hidden_dim=16,
                    num_message_passing_steps=3, graph_targets=2)


@pytest.mark.parametrize("method", ["mlp_head", "finetune", "engine"])
def test_multitarget_regression(tmp_path, method):
    model = MolecularQuantumGNN(**MODEL_CONFIG)
    bb = tmp_path / "bb.pt"
    save_checkpoint(bb, {"model_state_dict": model.state_dict(), "model_config": MODEL_CONFIG})
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "CCN", "CCCCO", "c1ccncc1", "CCCl",
              "CCBr", "CCCCC", "CCOC", "CC#N", "CCS", "CCCCCC", "c1ccccc1O", "CCCN"]
    n = len(smiles)
    csv = tmp_path / "d.csv"
    pd.DataFrame({"smiles": smiles,
                  "a": np.linspace(-3, 3, n),
                  "b": np.linspace(0, 10, n)}).to_csv(csv, index=False)
    out = tmp_path / f"{method}.pt"
    raw = {"command": "adapt", "method": method, "backbone": str(bb), "task": "regression",
           "dataset": {"csv": str(csv), "targets": ["a", "b"]},
           "training": {"epochs": 5}, "outputs": {"adapter": str(out)}}
    cfg = resolve_adapt_config(raw)
    summary = run(cfg)
    assert len(summary["test_metrics"]["per_target"]) == 2
    assert summary["target_names"] == ["a", "b"]
    preds, valid = predict_smiles(["CCO", "c1ccccc1"], out)
    assert preds.shape == (2, 2)
