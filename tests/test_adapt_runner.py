# tests/test_adapt_runner.py
from __future__ import annotations

import json

import pandas as pd

from qchem_gnn.adapt.config import resolve_adapt_config
from qchem_gnn.adapt.runner import run
from qchem_gnn.checkpoint import save_checkpoint
from qchem_gnn.model import MolecularQuantumGNN

MODEL_CONFIG = dict(atom_vocab_size=128, bond_vocab_size=8, hidden_dim=16,
                    num_message_passing_steps=2, graph_targets=2)


def _setup(tmp_path):
    model = MolecularQuantumGNN(**MODEL_CONFIG)
    bb = tmp_path / "bb.pt"
    save_checkpoint(bb, {"model_state_dict": model.state_dict(), "model_config": MODEL_CONFIG})
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "CCN", "CCCCO", "c1ccncc1", "CCCl",
              "CCBr", "CCCCC", "CCOC", "CC#N", "CCS", "CCCCCC", "c1ccccc1O", "CCCN"]
    y = [float(i) for i in range(len(smiles))]
    csv = tmp_path / "d.csv"
    pd.DataFrame({"smiles": smiles, "y": y}).to_csv(csv, index=False)
    return bb, csv


def test_run_writes_adapter_and_report(tmp_path):
    bb, csv = _setup(tmp_path)
    out = tmp_path / "adapter.pt"
    report = tmp_path / "report.json"
    raw = {"command": "adapt", "method": "mlp_head", "backbone": str(bb),
           "task": "regression", "dataset": {"csv": str(csv), "targets": ["y"]},
           "training": {"epochs": 5}, "outputs": {"adapter": str(out), "report": str(report)}}
    cfg = resolve_adapt_config(raw)
    result = run(cfg)
    assert out.exists() and report.exists()
    assert "test_metrics" in result
    payload = json.loads(report.read_text())
    assert payload["split"]["n_train"] > 0
