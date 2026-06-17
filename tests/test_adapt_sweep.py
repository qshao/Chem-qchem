from __future__ import annotations

import pandas as pd

from qchem_gnn.adapt.config import resolve_adapt_config
from qchem_gnn.adapt.sweep import expand_grid, run_sweep
from qchem_gnn.checkpoint import save_checkpoint
from qchem_gnn.model import MolecularQuantumGNN

MODEL_CONFIG = dict(atom_vocab_size=128, bond_vocab_size=8, hidden_dim=16,
                    num_message_passing_steps=2, graph_targets=2)


def test_expand_grid_cartesian():
    cells = expand_grid({"training.epochs": [1, 2], "adapter.dropout": [0.0, 0.1]})
    assert len(cells) == 4
    assert {"training": {"epochs": 1}, "adapter": {"dropout": 0.0}} in cells


def test_run_sweep_writes_report(tmp_path):
    model = MolecularQuantumGNN(**MODEL_CONFIG)
    bb = tmp_path / "bb.pt"
    save_checkpoint(bb, {"model_state_dict": model.state_dict(), "model_config": MODEL_CONFIG})
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "CCN", "CCCCO", "c1ccncc1", "CCCl",
              "CCBr", "CCCCC", "CCOC", "CC#N", "CCS", "CCCCCC", "c1ccccc1O", "CCCN"]
    csv = tmp_path / "d.csv"
    pd.DataFrame({"smiles": smiles, "y": [float(i) for i in range(len(smiles))]}).to_csv(csv, index=False)
    report = tmp_path / "sweep.csv"
    raw = {"command": "adapt", "method": "mlp_head", "backbone": str(bb), "task": "regression",
           "dataset": {"csv": str(csv), "targets": ["y"]}, "training": {"epochs": 2},
           "outputs": {"adapter": str(tmp_path / "a.pt")},
           "sweep": {"grid": {"training.epochs": [2, 3]}, "report": str(report)}}
    cfg = resolve_adapt_config(raw)
    rows = run_sweep(cfg)
    assert len(rows) == 2 and report.exists()
    df = pd.read_csv(report)
    assert "test_mae" in df.columns and len(df) == 2
