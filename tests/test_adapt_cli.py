# tests/test_adapt_cli.py
from __future__ import annotations

import pandas as pd
import yaml

from qchem_gnn.cli import main
from qchem_gnn.checkpoint import save_checkpoint
from qchem_gnn.model import MolecularQuantumGNN

MODEL_CONFIG = dict(atom_vocab_size=128, bond_vocab_size=8, hidden_dim=16,
                    num_message_passing_steps=2, graph_targets=2)


def test_adapt_cli_runs_and_writes_adapter(tmp_path):
    model = MolecularQuantumGNN(**MODEL_CONFIG)
    bb = tmp_path / "bb.pt"
    save_checkpoint(bb, {"model_state_dict": model.state_dict(), "model_config": MODEL_CONFIG})
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "CCN", "CCCCO", "c1ccncc1", "CCCl"]
    csv = tmp_path / "d.csv"
    pd.DataFrame({"smiles": smiles, "y": [float(i) for i in range(len(smiles))]}).to_csv(csv, index=False)
    out = tmp_path / "adapter.pt"
    cfg = {"command": "adapt", "method": "mlp_head", "backbone": str(bb),
           "task": "regression", "dataset": {"csv": str(csv), "targets": ["y"]},
           "training": {"epochs": 3}, "outputs": {"adapter": str(out)}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    assert main(["adapt", str(cfg_path)]) == 0
    assert out.exists()


def test_back_compat_imports():
    from qchem_gnn.engine_adapter import EngineAdapterHead  # noqa: F401
    from qchem_gnn.adapters import MLPHead  # noqa: F401
