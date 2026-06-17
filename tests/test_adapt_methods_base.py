from __future__ import annotations

import torch

from qchem_gnn.adapt.methods.base import MLPHead


def test_mlphead_output_dim():
    head = MLPHead(input_dim=16, output_dim=3, hidden_dims=(8,), dropout=0.0)
    out = head(torch.zeros(5, 16))
    assert out.shape == (5, 3)


def test_mlphead_config_round_trip():
    head = MLPHead(input_dim=16, output_dim=2, hidden_dims=(8, 4), dropout=0.1)
    cfg = head.config()
    clone = MLPHead(**cfg)
    clone.load_state_dict(head.state_dict())
    assert cfg["output_dim"] == 2 and cfg["hidden_dims"] == [8, 4]
