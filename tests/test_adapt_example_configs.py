from __future__ import annotations

from pathlib import Path

import pytest

from qchem_gnn.adapt.config import resolve_adapt_config
from qchem_gnn.config import load_yaml_config

CONFIGS = [
    "configs/adapt_mlp_head_solubility.yaml",
    "configs/adapt_finetune_solubility.yaml",
    "configs/adapt_engine_solubility.yaml",
    "configs/adapt_engine_epoch_sweep.yaml",
]


@pytest.mark.parametrize("path", CONFIGS)
def test_example_config_resolves(path):
    cfg = resolve_adapt_config(load_yaml_config(Path(path)))
    assert cfg.method in {"mlp_head", "finetune", "engine"}


def test_sweep_config_has_grid():
    cfg = resolve_adapt_config(load_yaml_config(Path("configs/adapt_engine_epoch_sweep.yaml")))
    assert cfg.sweep is not None and "grid" in cfg.sweep


def test_sweep_config_has_five_cells():
    cfg = resolve_adapt_config(load_yaml_config(Path("configs/adapt_engine_epoch_sweep.yaml")))
    assert cfg.sweep is not None
    grid = cfg.sweep["grid"]
    import itertools
    cells = list(itertools.product(*grid.values()))
    assert len(cells) == 5
