# tests/test_adapt_config.py
from __future__ import annotations

import pytest

from qchem_gnn.adapt.config import AdaptConfig, AdaptConfigError, resolve_adapt_config


def _base(**over):
    cfg = {"command": "adapt", "method": "finetune", "backbone": "bb.pt",
           "task": "regression", "dataset": {"csv": "d.csv", "targets": ["y"]},
           "outputs": {"adapter": "out.pt"}}
    cfg.update(over)
    return cfg


def test_resolve_fills_defaults():
    cfg = resolve_adapt_config(_base())
    assert isinstance(cfg, AdaptConfig)
    assert cfg.split["test_frac"] == 0.2
    assert cfg.training["epochs"] >= 1
    assert cfg.targets == ["y"]


def test_unknown_method_rejected():
    with pytest.raises(AdaptConfigError):
        resolve_adapt_config(_base(method="lora"))


def test_unknown_task_rejected():
    with pytest.raises(AdaptConfigError):
        resolve_adapt_config(_base(task="ranking"))


def test_backbone_lr_rejected_for_mlp_head():
    cfg = _base(method="mlp_head")
    cfg["training"] = {"backbone_lr": 1e-4}
    with pytest.raises(AdaptConfigError):
        resolve_adapt_config(cfg)


def test_overrides_applied():
    cfg = resolve_adapt_config(_base(), overrides={"training": {"epochs": 999}})
    assert cfg.training["epochs"] == 999


def test_missing_backbone_rejected():
    cfg = _base(); del cfg["backbone"]
    with pytest.raises(AdaptConfigError):
        resolve_adapt_config(cfg)
