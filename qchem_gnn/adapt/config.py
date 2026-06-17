# qchem_gnn/adapt/config.py
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

VALID_METHODS = {"mlp_head", "finetune", "engine"}
VALID_TASKS = {"regression", "classification"}
VALID_TOP = {"command", "method", "backbone", "task", "dataset", "adapter",
             "training", "split", "outputs", "sweep"}
FINETUNE_ONLY_TRAINING = {"backbone_lr", "grad_clip"}

DEFAULTS = {
    "adapter": {"hidden_dims": [128, 64], "dropout": 0.1},
    "training": {"epochs": 200, "head_lr": 1.0e-3, "backbone_lr": 5.0e-5,
                 "batch_size": 64, "patience": 30, "grad_clip": 1.0, "seed": 42},
    "split": {"test_frac": 0.2, "val_frac": 0.25, "seed": 42, "stratify": True},
    "outputs": {"adapter": None, "report": None},
}


class AdaptConfigError(ValueError):
    pass


@dataclass
class AdaptConfig:
    method: str
    backbone: str
    task: str
    csv: str
    smiles_col: str | None
    targets: object
    adapter: dict
    training: dict
    split: dict
    outputs: dict
    sweep: dict | None


def _deep_merge(base, override):
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def resolve_adapt_config(raw: dict, overrides: dict | None = None) -> AdaptConfig:
    cfg = _deep_merge(raw, overrides or {})

    extra = set(cfg) - VALID_TOP
    if extra:
        raise AdaptConfigError(f"unknown top-level keys: {sorted(extra)}")

    method = cfg.get("method")
    if method not in VALID_METHODS:
        raise AdaptConfigError(f"method must be one of {sorted(VALID_METHODS)}")
    task = cfg.get("task", "regression")
    if task not in VALID_TASKS:
        raise AdaptConfigError(f"task must be one of {sorted(VALID_TASKS)}")

    backbone = cfg.get("backbone")
    if not isinstance(backbone, str) or not backbone.strip():
        raise AdaptConfigError("backbone must be a non-empty path string")

    dataset = cfg.get("dataset") or {}
    csv = dataset.get("csv")
    if not isinstance(csv, str) or not csv.strip():
        raise AdaptConfigError("dataset.csv must be a non-empty path string")
    targets = dataset.get("targets", "auto")
    if targets != "auto":
        if not isinstance(targets, list) or not targets or not all(isinstance(t, str) for t in targets):
            raise AdaptConfigError("dataset.targets must be 'auto' or a non-empty list of strings")
    smiles_col = dataset.get("smiles_col")

    adapter = _deep_merge(DEFAULTS["adapter"], cfg.get("adapter"))
    training = _deep_merge(DEFAULTS["training"], cfg.get("training"))
    split = _deep_merge(DEFAULTS["split"], cfg.get("split"))
    outputs = _deep_merge(DEFAULTS["outputs"], cfg.get("outputs"))

    if method != "finetune":
        bad = FINETUNE_ONLY_TRAINING & set((cfg.get("training") or {}))
        if bad:
            raise AdaptConfigError(f"training keys {sorted(bad)} only valid for method 'finetune'")

    for key in ("test_frac", "val_frac"):
        v = split[key]
        if not (isinstance(v, (int, float)) and 0.0 < float(v) < 1.0):
            raise AdaptConfigError(f"split.{key} must be in (0, 1)")
    if not isinstance(split["seed"], int):
        raise AdaptConfigError("split.seed must be an integer")

    sweep = cfg.get("sweep")
    if sweep is not None:
        _validate_sweep(sweep)

    return AdaptConfig(method=method, backbone=backbone, task=task, csv=csv,
                       smiles_col=smiles_col, targets=targets, adapter=adapter,
                       training=training, split=split, outputs=outputs, sweep=sweep)


def to_raw_dict(cfg: AdaptConfig) -> dict:
    """Round-trip an AdaptConfig back to a raw mapping for re-resolving with overrides."""
    training = deepcopy(cfg.training)
    # Strip finetune-only keys for non-finetune methods so re-validation passes
    if cfg.method != "finetune":
        for k in FINETUNE_ONLY_TRAINING:
            training.pop(k, None)
    raw = {
        "command": "adapt", "method": cfg.method, "backbone": cfg.backbone, "task": cfg.task,
        "dataset": {"csv": cfg.csv, "targets": cfg.targets},
        "adapter": deepcopy(cfg.adapter), "training": training,
        "split": deepcopy(cfg.split), "outputs": deepcopy(cfg.outputs),
    }
    if cfg.smiles_col is not None:
        raw["dataset"]["smiles_col"] = cfg.smiles_col
    if cfg.sweep is not None:
        raw["sweep"] = deepcopy(cfg.sweep)
    return raw


def _validate_sweep(sweep: dict) -> None:
    if not isinstance(sweep, dict) or "grid" not in sweep:
        raise AdaptConfigError("sweep must be a mapping with a 'grid' key")
    grid = sweep["grid"]
    if not isinstance(grid, dict) or not grid:
        raise AdaptConfigError("sweep.grid must be a non-empty mapping")
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise AdaptConfigError(f"sweep.grid['{key}'] must be a non-empty list")
