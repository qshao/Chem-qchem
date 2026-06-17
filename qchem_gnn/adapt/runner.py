# qchem_gnn/adapt/runner.py
from __future__ import annotations

import json
from pathlib import Path

from .backbone import load_backbone
from .config import AdaptConfig
from .data import load_dataset, stratified_split
from .registry import get_method


def run(cfg: AdaptConfig) -> dict:
    data = load_dataset(cfg.csv, cfg.smiles_col, cfg.targets, cfg.task)
    train_idx, val_idx, test_idx = stratified_split(
        data.targets, cfg.task,
        test_frac=cfg.split["test_frac"], val_frac=cfg.split["val_frac"],
        seed=int(cfg.split["seed"]), stratify=bool(cfg.split["stratify"]),
    )
    backbone, model_config = load_backbone(cfg.backbone)
    method = get_method(cfg.method)

    result = method.train(backbone, model_config, data, train_idx, val_idx, test_idx, cfg)

    meta = {
        "backbone_ckpt": str(Path(cfg.backbone).resolve()),
        "target_names": data.target_names,
        "task": cfg.task,
        "training_info": {
            "dataset": Path(cfg.csv).name,
            "method": cfg.method,
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "target_col": ", ".join(data.target_names),
        },
    }
    if cfg.outputs.get("adapter"):
        method.save(cfg.outputs["adapter"], result, meta)

    summary = {
        "method": cfg.method,
        "test_metrics": result.test_metrics,
        "split": {"n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx)},
        "log": result.log,
        "target_names": data.target_names,
    }
    if cfg.outputs.get("report"):
        rp = Path(cfg.outputs["report"]); rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(summary, indent=2))
    return summary
