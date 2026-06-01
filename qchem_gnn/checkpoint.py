from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def _normalize_path(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(Path(value).expanduser().resolve(strict=False))


def _normalize_subset_ids(value: Any) -> tuple[int, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return tuple(int(part) for part in parts)
    return tuple(int(part) for part in value)


def normalize_dataset_config(dataset_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "csv": _normalize_path(dataset_config.get("csv")),
        "dataset_root": _normalize_path(dataset_config.get("dataset_root")),
        "subset_ids": _normalize_subset_ids(dataset_config.get("subset_ids")),
        "geometry": _normalize_path(dataset_config.get("geometry")),
        "results": _normalize_path(dataset_config.get("results")),
        "use_results": bool(dataset_config.get("use_results", False)),
        "limit": int(dataset_config.get("limit", 16)),
        "limit_per_shard": int(dataset_config.get("limit_per_shard", 16)),
    }


def build_checkpoint_state(
    *,
    loss_history,
    embeddings,
    model_state_dict,
    optimizer_state_dict,
    epoch,
    global_step,
    target_normalization,
    dataset_config,
    split_metadata,
    model_config,
    run_metadata,
    scheduler_state_dict=None,
) -> dict[str, Any]:
    state = {
        "loss_history": loss_history,
        "embeddings": embeddings,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "target_normalization": target_normalization,
        "dataset_config": normalize_dataset_config(dataset_config),
        "split_metadata": split_metadata,
        "model_config": model_config,
        "run_metadata": dict(run_metadata),
    }
    if scheduler_state_dict is not None:
        state["scheduler_state_dict"] = scheduler_state_dict
    return state
