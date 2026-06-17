from __future__ import annotations

from argparse import Namespace
import math
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

ConfigDict = dict[str, Any]


class ConfigError(ValueError):
    pass


VALID_COMMANDS = {"train", "pretrain", "contrastive-pretrain", "export-embeddings", "eval", "downstream"}
VALID_TOP_LEVEL_KEYS = {
    "command",
    "dataset",
    "model",
    "training",
    "contrastive",
    "outputs",
    "inference",
    "downstream",
}

VALID_SECTION_KEYS = {
    "dataset": {"csv", "dataset_root", "subset_ids", "geometry", "results", "use_results", "limit", "limit_per_shard"},
    "model": {"hidden_dim", "message_passing_steps"},
    "training": {"epochs", "learning_rate", "aux_weight", "resume_from"},
    "contrastive": {
        "batch_size",
        "supervised_weight",
        "contrastive_weight",
        "temperature",
        "teacher_weight",
        "energy_temperature",
        "hidden_dim_3d",
        "num_rbf",
        "cutoff",
        "message_passing_steps_3d",
        "conformer_pool_mode",
        "seed",
    },
    "outputs": {"checkpoint", "metrics", "embeddings"},
    "inference": {"checkpoint", "output"},
    "downstream": {"kind", "epochs", "learning_rate", "fractions", "seed"},
}

VALID_DOWNSTREAM_KINDS = {"fine-tune", "linear-probe", "morgan-baseline", "sample-efficiency"}

DEFAULT_CONFIG: ConfigDict = {
    "command": None,
    "dataset": {
        "csv": None,
        "dataset_root": None,
        "subset_ids": [],
        "geometry": None,
        "results": None,
        "use_results": False,
        "limit": 16,
        "limit_per_shard": 16,
    },
    "model": {
        "hidden_dim": 32,
        "message_passing_steps": 2,
    },
    "training": {
        "epochs": 200,
        "learning_rate": 0.02,
        "aux_weight": 0.1,
        "resume_from": None,
    },
    "contrastive": {
        "batch_size": 8,
        "supervised_weight": 1.0,
        "contrastive_weight": 1.0,
        "temperature": 0.1,
        "teacher_weight": 1.0,
        "energy_temperature": 298.15,
        "hidden_dim_3d": 32,
        "num_rbf": 16,
        "cutoff": 5.0,
        "message_passing_steps_3d": 2,
        "conformer_pool_mode": "mean",
        "seed": 0,
    },
    "outputs": {
        "checkpoint": None,
        "metrics": None,
        "embeddings": None,
    },
    "inference": {
        "checkpoint": None,
        "output": None,
    },
    "downstream": {
        "kind": "fine-tune",
        "epochs": 50,
        "learning_rate": 0.01,
        "fractions": [0.01, 0.05, 0.1, 1.0],
        "seed": 0,
    },
}


def load_yaml_config(path: str | Path) -> ConfigDict:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("PyYAML is required for YAML config support. Install pyyaml.") from exc

    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Failed to read YAML config {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError("YAML config must be a mapping at the top level")
    return loaded


def _deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _drop_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none_values(inner) for key, inner in value.items() if inner is not None}
    if isinstance(value, list):
        return [_drop_none_values(inner) for inner in value if inner is not None]
    return value


def resolve_config(config: ConfigDict, overrides: ConfigDict | None = None) -> ConfigDict:
    resolved = _deep_merge(DEFAULT_CONFIG, _drop_none_values(config))
    if overrides:
        resolved = _deep_merge(resolved, _drop_none_values(overrides))
    _validate_config(resolved)
    return resolved


def _validate_config(config: ConfigDict) -> None:
    extra_keys = sorted(set(config) - VALID_TOP_LEVEL_KEYS)
    if extra_keys:
        raise ConfigError("unknown top-level config keys: " + ", ".join(extra_keys))

    command = config.get("command")
    if command not in VALID_COMMANDS:
        raise ConfigError(f"command must be one of {sorted(VALID_COMMANDS)}")

    for section_name in ("dataset", "model", "training", "contrastive", "outputs", "inference", "downstream"):
        section = config.get(section_name)
        if not isinstance(section, dict):
            raise ConfigError(f"{section_name} section must be a mapping")
        extra_section_keys = sorted(set(section) - VALID_SECTION_KEYS[section_name])
        if extra_section_keys:
            raise ConfigError(f"{section_name} section has unknown keys: {', '.join(extra_section_keys)}")

    dataset = config["dataset"]
    model = config["model"]
    training = config["training"]
    contrastive = config["contrastive"]
    outputs = config["outputs"]
    inference = config["inference"]
    downstream = config["downstream"]

    for field_name, value in (
        ("dataset.csv", dataset["csv"]),
        ("dataset.dataset_root", dataset["dataset_root"]),
        ("dataset.geometry", dataset["geometry"]),
        ("dataset.results", dataset["results"]),
        ("training.resume_from", training["resume_from"]),
        ("outputs.checkpoint", outputs["checkpoint"]),
        ("outputs.metrics", outputs["metrics"]),
        ("outputs.embeddings", outputs["embeddings"]),
        ("inference.checkpoint", inference["checkpoint"]),
        ("inference.output", inference["output"]),
    ):
        _validate_optional_string(value, field_name)

    _validate_bool(dataset["use_results"], "dataset.use_results")

    dataset["limit"] = _ensure_positive_int(dataset["limit"], "dataset.limit")
    dataset["limit_per_shard"] = _ensure_positive_int(dataset["limit_per_shard"], "dataset.limit_per_shard")
    model["hidden_dim"] = _ensure_positive_int(model["hidden_dim"], "model.hidden_dim")
    model["message_passing_steps"] = _ensure_positive_int(model["message_passing_steps"], "model.message_passing_steps")
    training["epochs"] = _ensure_positive_int(training["epochs"], "training.epochs")
    training["learning_rate"] = _ensure_positive_float(training["learning_rate"], "training.learning_rate")
    training["aux_weight"] = _ensure_non_negative_float(training["aux_weight"], "training.aux_weight")
    downstream["epochs"] = _ensure_positive_int(downstream["epochs"], "downstream.epochs")
    downstream["learning_rate"] = _ensure_positive_float(downstream["learning_rate"], "downstream.learning_rate")
    downstream["seed"] = _coerce_int(downstream["seed"], "downstream.seed")
    contrastive["batch_size"] = _ensure_positive_int(contrastive["batch_size"], "contrastive.batch_size")
    contrastive["hidden_dim_3d"] = _ensure_positive_int(contrastive["hidden_dim_3d"], "contrastive.hidden_dim_3d")
    contrastive["num_rbf"] = _ensure_positive_int(contrastive["num_rbf"], "contrastive.num_rbf")
    contrastive["message_passing_steps_3d"] = _ensure_positive_int(
        contrastive["message_passing_steps_3d"], "contrastive.message_passing_steps_3d"
    )
    contrastive["seed"] = _coerce_int(contrastive["seed"], "contrastive.seed")
    contrastive["cutoff"] = _ensure_positive_float(contrastive["cutoff"], "contrastive.cutoff")
    contrastive["temperature"] = _ensure_positive_float(contrastive["temperature"], "contrastive.temperature")
    contrastive["supervised_weight"] = _ensure_non_negative_float(
        contrastive["supervised_weight"], "contrastive.supervised_weight"
    )
    contrastive["contrastive_weight"] = _ensure_non_negative_float(
        contrastive["contrastive_weight"], "contrastive.contrastive_weight"
    )
    contrastive["teacher_weight"] = _ensure_non_negative_float(
        contrastive["teacher_weight"], "contrastive.teacher_weight"
    )
    contrastive["energy_temperature"] = _ensure_positive_float(
        contrastive["energy_temperature"], "contrastive.energy_temperature"
    )
    if contrastive["conformer_pool_mode"] not in {"mean", "weighted", "energy"}:
        raise ConfigError("contrastive.conformer_pool_mode must be one of mean, weighted, energy")

    subset_ids = _ensure_sequence(dataset["subset_ids"], "dataset.subset_ids")
    dataset["subset_ids"] = [_coerce_int(value, "dataset.subset_ids[]") for value in subset_ids]
    fractions = _ensure_sequence(downstream["fractions"], "downstream.fractions")
    if not fractions:
        raise ConfigError("downstream.fractions must not be empty")
    downstream["fractions"] = []
    for value in fractions:
        fraction = _coerce_float(value, "downstream.fractions[]")
        if not (0.0 < fraction <= 1.0):
            raise ConfigError("downstream.fractions[] must be in the interval (0, 1]")
        downstream["fractions"].append(fraction)

    dataset = config["dataset"]
    has_csv = bool(dataset.get("csv"))
    has_dataset_root = bool(dataset.get("dataset_root"))
    if command in {"train", "pretrain", "contrastive-pretrain"}:
        if has_csv == has_dataset_root:
            raise ConfigError("training configs require exactly one of dataset.csv or dataset.dataset_root")
        if has_dataset_root and not dataset.get("subset_ids"):
            raise ConfigError("dataset.subset_ids is required when dataset.dataset_root is set")

    downstream = config["downstream"]
    if downstream.get("kind") not in VALID_DOWNSTREAM_KINDS:
        raise ConfigError(f"downstream.kind must be one of {sorted(VALID_DOWNSTREAM_KINDS)}")

    outputs = config["outputs"]
    inference = config["inference"]
    if command in {"train", "pretrain", "contrastive-pretrain"} and not outputs.get("checkpoint"):
        raise ConfigError("outputs.checkpoint is required for training commands")
    if command == "export-embeddings":
        if not inference.get("checkpoint"):
            raise ConfigError("inference.checkpoint is required for export-embeddings")
        if not inference.get("output"):
            raise ConfigError("inference.output is required for export-embeddings")
    if command in {"eval", "downstream"} and not inference.get("checkpoint"):
        raise ConfigError("inference.checkpoint is required")


def _coerce_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be an integer")
    if isinstance(value, float):
        if not value.is_integer():
            raise ConfigError(f"{field_name} must be an integer")
        return int(value)
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be an integer") from exc
    if isinstance(value, str) and str(coerced) != value.strip():
        raise ConfigError(f"{field_name} must be an integer")
    return coerced


def _coerce_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a number")
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a number") from exc
    if not math.isfinite(coerced):
        raise ConfigError(f"{field_name} must be a finite number")
    return coerced


def _ensure_sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"{field_name} must be a list")
    return value


def _validate_optional_string(value: Any, field_name: str) -> None:
    if value is not None:
        if not isinstance(value, str):
            raise ConfigError(f"{field_name} must be a string")
        if not value.strip():
            raise ConfigError(f"{field_name} must not be blank")


def _validate_bool(value: Any, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a boolean")


def _subset_ids_for_cli(subset_ids: Sequence[int]) -> str | None:
    if not subset_ids:
        return None
    values = _ensure_sequence(subset_ids, "dataset.subset_ids")
    return ",".join(str(_coerce_int(value, "dataset.subset_ids[]")) for value in values)


def _ensure_positive_int(value: Any, field_name: str) -> int:
    coerced = _coerce_int(value, field_name)
    if coerced <= 0:
        raise ConfigError(f"{field_name} must be greater than zero")
    return coerced


def _ensure_positive_float(value: Any, field_name: str) -> float:
    coerced = _coerce_float(value, field_name)
    if coerced <= 0.0:
        raise ConfigError(f"{field_name} must be greater than zero")
    return coerced


def _ensure_non_negative_float(value: Any, field_name: str) -> float:
    coerced = _coerce_float(value, field_name)
    if coerced < 0.0:
        raise ConfigError(f"{field_name} must be non-negative")
    return coerced


def _fraction_list_for_cli(values: Sequence[Any]) -> str:
    items = _ensure_sequence(values, "downstream.fractions")
    return ",".join(str(_coerce_float(value, "downstream.fractions[]")) for value in items)


def config_to_namespace(config: ConfigDict) -> Namespace:
    command = config["command"]
    dataset = config["dataset"]
    model = config["model"]
    training = config["training"]
    outputs = config["outputs"]
    inference = config["inference"]
    downstream = config["downstream"]

    if command in {"train", "pretrain", "contrastive-pretrain"}:
        values: ConfigDict = {
            "command": command,
            "csv": dataset["csv"],
            "dataset_root": dataset["dataset_root"],
            "subset_ids": _subset_ids_for_cli(dataset["subset_ids"]),
            "geometry": dataset["geometry"],
            "results": dataset["results"],
            "use_results": bool(dataset["use_results"]),
            "limit": _coerce_int(dataset["limit"], "dataset.limit"),
            "limit_per_shard": _coerce_int(dataset["limit_per_shard"], "dataset.limit_per_shard"),
            "epochs": _coerce_int(training["epochs"], "training.epochs"),
            "hidden_dim": _coerce_int(model["hidden_dim"], "model.hidden_dim"),
            "message_passing_steps": _coerce_int(model["message_passing_steps"], "model.message_passing_steps"),
            "learning_rate": _coerce_float(training["learning_rate"], "training.learning_rate"),
            "metrics_output": outputs["metrics"],
            "output": outputs["checkpoint"],
            "resume_from": training["resume_from"],
        }
        if command == "contrastive-pretrain":
            contrastive = config["contrastive"]
            values.update(
                {
                    "batch_size": _coerce_int(contrastive["batch_size"], "contrastive.batch_size"),
                    "supervised_weight": _coerce_float(contrastive["supervised_weight"], "contrastive.supervised_weight"),
                    "contrastive_weight": _coerce_float(contrastive["contrastive_weight"], "contrastive.contrastive_weight"),
                    "temperature": _coerce_float(contrastive["temperature"], "contrastive.temperature"),
                    "teacher_weight": _coerce_float(contrastive["teacher_weight"], "contrastive.teacher_weight"),
                    "energy_temperature": _coerce_float(contrastive["energy_temperature"], "contrastive.energy_temperature"),
                    "hidden_dim_3d": _coerce_int(contrastive["hidden_dim_3d"], "contrastive.hidden_dim_3d"),
                    "num_rbf": _coerce_int(contrastive["num_rbf"], "contrastive.num_rbf"),
                    "cutoff": _coerce_float(contrastive["cutoff"], "contrastive.cutoff"),
                    "message_passing_steps_3d": _coerce_int(contrastive["message_passing_steps_3d"], "contrastive.message_passing_steps_3d"),
                    "conformer_pool_mode": contrastive["conformer_pool_mode"],
                    "seed": _coerce_int(contrastive["seed"], "contrastive.seed"),
                }
            )
        if command == "pretrain":
            values["aux_weight"] = _coerce_float(training["aux_weight"], "training.aux_weight")
        return Namespace(**values)

    if command == "export-embeddings":
        return Namespace(command=command, checkpoint=inference["checkpoint"], output=inference["output"])

    if command == "eval":
        return Namespace(command=command, checkpoint=inference["checkpoint"], output=inference["output"])

    if command == "downstream":
        return Namespace(
            command=command,
            checkpoint=inference["checkpoint"],
            kind=downstream["kind"],
            output=inference["output"],
            epochs=_coerce_int(downstream["epochs"], "downstream.epochs"),
            learning_rate=_coerce_float(downstream["learning_rate"], "downstream.learning_rate"),
            fractions=_fraction_list_for_cli(downstream["fractions"]),
            seed=_coerce_int(downstream["seed"], "downstream.seed"),
        )

    raise ConfigError(f"Unsupported command: {command}")
