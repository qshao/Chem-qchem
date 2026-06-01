# YAML Config Training And Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add YAML configuration files that control `qchem_gnn` training, pretraining, embedding export, evaluation, and downstream inference workflows.

**Architecture:** Add a focused config module that loads YAML into validated dictionaries matching the current CLI concepts: dataset, model, training, output, inference, and downstream. Keep the existing argparse flags working, but let `--config` provide defaults and let explicit CLI flags override YAML values. Persist the resolved config into checkpoints and metrics so every run is reproducible.

**Tech Stack:** Python 3.13, PyTorch, RDKit, pandas, numpy, PyYAML via `yaml.safe_load`, pytest.

---

## File Structure

- Create `qchem_gnn/config.py`
  - Load YAML.
  - Validate top-level sections.
  - Merge YAML config with argparse overrides.
  - Convert config sections into the existing CLI argument namespace.
- Create `configs/minimal_pretrain.yaml`
  - Minimal example for proxy-target pretraining on one shard.
- Create `configs/minimal_train.yaml`
  - Minimal example for regular training on one shard.
- Create `configs/export_embeddings.yaml`
  - Minimal example for checkpoint-based embedding export.
- Create `configs/eval.yaml`
  - Minimal example for checkpoint evaluation.
- Modify `qchem_gnn/cli.py`
  - Add `--config` to `train`, `pretrain`, `export-embeddings`, `eval`, and `downstream`.
  - Resolve YAML values before calling existing `run_*` functions.
  - Store the resolved config in checkpoint `run_metadata`.
- Modify `qchem_gnn/checkpoint.py`
  - Add optional `resolved_config` to checkpoint state.
- Modify `qchem_gnn/__init__.py`
  - Export config helpers if they are useful programmatically.
- Test `tests/test_config.py`
  - Unit tests for YAML loading, validation, and override behavior.
- Modify `tests/test_cli.py`
  - CLI integration tests for config-driven train, pretrain, export, eval, and downstream.

## Config Schema

Use this schema for YAML files:

```yaml
command: pretrain

dataset:
  csv: zinc-250k/subsets/subset_000.csv
  dataset_root:
  subset_ids: []
  geometry: zinc-250k/geometries/coords_000.pkl
  results:
  use_results: false
  limit: 16
  limit_per_shard: 16

model:
  hidden_dim: 32
  message_passing_steps: 2

training:
  epochs: 200
  learning_rate: 0.02
  aux_weight: 0.1
  resume_from:

outputs:
  checkpoint: runs/pretrain.pt
  metrics: runs/pretrain_metrics.json
  embeddings:

inference:
  checkpoint:
  output:

downstream:
  kind: fine-tune
  epochs: 50
  learning_rate: 0.01
  fractions: [0.01, 0.05, 0.1, 1.0]
  seed: 0
```

Rules:

- `command` must be one of `train`, `pretrain`, `export-embeddings`, `eval`, or `downstream`.
- `dataset.csv` plus `dataset.geometry` is the single-shard path.
- `dataset.dataset_root` plus `dataset.subset_ids` is the shard-range path.
- For `train` and `pretrain`, `outputs.checkpoint` is required.
- For `export-embeddings`, `inference.checkpoint` and `inference.output` are required.
- For `eval`, `inference.checkpoint` is required and `inference.output` is optional.
- For `downstream`, `inference.checkpoint` is required and `downstream.kind` controls the evaluation mode.
- Explicit CLI flags override YAML values.

---

### Task 1: Add Config Loader And Validation

**Files:**
- Create: `qchem_gnn/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for loading and validation**

Add `tests/test_config.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from qchem_gnn.config import ConfigError, load_yaml_config, resolve_config


def test_load_yaml_config_reads_nested_sections(tmp_path: Path):
    config_path = tmp_path / "pretrain.yaml"
    config_path.write_text(
        """
command: pretrain
dataset:
  csv: zinc-250k/subsets/subset_000.csv
  geometry: zinc-250k/geometries/coords_000.pkl
  use_results: false
  limit: 4
model:
  hidden_dim: 16
  message_passing_steps: 1
training:
  epochs: 3
  learning_rate: 0.01
  aux_weight: 0.2
outputs:
  checkpoint: runs/test.pt
  metrics: runs/test.json
""",
        encoding="utf-8",
    )

    config = load_yaml_config(config_path)

    assert config["command"] == "pretrain"
    assert config["dataset"]["limit"] == 4
    assert config["model"]["hidden_dim"] == 16
    assert config["training"]["aux_weight"] == 0.2
    assert config["outputs"]["checkpoint"] == "runs/test.pt"


def test_resolve_config_rejects_unknown_command(tmp_path: Path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("command: unknown\noutputs:\n  checkpoint: runs/out.pt\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="command"):
        resolve_config(load_yaml_config(config_path))


def test_resolve_config_applies_defaults_and_cli_overrides(tmp_path: Path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
command: train
dataset:
  csv: subset_000.csv
  geometry: coords_000.pkl
outputs:
  checkpoint: runs/from_yaml.pt
model:
  hidden_dim: 16
training:
  epochs: 2
""",
        encoding="utf-8",
    )

    resolved = resolve_config(
        load_yaml_config(config_path),
        overrides={
            "training": {"epochs": 5},
            "outputs": {"checkpoint": "runs/from_cli.pt"},
        },
    )

    assert resolved["dataset"]["limit"] == 16
    assert resolved["dataset"]["limit_per_shard"] == 16
    assert resolved["model"]["hidden_dim"] == 16
    assert resolved["model"]["message_passing_steps"] == 2
    assert resolved["training"]["epochs"] == 5
    assert resolved["outputs"]["checkpoint"] == "runs/from_cli.pt"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_config.py -q
```

Expected: import failure because `qchem_gnn.config` does not exist.

- [ ] **Step 3: Implement config loader**

Create `qchem_gnn/config.py`:

```python
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


VALID_COMMANDS = {"train", "pretrain", "export-embeddings", "eval", "downstream"}

DEFAULT_CONFIG: dict[str, Any] = {
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


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("PyYAML is required for --config support. Install pyyaml.") from exc

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ConfigError("YAML config must be a mapping at the top level")
    return loaded


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _drop_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none_values(inner) for key, inner in value.items() if inner is not None}
    return value


def resolve_config(config: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _deep_merge(DEFAULT_CONFIG, _drop_none_values(config))
    if overrides:
        resolved = _deep_merge(resolved, _drop_none_values(overrides))
    _validate_config(resolved)
    return resolved


def _validate_config(config: dict[str, Any]) -> None:
    command = config.get("command")
    if command not in VALID_COMMANDS:
        raise ConfigError(f"command must be one of {sorted(VALID_COMMANDS)}")

    dataset = config["dataset"]
    has_single_shard = bool(dataset.get("csv"))
    has_root = bool(dataset.get("dataset_root"))
    if command in {"train", "pretrain"} and has_single_shard == has_root:
        raise ConfigError("training configs require exactly one of dataset.csv or dataset.dataset_root")
    if has_root and not dataset.get("subset_ids"):
        raise ConfigError("dataset.subset_ids is required when dataset.dataset_root is set")

    outputs = config["outputs"]
    inference = config["inference"]
    if command in {"train", "pretrain"} and not outputs.get("checkpoint"):
        raise ConfigError("outputs.checkpoint is required for training commands")
    if command == "export-embeddings":
        if not inference.get("checkpoint"):
            raise ConfigError("inference.checkpoint is required for export-embeddings")
        if not inference.get("output") and not outputs.get("embeddings"):
            raise ConfigError("inference.output or outputs.embeddings is required for export-embeddings")
    if command in {"eval", "downstream"} and not inference.get("checkpoint"):
        raise ConfigError("inference.checkpoint is required")
```

- [ ] **Step 4: Run config tests**

Run:

```bash
python -m pytest tests/test_config.py -q
```

Expected: config tests pass.

---

### Task 2: Convert Resolved Config Into CLI Arguments

**Files:**
- Modify: `qchem_gnn/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Add failing tests for argparse namespace conversion**

Append to `tests/test_config.py`:

```python
from qchem_gnn.config import config_to_namespace


def test_config_to_namespace_maps_pretrain_fields():
    resolved = resolve_config(
        {
            "command": "pretrain",
            "dataset": {
                "csv": "subset_000.csv",
                "geometry": "coords_000.pkl",
                "use_results": True,
                "limit": 8,
            },
            "model": {"hidden_dim": 24, "message_passing_steps": 3},
            "training": {"epochs": 7, "learning_rate": 0.03, "aux_weight": 0.4},
            "outputs": {"checkpoint": "runs/pretrain.pt", "metrics": "runs/pretrain.json"},
        }
    )

    args = config_to_namespace(resolved)

    assert args.command == "pretrain"
    assert args.csv == "subset_000.csv"
    assert args.geometry == "coords_000.pkl"
    assert args.use_results is True
    assert args.limit == 8
    assert args.hidden_dim == 24
    assert args.message_passing_steps == 3
    assert args.epochs == 7
    assert args.learning_rate == 0.03
    assert args.aux_weight == 0.4
    assert args.output == "runs/pretrain.pt"
    assert args.metrics_output == "runs/pretrain.json"


def test_config_to_namespace_maps_export_fields():
    resolved = resolve_config(
        {
            "command": "export-embeddings",
            "inference": {"checkpoint": "runs/model.pt", "output": "runs/embeddings.pt"},
        }
    )

    args = config_to_namespace(resolved)

    assert args.command == "export-embeddings"
    assert args.checkpoint == "runs/model.pt"
    assert args.output == "runs/embeddings.pt"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_config.py -q
```

Expected: import error for `config_to_namespace`.

- [ ] **Step 3: Implement namespace conversion**

Append to `qchem_gnn/config.py`:

```python
from argparse import Namespace
from collections.abc import Sequence


def _subset_ids_for_cli(subset_ids: Sequence[int]) -> str | None:
    if not subset_ids:
        return None
    return ",".join(str(int(value)) for value in subset_ids)


def config_to_namespace(config: dict[str, Any]) -> Namespace:
    command = config["command"]
    dataset = config["dataset"]
    model = config["model"]
    training = config["training"]
    outputs = config["outputs"]
    inference = config["inference"]
    downstream = config["downstream"]

    common = {
        "command": command,
        "csv": dataset["csv"],
        "dataset_root": dataset["dataset_root"],
        "subset_ids": _subset_ids_for_cli(dataset["subset_ids"]),
        "geometry": dataset["geometry"],
        "results": dataset["results"],
        "use_results": bool(dataset["use_results"]),
        "limit": int(dataset["limit"]),
        "limit_per_shard": int(dataset["limit_per_shard"]),
        "epochs": int(training["epochs"]),
        "hidden_dim": int(model["hidden_dim"]),
        "message_passing_steps": int(model["message_passing_steps"]),
        "learning_rate": float(training["learning_rate"]),
        "metrics_output": outputs["metrics"],
        "output": outputs["checkpoint"],
        "resume_from": training["resume_from"],
    }

    if command == "pretrain":
        common["aux_weight"] = float(training["aux_weight"])
    elif command == "export-embeddings":
        common = {
            "command": command,
            "checkpoint": inference["checkpoint"],
            "output": inference["output"] or outputs["embeddings"],
        }
    elif command == "eval":
        common = {
            "command": command,
            "checkpoint": inference["checkpoint"],
            "output": inference["output"],
        }
    elif command == "downstream":
        common = {
            "command": command,
            "checkpoint": inference["checkpoint"],
            "kind": downstream["kind"],
            "output": inference["output"],
            "epochs": int(downstream["epochs"]),
            "learning_rate": float(downstream["learning_rate"]),
            "fractions": ",".join(str(float(value)) for value in downstream["fractions"]),
            "seed": int(downstream["seed"]),
        }

    return Namespace(**common)
```

- [ ] **Step 4: Run config tests**

Run:

```bash
python -m pytest tests/test_config.py -q
```

Expected: all config tests pass.

---

### Task 3: Add `--config` To CLI Commands

**Files:**
- Modify: `qchem_gnn/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Add failing CLI config tests**

Append to `tests/test_cli.py`:

```python
def test_train_cli_accepts_yaml_config(tmp_path):
    csv_path = tmp_path / "subset_000.csv"
    geo_path = tmp_path / "coords_000.pkl"
    checkpoint_path = tmp_path / "configured.pt"
    metrics_path = tmp_path / "configured_metrics.json"
    config_path = tmp_path / "train.yaml"

    pd.DataFrame(
        [
            {"smiles": "C", "logP": 0.1, "qed": 0.2, "SAS": 0.3},
            {"smiles": "CC", "logP": 0.4, "qed": 0.5, "SAS": 0.6},
        ]
    ).to_csv(csv_path, index=False)
    with geo_path.open("wb") as handle:
        pickle.dump(
            {
                "subset_0_idx_0": {
                    "smiles": "C",
                    "charge": 0,
                    "symbols": ["C", "H", "H", "H", "H"],
                    "atomic_nums": [6, 1, 1, 1, 1],
                    "conformers": [np.zeros((5, 3), dtype=np.float32)],
                },
                "subset_0_idx_1": {
                    "smiles": "CC",
                    "charge": 0,
                    "symbols": ["C", "C"] + ["H"] * 6,
                    "atomic_nums": [6, 6] + [1] * 6,
                    "conformers": [np.zeros((8, 3), dtype=np.float32)],
                },
            },
            handle,
        )
    config_path.write_text(
        f"""
command: train
dataset:
  csv: {csv_path}
  geometry: {geo_path}
  limit: 2
model:
  hidden_dim: 32
  message_passing_steps: 2
training:
  epochs: 3
  learning_rate: 0.02
outputs:
  checkpoint: {checkpoint_path}
  metrics: {metrics_path}
""",
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_path)])

    assert exit_code == 0
    payload = torch.load(checkpoint_path, map_location="cpu")
    assert payload["run_metadata"]["resolved_config"]["training"]["epochs"] == 3
    assert metrics_path.exists()


def test_config_cli_flag_overrides_yaml_value(tmp_path):
    csv_path = tmp_path / "subset_000.csv"
    geo_path = tmp_path / "coords_000.pkl"
    yaml_checkpoint = tmp_path / "yaml.pt"
    cli_checkpoint = tmp_path / "cli.pt"
    config_path = tmp_path / "train.yaml"

    pd.DataFrame([{"smiles": "C", "logP": 0.1, "qed": 0.2, "SAS": 0.3}]).to_csv(csv_path, index=False)
    with geo_path.open("wb") as handle:
        pickle.dump(
            {
                "subset_0_idx_0": {
                    "smiles": "C",
                    "charge": 0,
                    "symbols": ["C", "H", "H", "H", "H"],
                    "atomic_nums": [6, 1, 1, 1, 1],
                    "conformers": [np.zeros((5, 3), dtype=np.float32)],
                }
            },
            handle,
        )
    config_path.write_text(
        f"""
command: train
dataset:
  csv: {csv_path}
  geometry: {geo_path}
training:
  epochs: 20
outputs:
  checkpoint: {yaml_checkpoint}
""",
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_path), "--epochs", "1", "--output", str(cli_checkpoint)])

    assert exit_code == 0
    assert cli_checkpoint.exists()
    assert not yaml_checkpoint.exists()
    payload = torch.load(cli_checkpoint, map_location="cpu")
    assert payload["run_metadata"]["resolved_config"]["training"]["epochs"] == 1
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_cli.py::test_train_cli_accepts_yaml_config tests/test_cli.py::test_config_cli_flag_overrides_yaml_value -q
```

Expected: argparse failure because `--config` is not accepted.

- [ ] **Step 3: Add config argument and resolver hook**

Modify `qchem_gnn/cli.py`:

```python
from .config import config_to_namespace, load_yaml_config, resolve_config
```

Add `--config` to `train`, `pretrain`, `export-embeddings`, `eval`, and `downstream` parsers:

```python
train.add_argument("--config", default=None, help="YAML config path")
pretrain.add_argument("--config", default=None, help="YAML config path")
export.add_argument("--config", default=None, help="YAML config path")
eval_cmd.add_argument("--config", default=None, help="YAML config path")
downstream.add_argument("--config", default=None, help="YAML config path")
```

Add helper functions near `_dataset_kwargs_from_args`:

```python
def _provided_arg_map(argv: list[str]) -> dict[str, object]:
    pairs = {
        "--csv": ("dataset", "csv"),
        "--dataset-root": ("dataset", "dataset_root"),
        "--subset-ids": ("dataset", "subset_ids"),
        "--geometry": ("dataset", "geometry"),
        "--results": ("dataset", "results"),
        "--limit": ("dataset", "limit"),
        "--limit-per-shard": ("dataset", "limit_per_shard"),
        "--epochs": ("training", "epochs"),
        "--learning-rate": ("training", "learning_rate"),
        "--aux-weight": ("training", "aux_weight"),
        "--resume-from": ("training", "resume_from"),
        "--hidden-dim": ("model", "hidden_dim"),
        "--message-passing-steps": ("model", "message_passing_steps"),
        "--metrics-output": ("outputs", "metrics"),
        "--output": ("outputs", "checkpoint"),
        "--checkpoint": ("inference", "checkpoint"),
        "--kind": ("downstream", "kind"),
        "--fractions": ("downstream", "fractions"),
        "--seed": ("downstream", "seed"),
    }
    return pairs


def _set_nested(overrides: dict[str, object], section: str, key: str, value: object) -> None:
    overrides.setdefault(section, {})[key] = value


def _config_overrides_from_args(args, argv: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {"command": args.command}
    flags = _provided_arg_map(argv)
    tokens = set(argv)
    for flag, (section, key) in flags.items():
        if flag in tokens:
            attr = flag.removeprefix("--").replace("-", "_")
            value = getattr(args, attr)
            if flag == "--output" and args.command == "export-embeddings":
                _set_nested(overrides, "inference", "output", value)
            else:
                _set_nested(overrides, section, key, value)
    if "--use-results" in tokens:
        _set_nested(overrides, "dataset", "use_results", True)
    return overrides


def _args_from_config_if_present(args, argv: list[str]):
    config_path = getattr(args, "config", None)
    if not config_path:
        return args, None
    config = load_yaml_config(config_path)
    resolved = resolve_config(config, overrides=_config_overrides_from_args(args, argv))
    return config_to_namespace(resolved), resolved
```

Modify `main`:

```python
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(argv or [])
    args = parser.parse_args(argv)
    args, resolved_config = _args_from_config_if_present(args, raw_argv)
    if resolved_config is not None:
        setattr(args, "resolved_config", resolved_config)
```

- [ ] **Step 4: Persist resolved config in run metadata**

In `run_train` and `run_pretrain`, add:

```python
resolved_config = getattr(args, "resolved_config", None)
```

Then add this field to `run_metadata` dictionaries:

```python
"resolved_config": resolved_config,
```

Do not add `resolved_config` when it is `None`.

- [ ] **Step 5: Run config CLI tests**

Run:

```bash
python -m pytest tests/test_config.py tests/test_cli.py::test_train_cli_accepts_yaml_config tests/test_cli.py::test_config_cli_flag_overrides_yaml_value -q
```

Expected: tests pass.

---

### Task 4: Add Config Support For Pretraining And Inference Commands

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `qchem_gnn/cli.py`

- [ ] **Step 1: Add failing tests for pretrain, export, eval, and downstream configs**

Append to `tests/test_cli.py`:

```python
def test_pretrain_cli_accepts_yaml_config(tmp_path):
    csv_path = tmp_path / "subset_000.csv"
    geo_path = tmp_path / "coords_000.pkl"
    checkpoint_path = tmp_path / "pretrain.pt"
    config_path = tmp_path / "pretrain.yaml"

    pd.DataFrame([{"smiles": "C", "logP": 0.1, "qed": 0.2, "SAS": 0.3}]).to_csv(csv_path, index=False)
    with geo_path.open("wb") as handle:
        pickle.dump(
            {
                "subset_0_idx_0": {
                    "smiles": "C",
                    "charge": 0,
                    "symbols": ["C", "H", "H", "H", "H"],
                    "atomic_nums": [6, 1, 1, 1, 1],
                    "conformers": [np.zeros((5, 3), dtype=np.float32)],
                }
            },
            handle,
        )
    config_path.write_text(
        f"""
command: pretrain
dataset:
  csv: {csv_path}
  geometry: {geo_path}
model:
  hidden_dim: 32
training:
  epochs: 1
  aux_weight: 0.1
outputs:
  checkpoint: {checkpoint_path}
""",
        encoding="utf-8",
    )

    assert main(["pretrain", "--config", str(config_path)]) == 0
    payload = torch.load(checkpoint_path, map_location="cpu")
    assert payload["aux_head_state_dict"]
    assert payload["run_metadata"]["resolved_config"]["command"] == "pretrain"


def test_export_embeddings_cli_accepts_yaml_config(tmp_path):
    checkpoint_path = _write_minimal_checkpoint_for_cli(tmp_path)
    export_path = tmp_path / "embeddings.pt"
    config_path = tmp_path / "export.yaml"
    config_path.write_text(
        f"""
command: export-embeddings
inference:
  checkpoint: {checkpoint_path}
  output: {export_path}
""",
        encoding="utf-8",
    )

    assert main(["export-embeddings", "--config", str(config_path)]) == 0
    exported = torch.load(export_path, map_location="cpu")
    assert exported["embeddings"].shape[0] == 1


def test_eval_cli_accepts_yaml_config(tmp_path):
    checkpoint_path = _write_minimal_checkpoint_for_cli(tmp_path)
    metrics_path = tmp_path / "eval.pt"
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        f"""
command: eval
inference:
  checkpoint: {checkpoint_path}
  output: {metrics_path}
""",
        encoding="utf-8",
    )

    assert main(["eval", "--config", str(config_path)]) == 0
    metrics = torch.load(metrics_path, map_location="cpu")
    assert metrics["num_examples"] == 1
```

Before these tests, add a helper in `tests/test_cli.py`:

```python
def _write_minimal_checkpoint_for_cli(tmp_path: Path) -> Path:
    csv_path = tmp_path / "subset_000.csv"
    geo_path = tmp_path / "coords_000.pkl"
    checkpoint_path = tmp_path / "checkpoint.pt"
    pd.DataFrame([{"smiles": "C", "logP": 0.1, "qed": 0.2, "SAS": 0.3}]).to_csv(csv_path, index=False)
    with geo_path.open("wb") as handle:
        pickle.dump(
            {
                "subset_0_idx_0": {
                    "smiles": "C",
                    "charge": 0,
                    "symbols": ["C", "H", "H", "H", "H"],
                    "atomic_nums": [6, 1, 1, 1, 1],
                    "conformers": [np.zeros((5, 3), dtype=np.float32)],
                }
            },
            handle,
        )
    assert main(
        [
            "train",
            "--csv",
            str(csv_path),
            "--geometry",
            str(geo_path),
            "--limit",
            "1",
            "--epochs",
            "1",
            "--output",
            str(checkpoint_path),
        ]
    ) == 0
    return checkpoint_path
```

- [ ] **Step 2: Run tests to verify current failures**

Run:

```bash
python -m pytest tests/test_cli.py::test_pretrain_cli_accepts_yaml_config tests/test_cli.py::test_export_embeddings_cli_accepts_yaml_config tests/test_cli.py::test_eval_cli_accepts_yaml_config -q
```

Expected: failures until config conversion covers all commands.

- [ ] **Step 3: Ensure command conversion handles inference output correctly**

If Task 3 was implemented as written, this task should only require small fixes in `config_to_namespace` or `_config_overrides_from_args`. Verify these mappings:

```python
export-embeddings:
  inference.checkpoint -> args.checkpoint
  inference.output -> args.output

eval:
  inference.checkpoint -> args.checkpoint
  inference.output -> args.output

downstream:
  inference.checkpoint -> args.checkpoint
  inference.output -> args.output
  downstream.kind -> args.kind
```

- [ ] **Step 4: Run CLI config tests**

Run:

```bash
python -m pytest tests/test_config.py tests/test_cli.py -q
```

Expected: all config and CLI tests pass.

---

### Task 5: Add Example YAML Files

**Files:**
- Create: `configs/minimal_train.yaml`
- Create: `configs/minimal_pretrain.yaml`
- Create: `configs/export_embeddings.yaml`
- Create: `configs/eval.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Add failing test that example configs load**

Append to `tests/test_config.py`:

```python
def test_example_configs_load():
    for path in [
        Path("configs/minimal_train.yaml"),
        Path("configs/minimal_pretrain.yaml"),
        Path("configs/export_embeddings.yaml"),
        Path("configs/eval.yaml"),
    ]:
        resolved = resolve_config(load_yaml_config(path))
        assert resolved["command"]
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python -m pytest tests/test_config.py::test_example_configs_load -q
```

Expected: failure because config files do not exist.

- [ ] **Step 3: Create `configs/minimal_train.yaml`**

```yaml
command: train

dataset:
  csv: zinc-250k/subsets/subset_000.csv
  dataset_root:
  subset_ids: []
  geometry: zinc-250k/geometries/coords_000.pkl
  results:
  use_results: false
  limit: 16
  limit_per_shard: 16

model:
  hidden_dim: 32
  message_passing_steps: 2

training:
  epochs: 200
  learning_rate: 0.02
  resume_from:

outputs:
  checkpoint: runs/minimal_train.pt
  metrics: runs/minimal_train_metrics.json
```

- [ ] **Step 4: Create `configs/minimal_pretrain.yaml`**

```yaml
command: pretrain

dataset:
  csv: zinc-250k/subsets/subset_000.csv
  dataset_root:
  subset_ids: []
  geometry: zinc-250k/geometries/coords_000.pkl
  results:
  use_results: false
  limit: 16
  limit_per_shard: 16

model:
  hidden_dim: 32
  message_passing_steps: 2

training:
  epochs: 200
  learning_rate: 0.02
  aux_weight: 0.1

outputs:
  checkpoint: runs/minimal_pretrain.pt
  metrics: runs/minimal_pretrain_metrics.json
```

- [ ] **Step 5: Create `configs/export_embeddings.yaml`**

```yaml
command: export-embeddings

inference:
  checkpoint: runs/minimal_pretrain.pt
  output: runs/minimal_embeddings.pt
```

- [ ] **Step 6: Create `configs/eval.yaml`**

```yaml
command: eval

inference:
  checkpoint: runs/minimal_pretrain.pt
  output: runs/minimal_eval.pt
```

- [ ] **Step 7: Run config example test**

Run:

```bash
python -m pytest tests/test_config.py::test_example_configs_load -q
```

Expected: test passes.

---

### Task 6: Persist Resolved Config In Checkpoints And Metrics

**Files:**
- Modify: `qchem_gnn/checkpoint.py`
- Modify: `qchem_gnn/cli.py`
- Test: `tests/test_checkpoint.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Add failing checkpoint test for `resolved_config`**

Append to `tests/test_checkpoint.py`:

```python
def test_checkpoint_state_can_store_resolved_config():
    model = MolecularQuantumGNN(
        atom_vocab_size=128,
        bond_vocab_size=8,
        hidden_dim=16,
        num_message_passing_steps=1,
        graph_targets=2,
    )
    state = build_checkpoint_state(
        loss_history=[],
        embeddings=torch.zeros(1, 16),
        model_state_dict=model.state_dict(),
        optimizer_state_dict={},
        epoch=0,
        global_step=0,
        target_normalization={},
        dataset_config={"csv": "subset_000.csv", "geometry": "coords_000.pkl"},
        split_metadata={},
        model_config={
            "atom_vocab_size": 128,
            "bond_vocab_size": 8,
            "hidden_dim": 16,
            "num_message_passing_steps": 1,
            "graph_targets": 2,
        },
        run_metadata={"num_examples": 1},
        resolved_config={"command": "train"},
    )

    assert state["resolved_config"]["command"] == "train"
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python -m pytest tests/test_checkpoint.py::test_checkpoint_state_can_store_resolved_config -q
```

Expected: `build_checkpoint_state` does not accept `resolved_config`.

- [ ] **Step 3: Add optional `resolved_config` to checkpoint builder**

Modify `qchem_gnn/checkpoint.py`:

```python
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
    resolved_config=None,
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
    if resolved_config is not None:
        state["resolved_config"] = resolved_config
    return state
```

- [ ] **Step 4: Pass resolved config from CLI to checkpoint builder**

In `run_train` and `run_pretrain`:

```python
resolved_config = getattr(args, "resolved_config", None)
```

Pass:

```python
resolved_config=resolved_config,
```

Also include it in metrics:

```python
if resolved_config is not None:
    metrics["resolved_config"] = resolved_config
```

- [ ] **Step 5: Run checkpoint and CLI tests**

Run:

```bash
python -m pytest tests/test_checkpoint.py tests/test_cli.py -q
```

Expected: tests pass.

---

### Task 7: Update Public Exports And Run Full Verification

**Files:**
- Modify: `qchem_gnn/__init__.py`
- Test: full suite

- [ ] **Step 1: Export config helpers**

Modify `qchem_gnn/__init__.py`:

```python
from .config import ConfigError, config_to_namespace, load_yaml_config, resolve_config
```

Add to `__all__`:

```python
"ConfigError",
"config_to_namespace",
"load_yaml_config",
"resolve_config",
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
python -m pytest tests/test_config.py tests/test_cli.py tests/test_checkpoint.py -q
```

Expected: focused tests pass.

- [ ] **Step 3: Run full test suite**

Run:

```bash
python -m pytest tests -q
```

Expected: all tests pass.

- [ ] **Step 4: Manual smoke commands**

Run:

```bash
python -m qchem_gnn.cli train --config configs/minimal_train.yaml --epochs 1 --output runs/config_smoke_train.pt
python -m qchem_gnn.cli pretrain --config configs/minimal_pretrain.yaml --epochs 1 --output runs/config_smoke_pretrain.pt
python -m qchem_gnn.cli export-embeddings --config configs/export_embeddings.yaml --checkpoint runs/config_smoke_pretrain.pt --output runs/config_smoke_embeddings.pt
python -m qchem_gnn.cli eval --config configs/eval.yaml --checkpoint runs/config_smoke_pretrain.pt --output runs/config_smoke_eval.pt
```

Expected: each command exits with code 0 and writes the requested artifact.

---

## Success Criteria

- YAML files can drive `train`, `pretrain`, `export-embeddings`, `eval`, and `downstream`.
- Existing CLI flags still work.
- Explicit CLI flags override YAML values.
- Checkpoints and metrics include the resolved config for reproducibility.
- Example configs exist under `configs/`.
- `python -m pytest tests -q` passes.
