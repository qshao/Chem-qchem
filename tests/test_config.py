from __future__ import annotations

from pathlib import Path

import pytest

from qchem_gnn.config import ConfigError, config_to_namespace, load_yaml_config, resolve_config


def test_config_error_is_value_error_subclass():
    assert issubclass(ConfigError, ValueError)


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


def test_load_yaml_config_rejects_non_mapping(tmp_path: Path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("- train\n- pretrain\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="mapping"):
        load_yaml_config(config_path)


@pytest.mark.parametrize("content", ["", "null\n"])
def test_load_yaml_config_returns_empty_mapping_for_empty_or_null_document(tmp_path: Path, content: str):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text(content, encoding="utf-8")

    assert load_yaml_config(config_path) == {}


def test_load_yaml_config_rejects_missing_file(tmp_path: Path):
    config_path = tmp_path / "missing.yaml"

    with pytest.raises(ConfigError, match="Failed to read YAML config"):
        load_yaml_config(config_path)


def test_load_yaml_config_rejects_parse_errors(tmp_path: Path):
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("command: train\n  bad-indent: true\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Failed to parse YAML config"):
        load_yaml_config(config_path)


def test_resolve_config_rejects_unknown_top_level_keys():
    with pytest.raises(ConfigError, match="unknown top-level config keys"):
        resolve_config(
            {
                "command": "train",
                "dataset": {"csv": "subset.csv"},
                "outputs": {"checkpoint": "runs/out.pt"},
                "training": {},
                "traing": {"epochs": 1},
            }
        )


def test_resolve_config_rejects_non_mapping_sections():
    with pytest.raises(ConfigError, match="dataset section must be a mapping"):
        resolve_config(
            {
                "command": "train",
                "dataset": [],
                "outputs": {"checkpoint": "runs/out.pt"},
            }
        )


def test_resolve_config_rejects_bad_leaf_types():
    with pytest.raises(ConfigError, match="dataset.csv must be a string"):
        resolve_config(
            {
                "command": "train",
                "dataset": {"csv": 123},
                "outputs": {"checkpoint": "runs/out.pt"},
            }
        )

    with pytest.raises(ConfigError, match="dataset.csv must not be blank"):
        resolve_config(
            {
                "command": "train",
                "dataset": {"csv": "   "},
                "outputs": {"checkpoint": "runs/out.pt"},
            }
        )

    with pytest.raises(ConfigError, match="dataset.use_results must be a boolean"):
        resolve_config(
            {
                "command": "train",
                "dataset": {"csv": "subset.csv", "use_results": "true"},
                "outputs": {"checkpoint": "runs/out.pt"},
            }
        )

    with pytest.raises(ConfigError, match="training.learning_rate must be a finite number"):
        resolve_config(
            {
                "command": "train",
                "dataset": {"csv": "subset.csv"},
                "training": {"learning_rate": float("nan")},
                "outputs": {"checkpoint": "runs/out.pt"},
            }
        )


def test_resolve_config_merges_defaults_and_drops_none_overrides():
    resolved = resolve_config(
        {
            "command": "train",
            "dataset": {
                "csv": "subset_000.csv",
                "geometry": "coords_000.pkl",
                "limit": 8,
            },
            "outputs": {"checkpoint": "runs/from_yaml.pt"},
            "training": {"epochs": 2, "resume_from": "runs/resume.pt"},
        },
        overrides={
            "outputs": {"checkpoint": None, "metrics": "runs/from_cli.json"},
            "training": {"resume_from": None},
        },
    )

    assert resolved["dataset"]["limit"] == 8
    assert resolved["dataset"]["limit_per_shard"] == 16
    assert resolved["model"]["hidden_dim"] == 32
    assert resolved["training"]["epochs"] == 2
    assert resolved["training"]["resume_from"] == "runs/resume.pt"
    assert resolved["outputs"]["checkpoint"] == "runs/from_yaml.pt"
    assert resolved["outputs"]["metrics"] == "runs/from_cli.json"


def test_resolve_config_drops_none_items_from_lists():
    resolved = resolve_config(
        {
            "command": "downstream",
            "inference": {"checkpoint": "runs/model.pt"},
            "downstream": {
                "fractions": [0.1, None, 1.0],
            },
        }
    )

    assert resolved["downstream"]["fractions"] == [0.1, 1.0]


def test_resolve_config_normalizes_scalar_values():
    resolved = resolve_config(
        {
            "command": "train",
            "dataset": {
                "csv": "subset.csv",
                "subset_ids": ["0", "1"],
                "limit": "8",
                "use_results": False,
            },
            "training": {
                "epochs": "7",
                "learning_rate": "0.03",
                "resume_from": "runs/resume.pt",
            },
            "outputs": {"checkpoint": "runs/out.pt"},
        }
    )

    assert resolved["dataset"]["subset_ids"] == [0, 1]
    assert resolved["dataset"]["limit"] == 8
    assert resolved["training"]["epochs"] == 7
    assert resolved["training"]["learning_rate"] == 0.03


def test_resolve_config_rejects_range_errors_and_blank_paths():
    with pytest.raises(ConfigError, match="dataset.limit must be greater than zero"):
        resolve_config(
            {
                "command": "train",
                "dataset": {"csv": "subset.csv", "limit": 0},
                "outputs": {"checkpoint": "runs/out.pt"},
            }
        )

    with pytest.raises(ConfigError, match="training.epochs must be greater than zero"):
        resolve_config(
            {
                "command": "train",
                "dataset": {"csv": "subset.csv"},
                "training": {"epochs": 0},
                "outputs": {"checkpoint": "runs/out.pt"},
            }
        )

    with pytest.raises(ConfigError, match="training.learning_rate must be greater than zero"):
        resolve_config(
            {
                "command": "train",
                "dataset": {"csv": "subset.csv"},
                "training": {"learning_rate": -0.001},
                "outputs": {"checkpoint": "runs/out.pt"},
            }
        )

    with pytest.raises(ConfigError, match="downstream.fractions\[\] must be in the interval \(0, 1\]"):
        resolve_config(
            {
                "command": "downstream",
                "inference": {"checkpoint": "runs/model.pt"},
                "downstream": {"fractions": [0.0, 1.5]},
            }
        )

    with pytest.raises(ConfigError, match="outputs.checkpoint must not be blank"):
        resolve_config(
            {
                "command": "train",
                "dataset": {"csv": "subset.csv"},
                "outputs": {"checkpoint": "   "},
            }
        )


def test_resolve_config_rejects_nested_typos_and_bad_downstream_kind():
    with pytest.raises(ConfigError, match="training section has unknown keys"):
        resolve_config(
            {
                "command": "train",
                "dataset": {"csv": "subset.csv"},
                "training": {"epochs": 1, "learning_ratee": 0.1},
                "outputs": {"checkpoint": "runs/out.pt"},
            }
        )

    with pytest.raises(ConfigError, match="downstream.kind must be one of"):
        resolve_config(
            {
                "command": "downstream",
                "inference": {"checkpoint": "runs/model.pt"},
                "downstream": {"kind": "bad-kind"},
            }
        )

    with pytest.raises(ConfigError, match="downstream.fractions must not be empty"):
        resolve_config(
            {
                "command": "downstream",
                "inference": {"checkpoint": "runs/model.pt"},
                "downstream": {"fractions": []},
            }
        )


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (
            {
                "command": "train",
                "dataset": {"csv": "subset.csv", "dataset_root": "root", "subset_ids": [0]},
                "outputs": {"checkpoint": "runs/out.pt"},
            },
            "exactly one",
        ),
        (
            {
                "command": "train",
                "dataset": {"geometry": "coords.pkl"},
                "outputs": {"checkpoint": "runs/out.pt"},
            },
            "exactly one",
        ),
        (
            {
                "command": "train",
                "dataset": {"dataset_root": "root"},
                "outputs": {"checkpoint": "runs/out.pt"},
            },
            "subset_ids",
        ),
        (
            {
                "command": "pretrain",
                "dataset": {"csv": "subset.csv"},
            },
            "outputs.checkpoint",
        ),
        (
            {
                "command": "export-embeddings",
                "inference": {"output": "runs/embeddings.pt"},
            },
            "inference.checkpoint",
        ),
        (
            {
                "command": "export-embeddings",
                "inference": {"checkpoint": "runs/model.pt"},
            },
            "inference.output is required",
        ),
        (
            {
                "command": "export-embeddings",
                "inference": {"checkpoint": "runs/model.pt"},
                "outputs": {"embeddings": "runs/fallback.pt"},
            },
            "inference.output is required",
        ),
        (
            {
                "command": "eval",
                "inference": {},
            },
            "inference.checkpoint",
        ),
        (
            {
                "command": "downstream",
                "inference": {},
            },
            "inference.checkpoint",
        ),
    ],
)
def test_resolve_config_validates_command_specific_rules(config, match):
    with pytest.raises(ConfigError, match=match):
        resolve_config(config)


def test_config_to_namespace_rejects_bad_numeric_values():
    with pytest.raises(ConfigError, match=r"dataset.subset_ids\[\] must be an integer"):
        config_to_namespace(
            resolve_config(
                {
                    "command": "train",
                    "dataset": {"csv": "subset.csv", "subset_ids": [0, "bad"]},
                    "outputs": {"checkpoint": "runs/out.pt"},
                }
            )
        )

    with pytest.raises(ConfigError, match=r"downstream.fractions\[\] must be a number"):
        config_to_namespace(
            resolve_config(
                {
                    "command": "downstream",
                    "inference": {"checkpoint": "runs/model.pt"},
                    "downstream": {"fractions": [0.1, "bad"]},
                }
            )
        )

    with pytest.raises(ConfigError, match=r"training.epochs must be an integer"):
        config_to_namespace(
            resolve_config(
                {
                    "command": "train",
                    "dataset": {"csv": "subset.csv"},
                    "training": {"epochs": True},
                    "outputs": {"checkpoint": "runs/out.pt"},
                }
            )
        )

    with pytest.raises(ConfigError, match=r"training.epochs must be an integer"):
        config_to_namespace(
            resolve_config(
                {
                    "command": "train",
                    "dataset": {"csv": "subset.csv"},
                    "training": {"epochs": 3.9},
                    "outputs": {"checkpoint": "runs/out.pt"},
                }
            )
        )

    with pytest.raises(ConfigError, match=r"dataset.subset_ids must be a list"):
        config_to_namespace(
            resolve_config(
                {
                    "command": "train",
                    "dataset": {"csv": "subset.csv", "subset_ids": 1},
                    "outputs": {"checkpoint": "runs/out.pt"},
                }
            )
        )

    with pytest.raises(ConfigError, match=r"downstream.fractions must be a list"):
        config_to_namespace(
            resolve_config(
                {
                    "command": "downstream",
                    "inference": {"checkpoint": "runs/model.pt"},
                    "downstream": {"fractions": 0.1},
                }
            )
        )


def test_config_to_namespace_maps_pretrain_fields():
    resolved = resolve_config(
        {
            "command": "pretrain",
            "dataset": {
                "csv": "subset_000.csv",
                "geometry": "coords_000.pkl",
                "use_results": True,
                "limit": 8,
                "subset_ids": [0, 1],
            },
            "model": {"hidden_dim": 24, "message_passing_steps": 3},
            "training": {"epochs": 7, "learning_rate": 0.03, "aux_weight": 0.4},
            "outputs": {"checkpoint": "runs/pretrain.pt", "metrics": "runs/pretrain.json"},
        }
    )

    args = config_to_namespace(resolved)

    assert args.command == "pretrain"
    assert args.csv == "subset_000.csv"
    assert args.dataset_root is None
    assert args.subset_ids == "0,1"
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


def test_config_to_namespace_maps_export_eval_and_downstream_fields():
    export_resolved = resolve_config(
        {
            "command": "export-embeddings",
            "inference": {"checkpoint": "runs/model.pt", "output": "runs/embeddings.pt"},
        }
    )
    eval_resolved = resolve_config(
        {
            "command": "eval",
            "inference": {"checkpoint": "runs/model.pt", "output": "runs/eval.json"},
        }
    )
    downstream_resolved = resolve_config(
        {
            "command": "downstream",
            "inference": {"checkpoint": "runs/model.pt", "output": "runs/downstream.json"},
            "downstream": {
                "kind": "linear-probe",
                "epochs": 11,
                "learning_rate": 0.05,
                "fractions": [0.1, 1.0],
                "seed": 13,
            },
        }
    )

    export_args = config_to_namespace(export_resolved)
    eval_args = config_to_namespace(eval_resolved)
    downstream_args = config_to_namespace(downstream_resolved)

    assert export_args.command == "export-embeddings"
    assert export_args.checkpoint == "runs/model.pt"
    assert export_args.output == "runs/embeddings.pt"

    assert eval_args.command == "eval"
    assert eval_args.checkpoint == "runs/model.pt"
    assert eval_args.output == "runs/eval.json"

    assert downstream_args.command == "downstream"
    assert downstream_args.checkpoint == "runs/model.pt"
    assert downstream_args.kind == "linear-probe"
    assert downstream_args.output == "runs/downstream.json"
    assert downstream_args.epochs == 11
    assert downstream_args.learning_rate == 0.05
    assert downstream_args.fractions == "0.1,1.0"
    assert downstream_args.seed == 13

@pytest.mark.parametrize(
    "filename",
    [
        "minimal_train.yaml",
        "minimal_pretrain.yaml",
        "export_embeddings.yaml",
        "eval.yaml",
    ],
)
def test_example_configs_load_from_configs_directory(filename: str):
    config_path = Path(__file__).resolve().parents[1] / "configs" / filename
    config = load_yaml_config(config_path)
    resolved = resolve_config(config)

    assert config_path.exists()
    assert resolved["command"] in {"train", "pretrain", "export-embeddings", "eval"}


def test_contrastive_pretrain_config_resolves():
    config = {
        "command": "contrastive-pretrain",
        "dataset": {"csv": "subset_000.csv", "geometry": "coords_000.pkl", "limit": 4},
        "model": {"hidden_dim": 16, "message_passing_steps": 2},
        "training": {"epochs": 50, "learning_rate": 0.01},
        "contrastive": {"batch_size": 4, "contrastive_weight": 1.0, "temperature": 0.1, "hidden_dim_3d": 16},
        "outputs": {"checkpoint": "runs/contrastive.pt"},
    }

    resolved = resolve_config(config)
    namespace = config_to_namespace(resolved)

    assert namespace.command == "contrastive-pretrain"
    assert namespace.batch_size == 4
    assert namespace.temperature == 0.1
    assert namespace.hidden_dim_3d == 16
    assert namespace.output == "runs/contrastive.pt"


def test_minimal_contrastive_yaml_resolves():
    path = Path("configs/minimal_contrastive_pretrain.yaml")
    resolved = resolve_config(load_yaml_config(path))

    assert resolved["command"] == "contrastive-pretrain"
    assert resolved["contrastive"]["batch_size"] >= 2
    assert resolved["outputs"]["checkpoint"]


def _vicreg_base(extra_contrastive=None):
    return {
        "command": "contrastive-pretrain",
        "dataset": {"dataset_root": "zinc-250k", "subset_ids": [44]},
        "contrastive": extra_contrastive or {},
        "outputs": {"checkpoint": "out.pt"},
    }


def test_config_default_contrastive_loss_is_infonce():
    cfg = resolve_config(_vicreg_base())
    assert cfg["contrastive"]["contrastive_loss"] == "infonce"
    assert cfg["contrastive"]["vicreg_sim_weight"] == 25.0
    assert cfg["contrastive"]["vicreg_var_weight"] == 25.0
    assert cfg["contrastive"]["vicreg_cov_weight"] == 1.0


def test_config_accepts_vicreg_contrastive_loss():
    cfg = resolve_config(_vicreg_base({"contrastive_loss": "vicreg", "vicreg_cov_weight": 2.0}))
    assert cfg["contrastive"]["contrastive_loss"] == "vicreg"
    assert cfg["contrastive"]["vicreg_cov_weight"] == 2.0


def test_config_rejects_invalid_contrastive_loss():
    with pytest.raises(ConfigError):
        resolve_config(_vicreg_base({"contrastive_loss": "barlow"}))


def test_config_rejects_negative_vicreg_weight():
    with pytest.raises(ConfigError):
        resolve_config(_vicreg_base({"vicreg_sim_weight": -1.0}))


def test_namespace_includes_vicreg_fields():
    cfg = resolve_config(_vicreg_base({"contrastive_loss": "vicreg"}))
    ns = config_to_namespace(cfg)
    assert ns.contrastive_loss == "vicreg"
    assert ns.vicreg_sim_weight == 25.0
    assert ns.vicreg_cov_weight == 1.0
