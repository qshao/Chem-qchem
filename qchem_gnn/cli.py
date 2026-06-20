from __future__ import annotations

import argparse
import subprocess
import uuid
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .checkpoint import build_checkpoint_state, load_checkpoint, normalize_dataset_config, save_checkpoint
from .config import config_to_namespace, load_yaml_config, resolve_config
from .metrics import save_metrics
from .eval import run_fine_tuning, run_linear_probe, run_morgan_baseline, run_sample_efficiency
from .graph import GraphBatch
from .losses import compute_multitask_loss
from .minimal import load_quantum_zinc_dataset, train_on_minimal_dataset
from .pretrain import pretrain_on_minimal_dataset
from .model import MolecularQuantumGNN
from .quantum_data import load_quantum_zinc_subset_range, normalize_targets
from .shard_cache import preprocess_shard
from .splits import scaffold_or_random_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qchem_gnn", argument_default=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a tiny quantum-informed GNN")
    train.add_argument("--config", default=argparse.SUPPRESS, help="YAML config path")
    train.add_argument("--csv", help="Subset CSV path")
    train.add_argument("--dataset-root", help="Dataset root containing subsets/, geometries/, and results/")
    train.add_argument("--subset-ids", help="Comma-separated subset ids for dataset-root mode")
    train.add_argument("--geometry", help="Geometry pickle path")
    train.add_argument("--results", help="Optional HDF5 results path or directory")
    train.add_argument("--use-results", action="store_true", default=argparse.SUPPRESS, help="Load HDF5 quantum targets when available")
    train.add_argument("--resume-from", help="Optional checkpoint to resume from")
    train.add_argument("--limit", type=int, help="Maximum molecules to load")
    train.add_argument("--limit-per-shard", type=int, help="Maximum molecules to load per shard in dataset-root mode")
    train.add_argument("--epochs", type=int, help="Training epochs")
    train.add_argument("--hidden-dim", type=int, help="Hidden dimension")
    train.add_argument(
        "--message-passing-steps", type=int, help="Number of message passing steps"
    )
    train.add_argument("--learning-rate", type=float, help="Adam learning rate")
    train.add_argument("--metrics-output", help="Optional metrics output path")
    train.add_argument("--output", help="Checkpoint output path")

    pretrain = subparsers.add_parser("pretrain", help="Pretrain a quantum-informed GNN with auxiliary labels")
    pretrain.add_argument("--config", default=argparse.SUPPRESS, help="YAML config path")
    pretrain.add_argument("--csv", help="Subset CSV path")
    pretrain.add_argument("--dataset-root", help="Dataset root containing subsets/, geometries/, and results/")
    pretrain.add_argument("--subset-ids", help="Comma-separated subset ids for dataset-root mode")
    pretrain.add_argument("--geometry", help="Geometry pickle path")
    pretrain.add_argument("--results", help="Optional HDF5 results path or directory")
    pretrain.add_argument("--use-results", action="store_true", default=argparse.SUPPRESS, help="Load HDF5 quantum targets when available")
    pretrain.add_argument("--limit", type=int, help="Maximum molecules to load")
    pretrain.add_argument("--limit-per-shard", type=int, help="Maximum molecules to load per shard in dataset-root mode")
    pretrain.add_argument("--epochs", type=int, help="Training epochs")
    pretrain.add_argument("--hidden-dim", type=int, help="Hidden dimension")
    pretrain.add_argument(
        "--message-passing-steps", type=int, help="Number of message passing steps"
    )
    pretrain.add_argument("--learning-rate", type=float, help="Adam learning rate")
    pretrain.add_argument("--aux-weight", type=float, help="Weight for the auxiliary CSV target head")
    pretrain.add_argument("--metrics-output", help="Optional metrics output path")
    pretrain.add_argument("--output", help="Checkpoint output path")

    contrastive = subparsers.add_parser("contrastive-pretrain", help="Cross-modal 2D/3D contrastive pretraining")
    contrastive.add_argument("--config", default=argparse.SUPPRESS, help="YAML config path")
    contrastive.add_argument("--csv", help="Subset CSV path")
    contrastive.add_argument("--dataset-root", help="Dataset root containing subsets/, geometries/, and results/")
    contrastive.add_argument("--subset-ids", help="Comma-separated subset ids for dataset-root mode")
    contrastive.add_argument("--geometry", help="Geometry pickle path")
    contrastive.add_argument("--results", help="Optional HDF5 results path or directory")
    contrastive.add_argument("--use-results", action="store_true", default=argparse.SUPPRESS, help="Load HDF5 quantum targets when available")
    contrastive.add_argument("--limit", type=int, help="Maximum molecules to load")
    contrastive.add_argument("--limit-per-shard", type=int, help="Maximum molecules to load per shard in dataset-root mode")
    contrastive.add_argument("--total-steps", type=int, dest="total_steps", help="Total training steps")
    contrastive.add_argument("--hidden-dim", type=int, help="2D encoder hidden dimension")
    contrastive.add_argument("--message-passing-steps", type=int, help="2D message passing steps")
    contrastive.add_argument("--hidden-dim-3d", type=int, help="3D teacher hidden dimension")
    contrastive.add_argument("--message-passing-steps-3d", type=int, help="3D message passing steps")
    contrastive.add_argument("--num-rbf", type=int, help="Number of Gaussian RBF centers")
    contrastive.add_argument("--cutoff", type=float, help="RBF cutoff distance")
    contrastive.add_argument("--batch-size", type=int, help="Minibatch size (in-batch negatives)")
    contrastive.add_argument("--learning-rate", type=float, help="Adam learning rate")
    contrastive.add_argument("--supervised-weight", type=float, help="Weight for supervised quantum loss")
    contrastive.add_argument("--contrastive-weight", type=float, help="Weight for contrastive loss")
    contrastive.add_argument("--temperature", type=float, help="InfoNCE temperature")
    contrastive.add_argument("--teacher-weight", type=float, help="Weight for teacher regression loss")
    contrastive.add_argument("--energy-temperature", type=float, help="Temperature (K) for Boltzmann conformer pooling")
    contrastive.add_argument("--conformer-pool-mode", help="Conformer pooling mode: mean, weighted, or energy")
    contrastive.add_argument("--contrastive-loss", help="Contrastive objective: infonce or vicreg")
    contrastive.add_argument("--vicreg-sim-weight", type=float, help="VICReg invariance weight")
    contrastive.add_argument("--vicreg-var-weight", type=float, help="VICReg variance weight")
    contrastive.add_argument("--vicreg-cov-weight", type=float, help="VICReg covariance weight")
    contrastive.add_argument("--use-scaffold-negmask", action="store_true",
                              default=argparse.SUPPRESS,
                              help="Mask scaffold-similar negatives in InfoNCE")
    contrastive.add_argument("--seed", type=int, help="Random seed")
    contrastive.add_argument("--output", help="Checkpoint output path")

    export = subparsers.add_parser("export-embeddings", help="Export molecular embeddings from a checkpoint")
    export.add_argument("--config", default=argparse.SUPPRESS, help="YAML config path")
    export.add_argument("--checkpoint", help="Checkpoint path")
    export.add_argument("--output", help="Embedding output path")

    eval_cmd = subparsers.add_parser("eval", help="Evaluate a checkpoint on its saved dataset")
    eval_cmd.add_argument("--config", default=argparse.SUPPRESS, help="YAML config path")
    eval_cmd.add_argument("--checkpoint", help="Checkpoint path")
    eval_cmd.add_argument("--output", help="Optional metrics output path")

    downstream = subparsers.add_parser("downstream", help="Run downstream evaluation on a checkpoint")
    downstream.add_argument("--config", default=argparse.SUPPRESS, help="YAML config path")
    downstream.add_argument("--checkpoint", help="Checkpoint path")
    downstream.add_argument(
        "--kind",
        choices=("fine-tune", "linear-probe", "morgan-baseline", "sample-efficiency"),
        help="Downstream evaluation type",
    )
    downstream.add_argument("--output", help="Optional metrics output path")
    downstream.add_argument("--epochs", type=int, help="Fine-tuning epochs")
    downstream.add_argument("--learning-rate", type=float, help="Fine-tuning learning rate")
    downstream.add_argument(
        "--fractions",
        help="Comma-separated sample-efficiency fractions",
    )
    downstream.add_argument("--seed", type=int, help="Random seed")

    adapt_cmd = subparsers.add_parser("adapt", help="Adapt a frozen/fine-tuned backbone to a property CSV")
    adapt_cmd.add_argument("config", help="YAML config path")
    adapt_cmd.add_argument(
        "--override", "-O",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted-key overrides, e.g. training.epochs=100",
    )

    preprocess = subparsers.add_parser(
        "preprocess", help="Extract shards into compact scaffold-keyed caches"
    )
    preprocess.add_argument("--dataset-root", required=True, help="Dataset root with subsets/, geometries/, results/")
    preprocess.add_argument("--subset-ids", required=True, help="Comma-separated subset ids, e.g. 0,1,2")
    preprocess.add_argument("--cache-dir", required=True, help="Output directory for compact shard caches")
    preprocess.add_argument("--overwrite", action="store_true", default=False, help="Re-extract even if a valid cache exists")

    return parser


def _deep_merge_dict(base: dict[str, object], update: dict[str, object]) -> dict[str, object]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _config_from_args(args) -> dict[str, object]:
    config: dict[str, object] = {"command": args.command}

    if args.command in {"train", "pretrain", "contrastive-pretrain"}:
        dataset: dict[str, object] = {}
        for key in ("csv", "dataset_root", "subset_ids", "geometry", "results", "use_results", "limit", "limit_per_shard"):
            if hasattr(args, key):
                value = getattr(args, key)
                if key == "subset_ids" and isinstance(value, str):
                    value = [int(part.strip()) for part in value.split(",") if part.strip()]
                dataset[key] = value
        if dataset.get("dataset_root") is not None:
            dataset["csv"] = None
            dataset["geometry"] = None
        if dataset:
            config["dataset"] = dataset

        model: dict[str, object] = {}
        for key in ("hidden_dim", "message_passing_steps"):
            if hasattr(args, key):
                model[key] = getattr(args, key)
        if model:
            config["model"] = model

        training: dict[str, object] = {}
        for key in ("epochs", "learning_rate", "aux_weight", "resume_from"):
            if hasattr(args, key):
                training[key] = getattr(args, key)
        if training:
            config["training"] = training

        if args.command == "contrastive-pretrain":
            contrastive_cfg: dict[str, object] = {}
            for arg_name, key in (
                ("total_steps", "total_steps"),
                ("hidden_dim_3d", "hidden_dim_3d"),
                ("message_passing_steps_3d", "message_passing_steps_3d"),
                ("num_rbf", "num_rbf"),
                ("cutoff", "cutoff"),
                ("batch_size", "batch_size"),
                ("supervised_weight", "supervised_weight"),
                ("contrastive_weight", "contrastive_weight"),
                ("temperature", "temperature"),
                ("teacher_weight", "teacher_weight"),
                ("energy_temperature", "energy_temperature"),
                ("conformer_pool_mode", "conformer_pool_mode"),
                ("contrastive_loss", "contrastive_loss"),
                ("vicreg_sim_weight", "vicreg_sim_weight"),
                ("vicreg_var_weight", "vicreg_var_weight"),
                ("vicreg_cov_weight", "vicreg_cov_weight"),
                ("use_scaffold_negmask", "use_scaffold_negmask"),
                ("seed", "seed"),
            ):
                if hasattr(args, arg_name):
                    contrastive_cfg[key] = getattr(args, arg_name)
            if contrastive_cfg:
                config["contrastive"] = contrastive_cfg

        outputs: dict[str, object] = {}
        for arg_name, key in (("metrics_output", "metrics"), ("output", "checkpoint")):
            if hasattr(args, arg_name):
                outputs[key] = getattr(args, arg_name)
        if outputs:
            config["outputs"] = outputs

    elif args.command in {"export-embeddings", "eval"}:
        inference: dict[str, object] = {}
        for arg_name, key in (("checkpoint", "checkpoint"), ("output", "output")):
            if hasattr(args, arg_name):
                inference[key] = getattr(args, arg_name)
        if inference:
            config["inference"] = inference

    elif args.command == "downstream":
        inference: dict[str, object] = {}
        for arg_name, key in (("checkpoint", "checkpoint"), ("output", "output")):
            if hasattr(args, arg_name):
                inference[key] = getattr(args, arg_name)
        if inference:
            config["inference"] = inference

        downstream: dict[str, object] = {}
        for arg_name, key in (("kind", "kind"), ("epochs", "epochs"), ("learning_rate", "learning_rate"), ("fractions", "fractions"), ("seed", "seed")):
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if arg_name == "fractions" and isinstance(value, str):
                    value = [float(part) for part in value.split(",") if part.strip()]
                downstream[key] = value
        if downstream:
            config["downstream"] = downstream

    return config


def _resolved_namespace_from_args(args) -> argparse.Namespace:
    yaml_config = load_yaml_config(args.config) if hasattr(args, "config") else {}
    cli_config = _config_from_args(args)

    resume_from = None
    for source in (cli_config, yaml_config):
        training = source.get("training") if isinstance(source.get("training"), dict) else {}
        resume_from = training.get("resume_from") if isinstance(training, dict) else None
        if resume_from:
            break

    base_config: dict[str, object] = {}
    if args.command == "train" and resume_from:
        checkpoint = load_checkpoint(Path(resume_from))
        checkpoint_model = dict(checkpoint.get("model_config", {}))
        base_config = {
            "dataset": dict(checkpoint.get("dataset_config", {})),
            "model": {
                key: checkpoint_model[key]
                for key in ("hidden_dim", "message_passing_steps")
                if key in checkpoint_model
            },
        }

    merged_config = _deep_merge_dict(base_config, yaml_config) if base_config else yaml_config
    if args.command in {"train", "pretrain", "contrastive-pretrain"} and isinstance(merged_config.get("dataset"), dict):
        cli_dataset = cli_config.get("dataset") if isinstance(cli_config.get("dataset"), dict) else {}
        merged_dataset = merged_config["dataset"]
        if cli_dataset.get("csv") is not None:
            merged_dataset["dataset_root"] = None
            merged_dataset["subset_ids"] = []
        elif cli_dataset.get("dataset_root") is not None:
            merged_dataset["csv"] = None
            merged_dataset["geometry"] = None
    resolved_config = resolve_config(merged_config, overrides=cli_config)
    namespace = config_to_namespace(resolved_config)
    namespace.resolved_config = resolved_config
    return namespace



def _run_identity() -> dict[str, object]:
    run_id = uuid.uuid4().hex
    git_commit = None
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except Exception:
        pass
    else:
        git_commit = completed.stdout.strip() or None
    return {"run_id": run_id, "git_commit": git_commit}


def _checkpoint_run_metadata(existing: dict[str, object] | None = None) -> dict[str, object]:
    metadata = dict(existing or {})
    generated = _run_identity()
    metadata["run_id"] = metadata.get("run_id") or generated["run_id"]
    metadata["git_commit"] = metadata.get("git_commit") or generated["git_commit"]
    return metadata


def _training_metrics(dataset, result, *, use_results: bool, hidden_dim: int, message_passing_steps: int, learning_rate: float, epochs: int, run_metadata: dict[str, object] | None = None, extra: dict[str, object] | None = None) -> dict[str, object]:
    metrics = {
        "num_examples": len(dataset),
        "num_skipped": len(dataset.skipped_mol_ids),
        "use_results": bool(use_results),
        "hidden_dim": int(hidden_dim),
        "message_passing_steps": int(message_passing_steps),
        "learning_rate": float(learning_rate),
        "epochs": int(epochs),
        "final_loss": float(result.loss_history[-1]) if result.loss_history else None,
        "best_loss": float(min(result.loss_history)) if result.loss_history else None,
        "loss_history": result.loss_history,
        "run_id": (run_metadata or {}).get("run_id"),
        "git_commit": (run_metadata or {}).get("git_commit"),
    }
    if extra:
        metrics.update(extra)
    return metrics


def _dataset_kwargs_from_args(args) -> dict[str, object]:
    return {
        "csv": args.csv,
        "dataset_root": args.dataset_root,
        "subset_ids": args.subset_ids,
        "geometry": args.geometry,
        "results": args.results,
        "use_results": bool(args.use_results),
        "limit": int(args.limit),
        "limit_per_shard": int(args.limit_per_shard),
    }


def _build_dataset_from_config(dataset_config: dict[str, object]):
    if dataset_config["dataset_root"]:
        if not dataset_config["subset_ids"]:
            raise SystemExit("checkpoint is missing subset_ids")
        subset_ids = list(dataset_config["subset_ids"])
        return load_quantum_zinc_subset_range(
            dataset_config["dataset_root"],
            subset_ids=subset_ids,
            limit_per_shard=int(dataset_config["limit_per_shard"]),
            results_path=dataset_config["results"],
            use_results=bool(dataset_config["use_results"]),
        )
    if not dataset_config["csv"]:
        raise SystemExit("checkpoint is missing csv")
    return load_quantum_zinc_dataset(
        dataset_config["csv"],
        geometry_path=dataset_config["geometry"],
        results_path=dataset_config["results"],
        limit=int(dataset_config["limit"]),
        use_results=bool(dataset_config["use_results"]),
    )


def run_pretrain(args) -> int:
    current_config = normalize_dataset_config(_dataset_kwargs_from_args(args))
    if args.dataset_root:
        if not args.subset_ids:
            raise SystemExit("--subset-ids is required when --dataset-root is used")
        subset_ids = [int(part.strip()) for part in args.subset_ids.split(",") if part.strip()]
        dataset = load_quantum_zinc_subset_range(
            args.dataset_root,
            subset_ids=subset_ids,
            limit_per_shard=args.limit_per_shard,
            results_path=args.results,
            use_results=args.use_results,
        )
    else:
        if not args.csv:
            raise SystemExit("--csv is required when --dataset-root is not used")
        dataset = load_quantum_zinc_dataset(
            args.csv,
            geometry_path=args.geometry,
            results_path=args.results,
            limit=args.limit,
            use_results=args.use_results,
        )
    split_metadata = {
        "subset_ids": [int(part.strip()) for part in args.subset_ids.split(",") if part.strip()]
        if args.subset_ids
        else [],
    }

    resolved_config = getattr(args, "resolved_config", None)
    run_metadata = _checkpoint_run_metadata()
    node_targets = int(dataset.examples[0].node_target.shape[-1])
    result = pretrain_on_minimal_dataset(
        dataset,
        hidden_dim=args.hidden_dim,
        num_message_passing_steps=args.message_passing_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        aux_weight=args.aux_weight,
    )

    output_path = Path(args.output)
    checkpoint_payload = build_checkpoint_state(
        loss_history=result.loss_history,
        embeddings=result.embeddings,
        model_state_dict=result.model.state_dict(),
        optimizer_state_dict=result.optimizer_state_dict,
        epoch=result.epoch,
        global_step=result.global_step,
        target_normalization=result.target_normalization,
        dataset_config=current_config,
        split_metadata=split_metadata,
        model_config={
            "atom_vocab_size": 128,
            "bond_vocab_size": 8,
            "hidden_dim": args.hidden_dim,
            "num_message_passing_steps": args.message_passing_steps,
            "node_targets": node_targets,
            "graph_targets": 2,
        },
        run_metadata={
            "num_examples": len(dataset),
            "num_skipped": len(dataset.skipped_mol_ids),
            "limit": args.limit,
            "limit_per_shard": args.limit_per_shard,
            "epochs": args.epochs,
            "aux_weight": args.aux_weight,
            **run_metadata,
            "resolved_config": resolved_config,
        },
    )
    checkpoint_payload["aux_head_state_dict"] = result.aux_head_state_dict
    checkpoint_payload["aux_normalization"] = result.aux_normalization
    save_checkpoint(output_path, checkpoint_payload)
    if args.metrics_output:
        save_metrics(args.metrics_output, _training_metrics(dataset, result, use_results=args.use_results, hidden_dim=args.hidden_dim, message_passing_steps=args.message_passing_steps, learning_rate=args.learning_rate, epochs=args.epochs, run_metadata=run_metadata, extra={"aux_weight": args.aux_weight}))
    return 0

def run_contrastive_pretrain(args) -> int:
    from .contrastive_pretrain import contrastive_pretrain_on_dataset

    current_config = normalize_dataset_config(_dataset_kwargs_from_args(args))
    if args.dataset_root:
        if not args.subset_ids:
            raise SystemExit("--subset-ids is required when --dataset-root is used")
        subset_ids = [int(part.strip()) for part in args.subset_ids.split(",") if part.strip()]
        dataset = load_quantum_zinc_subset_range(
            args.dataset_root,
            subset_ids=subset_ids,
            limit_per_shard=args.limit_per_shard,
            results_path=args.results,
            use_results=args.use_results,
        )
    else:
        if not args.csv:
            raise SystemExit("--csv is required when --dataset-root is not used")
        dataset = load_quantum_zinc_dataset(
            args.csv,
            geometry_path=args.geometry,
            results_path=args.results,
            limit=args.limit,
            use_results=args.use_results,
        )

    result = contrastive_pretrain_on_dataset(
        dataset,
        metrics_path=Path(args.output).with_suffix(".metrics.jsonl"),
        hidden_dim=args.hidden_dim,
        num_message_passing_steps=args.message_passing_steps,
        hidden_dim_3d=args.hidden_dim_3d,
        num_rbf=args.num_rbf,
        cutoff=args.cutoff,
        num_message_passing_steps_3d=args.message_passing_steps_3d,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        supervised_weight=args.supervised_weight,
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature,
        teacher_weight=args.teacher_weight,
        energy_temperature=args.energy_temperature,
        conformer_pool_mode=args.conformer_pool_mode,
        contrastive_loss=args.contrastive_loss,
        vicreg_sim_weight=args.vicreg_sim_weight,
        vicreg_var_weight=args.vicreg_var_weight,
        vicreg_cov_weight=args.vicreg_cov_weight,
        use_scaffold_negmask=args.use_scaffold_negmask,
        seed=args.seed,
    )

    split_metadata = {
        "subset_ids": [int(part.strip()) for part in args.subset_ids.split(",") if part.strip()]
        if args.subset_ids
        else [],
    }
    run_metadata = _checkpoint_run_metadata()
    checkpoint_payload = build_checkpoint_state(
        loss_history=result.loss_history,
        embeddings=result.embeddings,
        model_state_dict=result.model.state_dict(),
        optimizer_state_dict=result.optimizer_state_dict,
        epoch=result.global_step,
        global_step=result.global_step,
        target_normalization=result.target_normalization,
        dataset_config=current_config,
        split_metadata=split_metadata,
        model_config={
            "atom_vocab_size": 128,
            "bond_vocab_size": 8,
            "hidden_dim": args.hidden_dim,
            "num_message_passing_steps": args.message_passing_steps,
            "graph_targets": 2,
        },
        run_metadata={
            "num_examples": len(dataset),
            "num_skipped": len(dataset.skipped_mol_ids),
            "total_steps": args.total_steps,
            "contrastive_weight": args.contrastive_weight,
            "contrastive_loss_history": result.contrastive_loss_history,
            **run_metadata,
            "resolved_config": getattr(args, "resolved_config", None),
        },
    )
    save_checkpoint(Path(args.output), checkpoint_payload)
    return 0


def run_train(args) -> int:
    resolved_config = getattr(args, "resolved_config", None)
    initial_model_state_dict = None
    initial_optimizer_state_dict = None
    start_epoch = 0
    start_global_step = 0
    if args.resume_from:
        checkpoint = load_checkpoint(Path(args.resume_from))
        checkpoint_model = dict(checkpoint.get("model_config", {}))
        dataset_config = normalize_dataset_config(_dataset_kwargs_from_args(args))
        split_metadata = {
            "subset_ids": [int(part.strip()) for part in args.subset_ids.split(",") if part.strip()]
            if args.subset_ids
            else [],
        }
        model_config = {
            "atom_vocab_size": 128,
            "bond_vocab_size": 8,
            "hidden_dim": args.hidden_dim,
            "num_message_passing_steps": args.message_passing_steps,
            "graph_targets": 2,
        }
        initial_model_state_dict = checkpoint.get("model_state_dict")
        initial_optimizer_state_dict = checkpoint.get("optimizer_state_dict")
        start_epoch = int(checkpoint.get("epoch", 0))
        start_global_step = int(checkpoint.get("global_step", 0))
        checkpoint_hidden_dim = int(checkpoint_model.get("hidden_dim", args.hidden_dim))
        checkpoint_message_passing_steps = int(checkpoint_model.get("num_message_passing_steps", args.message_passing_steps))
        if args.hidden_dim != checkpoint_hidden_dim or args.message_passing_steps != checkpoint_message_passing_steps:
            raise SystemExit("--resume-from requires hidden_dim and message-passing-steps to match the checkpoint")
        dataset = _build_dataset_from_config(dataset_config)
        node_targets = int(dataset.examples[0].node_target.shape[-1])
        model_config["node_targets"] = node_targets
        result = train_on_minimal_dataset(
            dataset,
            hidden_dim=args.hidden_dim,
            num_message_passing_steps=args.message_passing_steps,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            initial_model_state_dict=initial_model_state_dict,
            initial_optimizer_state_dict=initial_optimizer_state_dict,
            initial_scheduler_state_dict=checkpoint.get("scheduler_state_dict"),
            start_epoch=start_epoch,
            start_global_step=start_global_step,
        )
        output_path = Path(args.output)
        checkpoint_payload = build_checkpoint_state(
            loss_history=result.loss_history,
            embeddings=result.embeddings,
            model_state_dict=result.model.state_dict(),
            optimizer_state_dict=result.optimizer_state_dict,
            epoch=result.epoch,
            global_step=result.global_step,
            target_normalization=result.target_normalization,
            dataset_config=dataset_config,
            split_metadata=split_metadata,
            model_config=model_config,
            run_metadata={
                "num_examples": len(dataset),
                "num_skipped": len(dataset.skipped_mol_ids),
                "limit": int(dataset_config.get("limit", args.limit)),
                "limit_per_shard": int(dataset_config.get("limit_per_shard", args.limit_per_shard)),
                "epochs": args.epochs,
                **_checkpoint_run_metadata(checkpoint.get("run_metadata")),
                "resolved_config": resolved_config,
            },
            scheduler_state_dict=result.scheduler_state_dict,
        )
        save_checkpoint(output_path, checkpoint_payload)
        return 0

    current_config = normalize_dataset_config(_dataset_kwargs_from_args(args))
    if args.dataset_root:
        if not args.subset_ids:
            raise SystemExit("--subset-ids is required when --dataset-root is used")
        subset_ids = [int(part.strip()) for part in args.subset_ids.split(",") if part.strip()]
        dataset = load_quantum_zinc_subset_range(
            args.dataset_root,
            subset_ids=subset_ids,
            limit_per_shard=args.limit_per_shard,
            results_path=args.results,
            use_results=args.use_results,
        )
    else:
        if not args.csv:
            raise SystemExit("--csv is required when --dataset-root is not used")
        dataset = load_quantum_zinc_dataset(
            args.csv,
            geometry_path=args.geometry,
            results_path=args.results,
            limit=args.limit,
            use_results=args.use_results,
        )
    split_metadata = {
        "subset_ids": [int(part.strip()) for part in args.subset_ids.split(",") if part.strip()]
        if args.subset_ids
        else [],
    }

    resolved_config = getattr(args, "resolved_config", None)
    run_metadata = _checkpoint_run_metadata()
    node_targets = int(dataset.examples[0].node_target.shape[-1])
    result = train_on_minimal_dataset(
        dataset,
        hidden_dim=args.hidden_dim,
        num_message_passing_steps=args.message_passing_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        start_epoch=start_epoch,
        start_global_step=start_global_step,
    )

    output_path = Path(args.output)
    checkpoint_payload = build_checkpoint_state(
        loss_history=result.loss_history,
        embeddings=result.embeddings,
        model_state_dict=result.model.state_dict(),
        optimizer_state_dict=result.optimizer_state_dict,
        epoch=result.epoch,
        global_step=result.global_step,
        target_normalization=result.target_normalization,
        dataset_config=current_config,
        split_metadata=split_metadata,
        model_config={
            "atom_vocab_size": 128,
            "bond_vocab_size": 8,
            "hidden_dim": args.hidden_dim,
            "num_message_passing_steps": args.message_passing_steps,
            "node_targets": node_targets,
            "graph_targets": 2,
        },
        run_metadata={
            "num_examples": len(dataset),
            "num_skipped": len(dataset.skipped_mol_ids),
            "limit": args.limit,
            "limit_per_shard": args.limit_per_shard,
            "epochs": args.epochs,
            **run_metadata,
            "resolved_config": resolved_config,
        },
        scheduler_state_dict=result.scheduler_state_dict,
    )
    save_checkpoint(output_path, checkpoint_payload)
    if args.metrics_output:
        save_metrics(args.metrics_output, _training_metrics(dataset, result, use_results=args.use_results, hidden_dim=args.hidden_dim, message_passing_steps=args.message_passing_steps, learning_rate=args.learning_rate, epochs=args.epochs, run_metadata=run_metadata))
    return 0


def _load_model_from_checkpoint(checkpoint_path: str) -> tuple[MolecularQuantumGNN, dict]:
    checkpoint = load_checkpoint(Path(checkpoint_path))
    model = MolecularQuantumGNN(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def run_export_embeddings(args) -> int:
    model, checkpoint = _load_model_from_checkpoint(args.checkpoint)
    dataset = _build_dataset_from_config(checkpoint["dataset_config"])
    batch = GraphBatch.from_graphs([example.graph for example in dataset.examples])
    model.eval()
    with torch.no_grad():
        embeddings = model.encode_graph_embeddings(batch)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings,
            "mol_ids": [example.mol_id for example in dataset.examples],
            "checkpoint_epoch": checkpoint.get("epoch", 0),
            "checkpoint_global_step": checkpoint.get("global_step", 0),
            "checkpoint_run_metadata": checkpoint.get("run_metadata", {}),
        },
        output_path,
    )
    return 0


def run_eval(args) -> int:
    model, checkpoint = _load_model_from_checkpoint(args.checkpoint)
    dataset = _build_dataset_from_config(checkpoint["dataset_config"])
    batch = GraphBatch.from_graphs([example.graph for example in dataset.examples])
    model.eval()
    with torch.no_grad():
        node_pred, edge_pred, graph_pred, embedding = model(batch)
        raw_targets = (
            torch.cat([example.node_target for example in dataset.examples], dim=0),
            torch.cat([example.edge_target for example in dataset.examples], dim=0),
            torch.stack([example.graph_target for example in dataset.examples], dim=0),
        )
        node_target, edge_target, graph_target = normalize_targets(
            raw_targets[0],
            raw_targets[1],
            raw_targets[2],
            checkpoint["target_normalization"],
        )
        loss = compute_multitask_loss((node_pred, edge_pred, graph_pred, embedding), (node_target, edge_target, graph_target))
    metrics = {
        "num_examples": len(dataset),
        "loss": float(loss.item()),
        "embedding_norm_mean": float(embedding.norm(dim=-1).mean().item()),
        "checkpoint_epoch": checkpoint.get("epoch", 0),
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(metrics, output_path)
    return 0


def run_downstream(args) -> int:
    model, checkpoint = _load_model_from_checkpoint(args.checkpoint)
    dataset = _build_dataset_from_config(checkpoint["dataset_config"])
    split = scaffold_or_random_split([example.mol_id for example in dataset.examples], seed=args.seed)
    batch = GraphBatch.from_graphs([example.graph for example in dataset.examples])
    labels = np.stack([example.graph_target.detach().cpu().numpy() for example in dataset.examples], axis=0)
    smiles = [example.smiles for example in dataset.examples]

    if args.kind == "fine-tune":
        metrics = run_fine_tuning(
            dataset,
            args.checkpoint,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    else:
        model.eval()
        with torch.no_grad():
            embeddings = model.encode_graph_embeddings(batch).detach().cpu().numpy()
        if args.kind == "linear-probe":
            metrics = run_linear_probe(embeddings, labels, split)
        elif args.kind == "morgan-baseline":
            metrics = run_morgan_baseline(smiles, labels, split)
        elif args.kind == "sample-efficiency":
            fractions = [float(part) for part in args.fractions.split(",") if part.strip()]
            metrics = run_sample_efficiency(embeddings, labels, split, fractions=fractions, seed=args.seed)
        else:
            raise SystemExit(f"Unknown downstream kind: {args.kind}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(metrics, output_path)
    return 0


def _parse_overrides(override_list):
    result = {}
    for item in (override_list or []):
        key, _, val_str = item.partition("=")
        # try to coerce to int, then float, else keep as str
        try:
            val = int(val_str)
        except ValueError:
            try:
                val = float(val_str)
            except ValueError:
                val = val_str
        # set dotted key
        parts = key.split(".")
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return result


def run_adapt(args) -> int:
    from .adapt.config import resolve_adapt_config
    from .adapt.runner import run as run_adapt_single
    from .adapt.sweep import run_sweep
    raw = load_yaml_config(args.config)
    cfg = resolve_adapt_config(raw, overrides=_parse_overrides(args.override))
    if cfg.sweep is not None:
        run_sweep(cfg)
    else:
        summary = run_adapt_single(cfg)
        print(summary)
    return 0


def run_preprocess(args) -> int:
    subset_ids = [int(s) for s in str(args.subset_ids).split(",") if s != ""]
    for subset_id in subset_ids:
        path = preprocess_shard(
            args.dataset_root,
            subset_id,
            args.cache_dir,
            overwrite=getattr(args, "overwrite", False),
        )
        print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = parser.parse_args(argv)
    if raw_args.command == "adapt":
        return run_adapt(raw_args)
    if raw_args.command == "preprocess":
        return run_preprocess(raw_args)
    args = _resolved_namespace_from_args(raw_args)

    if args.command == "train":
        return run_train(args)
    if args.command == "pretrain":
        return run_pretrain(args)
    if args.command == "contrastive-pretrain":
        return run_contrastive_pretrain(args)
    if args.command == "export-embeddings":
        return run_export_embeddings(args)
    if args.command == "eval":
        return run_eval(args)
    if args.command == "downstream":
        return run_downstream(args)

    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
