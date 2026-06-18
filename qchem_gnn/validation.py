from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

from .adapt import resolve_adapt_config
from .adapt import run as adapt_run
from .conformer import ConformerEncoderBatch
from .contrastive_pretrain import (
    CheckpointMismatchError,
    contrastive_pretrain_on_dataset,
)
from .eval import scaffold_key_from_smiles
from .minimal import MinimalQuantumDataset
from .teacher_heads import assemble_conformer_targets

ARMS = ("baseline", "quantum")
DECISIVE_METHOD = "mlp_head"
DECISIVE_METRIC = "mae"
INTRINSIC_PROPERTIES = ("chelpg", "energy", "iso_polarizability", "wbi")


def split_holdout(
    dataset: MinimalQuantumDataset, fraction: float, seed: int
) -> tuple[MinimalQuantumDataset, MinimalQuantumDataset]:
    """Split a dataset into (pretrain, holdout). Deterministic for a fixed seed."""
    examples = dataset.examples
    n = len(examples)
    n_holdout = max(1, round(n * fraction))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator).tolist()
    holdout_positions = set(order[:n_holdout])
    pretrain = [ex for i, ex in enumerate(examples) if i not in holdout_positions]
    holdout = [ex for i, ex in enumerate(examples) if i in holdout_positions]
    return (
        MinimalQuantumDataset(examples=pretrain),
        MinimalQuantumDataset(examples=holdout),
    )


def scaffold_hash_holdout(
    dataset: MinimalQuantumDataset, k: int
) -> tuple[MinimalQuantumDataset, MinimalQuantumDataset]:
    """Deterministic scaffold-disjoint holdout: holdout iff scaffold_key % k == 0.

    Uses each example's stored ``scaffold_key`` (falling back to computing it from
    SMILES when absent). A scaffold lands in the same split regardless of which
    shard it came from, so pretrain and holdout never share a scaffold.
    """
    if k < 1:
        raise ValueError(f"holdout k must be >= 1, got {k}")
    pretrain = []
    holdout = []
    for ex in dataset.examples:
        key = ex.scaffold_key
        if key is None:
            key = scaffold_key_from_smiles(ex.smiles)
        if key % k == 0:
            holdout.append(ex)
        else:
            pretrain.append(ex)
    if not pretrain or not holdout:
        raise ValueError(
            f"scaffold_hash_holdout(k={k}) produced an empty side "
            f"(pretrain={len(pretrain)}, holdout={len(holdout)})"
        )
    return (
        MinimalQuantumDataset(examples=pretrain),
        MinimalQuantumDataset(examples=holdout),
    )


def _pearson_mae(pred: torch.Tensor, target: torch.Tensor) -> dict:
    p = pred.detach().reshape(-1).float()
    t = target.detach().reshape(-1).float()
    mae = float((p - t).abs().mean())
    if p.numel() < 2 or float(p.std()) == 0.0 or float(t.std()) == 0.0:
        return {"r": 0.0, "mae": mae}
    pc = p - p.mean()
    tc = t - t.mean()
    r = float((pc @ tc) / (pc.norm() * tc.norm()))
    return {"r": r, "mae": mae}


def evaluate_teacher(teacher, encoder3d, holdout_examples: list) -> dict:
    """Score the trained teacher on held-out conformers vs DFT labels."""
    usable = [
        ex
        for ex in holdout_examples
        if ex.conformer_coords and ex.conformer_node_targets is not None
    ]
    if not usable:
        raise ValueError("holdout has no examples with per-conformer targets")

    batch = ConformerEncoderBatch.from_molecule_conformers(
        [ex.graph for ex in usable],
        [ex.conformer_coords for ex in usable],
        conformer_energies=[ex.conformer_energies for ex in usable],
    )
    encoder3d.eval()
    teacher.eval()
    with torch.no_grad():
        node_states, conf_emb = encoder3d.forward_with_nodes(
            batch.atomic_numbers,
            batch.edge_index,
            batch.positions,
            batch.node_conformer_index,
            batch.num_conformers,
        )
        node_pred, edge_pred, graph_pred = teacher(node_states, batch.edge_index, conf_emb)

    node_t, edge_t, graph_t, _ = assemble_conformer_targets(usable)
    return {
        "chelpg": _pearson_mae(node_pred[:, 0], node_t[:, 0]),
        "energy": _pearson_mae(graph_pred[:, 0], graph_t[:, 0]),
        "iso_polarizability": _pearson_mae(graph_pred[:, 1], graph_t[:, 1]),
        "wbi": _pearson_mae(edge_pred[:, 0], edge_t[:, 0]),
    }


def _mean_std(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "n": 0}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n >= 2 else None
    return {"mean": mean, "std": std, "n": n}


def _verdict(name, reference_arm, treatment_arm, reference: dict, treatment: dict) -> dict:
    # Lower MAE is better; the treatment "helps" if reference_mean - treatment_mean
    # exceeds the combined seed noise sqrt(std_ref^2 + std_treat^2).
    out = {"name": name, "reference": reference_arm, "treatment": treatment_arm,
           "method": DECISIVE_METHOD, "metric": DECISIVE_METRIC,
           "delta": None, "combined_std": None, "result": "n/a"}
    if reference["n"] == 0 or treatment["n"] == 0:
        out["result"] = "n/a"
        return out
    if reference["std"] is None or treatment["std"] is None:
        out["result"] = "insufficient seeds"
        return out
    delta = reference["mean"] - treatment["mean"]
    combined = (reference["std"] ** 2 + treatment["std"] ** 2) ** 0.5
    out["delta"] = delta
    out["combined_std"] = combined
    out["result"] = "helps" if delta > combined else "within noise"
    return out


_DEFAULT_COMPARISONS = [
    {"name": "teacher_vs_baseline", "reference": "baseline", "treatment": "quantum"}
]


def aggregate_results(extrinsic_rows: list[dict], intrinsic_rows: list[dict],
                      arms=None, comparisons=None) -> dict:
    if arms is None:
        arms = []
        for r in extrinsic_rows + intrinsic_rows:
            if r["arm"] not in arms:
                arms.append(r["arm"])
        if not arms:
            arms = list(ARMS)
    if comparisons is None:
        comparisons = list(_DEFAULT_COMPARISONS)

    methods = sorted({r["method"] for r in extrinsic_rows})
    extrinsic: dict = {}
    for method in methods:
        extrinsic[method] = {}
        for arm in arms:
            ok = [r for r in extrinsic_rows
                  if r["method"] == method and r["arm"] == arm and r["status"] == "ok"]
            mae = _mean_std([r["mae"] for r in ok])
            r2 = _mean_std([r["r2"] for r in ok])
            extrinsic[method][arm] = {
                "mae_mean": mae["mean"], "mae_std": mae["std"],
                "r2_mean": r2["mean"], "r2_std": r2["std"], "n": mae["n"],
            }

    decisive = extrinsic.get(DECISIVE_METHOD, {})
    verdicts: list[dict] = []
    for comp in comparisons:
        ref = decisive.get(comp["reference"])
        treat = decisive.get(comp["treatment"])
        if ref is None or treat is None:
            verdicts.append({"name": comp["name"], "reference": comp["reference"],
                             "treatment": comp["treatment"], "method": DECISIVE_METHOD,
                             "metric": DECISIVE_METRIC, "delta": None,
                             "combined_std": None, "result": "n/a"})
            continue
        verdicts.append(_verdict(
            comp["name"], comp["reference"], comp["treatment"],
            {"mean": ref["mae_mean"], "std": ref["mae_std"], "n": ref["n"]},
            {"mean": treat["mae_mean"], "std": treat["mae_std"], "n": treat["n"]},
        ))

    intrinsic: dict = {}
    for arm in arms:
        ok = [r for r in intrinsic_rows if r["arm"] == arm and r["status"] == "ok"]
        intrinsic[arm] = {}
        for prop in INTRINSIC_PROPERTIES:
            rs = _mean_std([r["properties"][prop]["r"] for r in ok if prop in r["properties"]])
            ms = _mean_std([r["properties"][prop]["mae"] for r in ok if prop in r["properties"]])
            intrinsic[arm][prop] = {"r_mean": rs["mean"], "mae_mean": ms["mean"], "n": rs["n"]}

    return {"arms": arms, "extrinsic": extrinsic, "verdicts": verdicts, "intrinsic": intrinsic}


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_report(aggregate: dict) -> str:
    arms = aggregate["arms"]
    lines: list[str] = ["# Quantum-Teacher Validation Report", "", "## Verdicts", ""]

    for v in aggregate["verdicts"]:
        lines += [
            f"### {v['name']} ({v['reference']} vs {v['treatment']})",
            "",
            f"- Decisive probe: `{v['method']}` {v['metric'].upper()}",
            f"- delta ({v['reference']} - {v['treatment']}): {_fmt(v['delta'])}",
            f"- combined seed std: {_fmt(v['combined_std'])}",
            f"- **Result: {v['result']}**",
            "",
        ]
    lines += ["_Heuristic, not a significance test: 'helps' iff delta > combined std._", ""]

    lines += ["## Extrinsic (ESOL transfer)", "",
              "| Method | Arm | MAE (mean) | MAE (std) | R2 (mean) | n |",
              "|---|---|---|---|---|---|"]
    for method, arm_map in aggregate["extrinsic"].items():
        for arm in arms:
            s = arm_map[arm]
            lines.append(
                f"| {method} | {arm} | {_fmt(s['mae_mean'])} | {_fmt(s['mae_std'])} "
                f"| {_fmt(s['r2_mean'])} | {s['n']} |"
            )
    lines.append("")

    lines += ["## Intrinsic (teacher on held-out conformers)", "",
              "| Property | Arm | r (mean) | MAE (mean) | n |",
              "|---|---|---|---|---|"]
    for arm in arms:
        for prop in INTRINSIC_PROPERTIES:
            s = aggregate["intrinsic"][arm][prop]
            lines.append(
                f"| {prop} | {arm} | {_fmt(s['r_mean'])} | {_fmt(s['mae_mean'])} | {s['n']} |"
            )
    lines.append("")
    return "\n".join(lines)


def _save_backbone(path: Path, pretrain_ds, result, pretrain_cfg) -> None:
    node_targets = int(pretrain_ds.examples[0].node_target.shape[-1])
    model_config = {
        "atom_vocab_size": 128,
        "bond_vocab_size": 8,
        "hidden_dim": pretrain_cfg["hidden_dim"],
        "num_message_passing_steps": pretrain_cfg["message_passing_steps"],
        "node_targets": node_targets,
        "graph_targets": 2,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_config": model_config, "model_state_dict": result.model.state_dict()},
        path,
    )


def _pretrain_kwargs(pretrain_cfg: dict, arm_overrides: dict, seed: int) -> dict:
    kwargs = dict(
        hidden_dim=pretrain_cfg["hidden_dim"],
        num_message_passing_steps=pretrain_cfg["message_passing_steps"],
        hidden_dim_3d=pretrain_cfg.get("hidden_dim_3d", pretrain_cfg["hidden_dim"]),
        num_rbf=pretrain_cfg.get("num_rbf", 16),
        cutoff=pretrain_cfg.get("cutoff", 5.0),
        num_message_passing_steps_3d=pretrain_cfg.get(
            "message_passing_steps_3d", pretrain_cfg["message_passing_steps"]
        ),
        epochs=pretrain_cfg["epochs"],
        batch_size=pretrain_cfg.get("batch_size", 16),
        learning_rate=pretrain_cfg["learning_rate"],
        supervised_weight=pretrain_cfg.get("supervised_weight", 1.0),
        contrastive_weight=pretrain_cfg.get("contrastive_weight", 1.0),
        temperature=pretrain_cfg.get("temperature", 0.1),
        seed=seed,
    )
    kwargs.update(arm_overrides)  # teacher_weight, conformer_pool_mode, energy_temperature
    return kwargs


def _run_probe(method: str, backbone_path: Path, adapt_cfg: dict, report_path: Path) -> dict:
    raw = {
        "command": "adapt",
        "method": method,
        "backbone": str(backbone_path),
        "task": adapt_cfg.get("task", "regression"),
        "dataset": adapt_cfg["dataset"],
        "adapter": adapt_cfg.get("adapter", {}),
        "training": adapt_cfg.get("training", {}),
        "split": adapt_cfg.get("split", {}),
        "outputs": {"report": str(report_path)},
    }
    summary = adapt_run(resolve_adapt_config(raw))
    return summary["test_metrics"]


def run_one_cell(
    arm_name, arm_overrides, seed, pretrain_ds, holdout_examples,
    pretrain_cfg, probes, adapt_cfg, out_dir, overwrite=False,
) -> dict:
    out_dir = Path(out_dir)
    backbone_path = out_dir / f"{arm_name}_s{seed}.pt"
    intrinsic_path = out_dir / f"{arm_name}_s{seed}_intrinsic.json"

    intrinsic_row = {"arm": arm_name, "seed": seed, "status": "ok", "properties": {}}
    extrinsic_rows: list[dict] = []

    cache_hit = backbone_path.exists() and intrinsic_path.exists() and not overwrite
    if cache_hit:
        intrinsic_row["properties"] = json.loads(intrinsic_path.read_text())
    else:
        checkpoint_path = out_dir / f"{arm_name}_s{seed}.ckpt.pt"
        resume = bool(pretrain_cfg.get("resume", False)) and not overwrite
        if overwrite and checkpoint_path.exists():
            checkpoint_path.unlink()
        try:
            result = contrastive_pretrain_on_dataset(
                pretrain_ds,
                checkpoint_path=checkpoint_path,
                checkpoint_every=int(pretrain_cfg.get("checkpoint_every", 10)),
                resume=resume,
                **_pretrain_kwargs(pretrain_cfg, arm_overrides, seed),
            )
        except CheckpointMismatchError:
            raise  # config error affects every cell — surface it, don't bury it
        except Exception as exc:  # noqa: BLE001 - per-cell isolation
            return _failed_cell(arm_name, seed, probes, f"pretrain failed: {exc}")

        if not result.loss_history or not math.isfinite(result.loss_history[-1]):
            return _failed_cell(arm_name, seed, probes, "non-finite pretraining loss")
        if not all(torch.isfinite(p).all() for p in result.model.parameters()):
            return _failed_cell(arm_name, seed, probes, "non-finite backbone weights")

        _save_backbone(backbone_path, pretrain_ds, result, pretrain_cfg)
        try:
            props = evaluate_teacher(result.teacher, result.encoder3d, holdout_examples)
        except Exception as exc:  # noqa: BLE001
            props = {}
            intrinsic_row["status"] = "failed"
            intrinsic_row["error"] = str(exc)
        intrinsic_row["properties"] = props
        intrinsic_path.parent.mkdir(parents=True, exist_ok=True)
        intrinsic_path.write_text(json.dumps(props, indent=2))
        if intrinsic_row["status"] == "ok" and checkpoint_path.exists():
            checkpoint_path.unlink()

    for probe in probes:
        method = probe["method"]
        report_path = out_dir / f"{arm_name}_s{seed}_{method}.json"
        try:
            if report_path.exists() and not overwrite:
                cached = json.loads(report_path.read_text())
                if "test_metrics" in cached:
                    metrics = cached["test_metrics"]
                else:
                    metrics = cached
            else:
                metrics = _run_probe(method, backbone_path, adapt_cfg, report_path)
            extrinsic_rows.append({
                "arm": arm_name, "seed": seed, "method": method, "status": "ok",
                "mae": float(metrics["mae"]), "r2": float(metrics["r2"]),
            })
        except Exception as exc:  # noqa: BLE001
            extrinsic_rows.append({
                "arm": arm_name, "seed": seed, "method": method, "status": "failed",
                "mae": None, "r2": None, "error": str(exc),
            })

    return {"extrinsic": extrinsic_rows, "intrinsic": intrinsic_row}


def _failed_cell(arm_name, seed, probes, message: str) -> dict:
    return {
        "extrinsic": [
            {"arm": arm_name, "seed": seed, "method": p["method"], "status": "failed",
             "mae": None, "r2": None, "error": message}
            for p in probes
        ],
        "intrinsic": {"arm": arm_name, "seed": seed, "status": "failed",
                      "properties": {}, "error": message},
    }


def _load_dataset(pretrain_cfg: dict):
    if "cache_dir" in pretrain_cfg:
        from .shard_cache import load_compact_shards

        return load_compact_shards(
            pretrain_cfg["cache_dir"],
            list(pretrain_cfg["subset_ids"]),
        )

    from .quantum_data import load_quantum_zinc_subset_range

    return load_quantum_zinc_subset_range(
        pretrain_cfg["dataset_root"],
        subset_ids=list(pretrain_cfg["subset_ids"]),
        limit_per_shard=pretrain_cfg.get("limit_per_shard", 400),
        results_path=pretrain_cfg.get("results"),
        use_results=True,
    )


def run_validation(cfg: dict, dataset=None, overwrite=False) -> dict:
    if dataset is None:
        dataset = _load_dataset(cfg["pretrain"])

    holdout_cfg = cfg["holdout"]
    if "k" in holdout_cfg:
        pretrain_ds, holdout = scaffold_hash_holdout(dataset, k=holdout_cfg["k"])
    else:
        pretrain_ds, holdout = split_holdout(
            dataset, fraction=holdout_cfg["fraction"], seed=holdout_cfg["seed"]
        )

    out_dir = Path(cfg["outputs"]["dir"])
    extrinsic_rows: list[dict] = []
    intrinsic_rows: list[dict] = []
    for arm_name in cfg["arms"]:
        arm_overrides = cfg["arms"][arm_name]
        for seed in cfg["seeds"]:
            cell = run_one_cell(
                arm_name, arm_overrides, seed, pretrain_ds, holdout.examples,
                cfg["pretrain"], cfg["probes"], cfg["adapt"], out_dir, overwrite=overwrite,
            )
            extrinsic_rows.extend(cell["extrinsic"])
            intrinsic_rows.append(cell["intrinsic"])

    aggregate = aggregate_results(
        extrinsic_rows, intrinsic_rows,
        arms=list(cfg["arms"].keys()),
        comparisons=cfg.get("comparisons"),
    )

    report_base = Path(cfg["outputs"]["report"])
    report_base.parent.mkdir(parents=True, exist_ok=True)
    report_base.with_suffix(".json").write_text(json.dumps(
        {"rows": {"extrinsic": extrinsic_rows, "intrinsic": intrinsic_rows},
         "aggregate": aggregate}, indent=2))
    report_base.with_suffix(".md").write_text(render_report(aggregate))
    return aggregate


def load_validation_config(path) -> dict:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text())
    required = ("pretrain", "arms", "seeds", "holdout", "probes", "adapt", "outputs")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"validation config missing keys: {missing}")
    return cfg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Quantum-teacher validation harness")
    parser.add_argument("--config", required=True, help="path to validation YAML")
    parser.add_argument("--overwrite", action="store_true", help="ignore cached artifacts")
    args = parser.parse_args(argv)
    cfg = load_validation_config(args.config)
    aggregate = run_validation(cfg, overwrite=args.overwrite)
    print(render_report(aggregate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
