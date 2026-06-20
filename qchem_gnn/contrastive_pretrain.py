from __future__ import annotations

import dataclasses
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .boltzmann import boltzmann_average
from .conformer import ConformerEncoderBatch
from .encoder3d import Conformer3DEncoder
from .graph import GraphBatch
from .losses import compute_multitask_loss, info_nce_contrastive_loss, vicreg_loss
from .minimal import MinimalQuantumDataset
from .model import MolecularQuantumGNN
from .quantum_data import compute_target_normalization, normalize_targets
from .teacher_heads import QuantumTeacherHeads, assemble_conformer_targets, teacher_loss
from .eval import scaffold_key_from_smiles, scaffold_mask_from_keys


PRETRAIN_CHECKPOINT_VERSION = 2

_FINGERPRINT_FIELDS = (
    "hidden_dim",
    "num_message_passing_steps",
    "hidden_dim_3d",
    "num_rbf",
    "cutoff",
    "num_message_passing_steps_3d",
    "batch_size",
    "learning_rate",
    "supervised_weight",
    "contrastive_weight",
    "teacher_weight",
    "temperature",
    "energy_temperature",
    "conformer_pool_mode",
    "contrastive_loss",
    "vicreg_sim_weight",
    "vicreg_var_weight",
    "vicreg_cov_weight",
    "use_scaffold_negmask",
    "seed",
    "node_targets",
    "num_examples",
)


class CheckpointMismatchError(RuntimeError):
    """Raised when a resume checkpoint is unreadable or disagrees with the run."""


def _build_fingerprint(params: dict) -> dict:
    return {key: params[key] for key in _FINGERPRINT_FIELDS}


def _validate_fingerprint(saved: dict, current: dict) -> None:
    differing = [k for k in _FINGERPRINT_FIELDS if saved.get(k) != current.get(k)]
    if differing:
        details = ", ".join(
            f"{k}: checkpoint={saved.get(k)!r} != config={current.get(k)!r}"
            for k in differing
        )
        raise CheckpointMismatchError(
            f"cannot resume: config differs from checkpoint ({details}). "
            "Use overwrite to restart from scratch."
        )


def _atomic_save_checkpoint(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_checkpoint(path: Path) -> dict:
    try:
        payload = torch.load(Path(path), weights_only=False)
    except Exception as exc:  # noqa: BLE001 - surface as a clear resume error
        raise CheckpointMismatchError(
            f"could not read checkpoint at {path}: {exc}. "
            "Use overwrite to restart from scratch."
        ) from exc
    if not isinstance(payload, dict) or payload.get("version") != PRETRAIN_CHECKPOINT_VERSION:
        got = payload.get("version") if isinstance(payload, dict) else None
        raise CheckpointMismatchError(
            f"checkpoint version mismatch at {path}: expected "
            f"{PRETRAIN_CHECKPOINT_VERSION}, got {got}. "
            "Use overwrite to restart from scratch."
        )
    return payload


def _append_metrics_line(path: Path, record: dict) -> None:
    """Append one JSON record to the metrics file, flushed. Never fatal."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
    except OSError as exc:  # logging must never crash training
        warnings.warn(f"could not write metrics to {path}: {exc}", stacklevel=2)


def _example_scaffold_key(example) -> int:
    key = getattr(example, "scaffold_key", None)
    if key is not None:
        return key
    return scaffold_key_from_smiles(example.smiles)


def _boltzmann_pool_molecules(
    conformer_embeddings: torch.Tensor,
    conformer_molecule_index: torch.Tensor,
    conformer_energy: torch.Tensor | None,
    num_molecules: int,
    temperature: float,
    mode: str,
) -> torch.Tensor:
    pooled = []
    for molecule_id in range(num_molecules):
        mask = conformer_molecule_index == molecule_id
        embeddings = conformer_embeddings[mask]
        if mode == "energy" and conformer_energy is not None:
            pooled.append(
                boltzmann_average(embeddings, conformer_energy[mask], temperature)
            )
        else:
            pooled.append(embeddings.mean(dim=0))
    return torch.stack(pooled, dim=0)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class ContrastivePretrainingResult:
    model: MolecularQuantumGNN
    loss_history: list[float]
    contrastive_loss_history: list[float]
    embeddings: torch.Tensor
    target_normalization: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, object]
    global_step: int
    teacher: nn.Module | None = None
    encoder3d: nn.Module | None = None


def _supervised_loss_for_batch(model_output, examples, normalization) -> torch.Tensor:
    dev = normalization["node_mean"].device
    node_target = torch.cat([example.node_target for example in examples], dim=0).to(dev)
    edge_target = torch.cat([example.edge_target for example in examples], dim=0).to(dev)
    graph_target = torch.stack([example.graph_target for example in examples], dim=0).to(dev)
    node_target, edge_target, graph_target = normalize_targets(
        node_target, edge_target, graph_target, normalization
    )
    return compute_multitask_loss(model_output, (node_target, edge_target, graph_target))


def contrastive_pretrain_on_dataset(
    dataset: MinimalQuantumDataset,
    *,
    hidden_dim: int = 32,
    num_message_passing_steps: int = 2,
    hidden_dim_3d: int = 32,
    num_rbf: int = 16,
    cutoff: float = 5.0,
    num_message_passing_steps_3d: int = 2,
    total_steps: int = 10000,
    batch_size: int = 8,
    log_every: int = 100,
    val_dataset: MinimalQuantumDataset | None = None,
    metrics_path: Path | None = None,
    learning_rate: float = 0.01,
    supervised_weight: float = 1.0,
    contrastive_weight: float = 1.0,
    temperature: float = 0.1,
    teacher_weight: float = 1.0,
    energy_temperature: float = 298.15,
    conformer_pool_mode: str = "mean",
    contrastive_loss: str = "infonce",
    vicreg_sim_weight: float = 25.0,
    vicreg_var_weight: float = 25.0,
    vicreg_cov_weight: float = 1.0,
    use_scaffold_negmask: bool = False,
    seed: int = 0,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 1000,
    resume: bool = False,
) -> ContrastivePretrainingResult:
    examples = dataset.examples
    normalization = compute_target_normalization(dataset)
    num_examples = len(examples)
    node_targets = int(examples[0].node_target.shape[-1])

    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    metrics_path = Path(metrics_path) if metrics_path is not None else None
    do_resume = resume and checkpoint_path is not None and checkpoint_path.exists()

    if not do_resume:
        torch.manual_seed(seed)
        if resume and checkpoint_path is not None:
            warnings.warn(
                f"resume requested but no checkpoint at {checkpoint_path}; "
                "starting fresh",
                stacklevel=2,
            )

    model = MolecularQuantumGNN(
        atom_vocab_size=128,
        bond_vocab_size=8,
        hidden_dim=hidden_dim,
        num_message_passing_steps=num_message_passing_steps,
        node_targets=node_targets,
        graph_targets=2,
    )
    encoder3d = Conformer3DEncoder(
        atom_vocab_size=128,
        hidden_dim=hidden_dim_3d,
        num_rbf=num_rbf,
        cutoff=cutoff,
        num_message_passing_steps=num_message_passing_steps_3d,
    )
    teacher = QuantumTeacherHeads(hidden_dim=hidden_dim_3d)
    proj_2d = ProjectionHead(hidden_dim, hidden_dim)
    proj_3d = ProjectionHead(hidden_dim_3d, hidden_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoder3d.to(device)
    teacher.to(device)
    proj_2d.to(device)
    proj_3d.to(device)
    normalization = {k: v.to(device) for k, v in normalization.items()}

    params = list(model.parameters()) + list(encoder3d.parameters())
    params += list(teacher.parameters())
    params += list(proj_2d.parameters()) + list(proj_3d.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    current_fingerprint = _build_fingerprint(
        {
            "hidden_dim": hidden_dim,
            "num_message_passing_steps": num_message_passing_steps,
            "hidden_dim_3d": hidden_dim_3d,
            "num_rbf": num_rbf,
            "cutoff": cutoff,
            "num_message_passing_steps_3d": num_message_passing_steps_3d,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "supervised_weight": supervised_weight,
            "contrastive_weight": contrastive_weight,
            "teacher_weight": teacher_weight,
            "temperature": temperature,
            "energy_temperature": energy_temperature,
            "conformer_pool_mode": conformer_pool_mode,
            "contrastive_loss": contrastive_loss,
            "vicreg_sim_weight": vicreg_sim_weight,
            "vicreg_var_weight": vicreg_var_weight,
            "vicreg_cov_weight": vicreg_cov_weight,
            "use_scaffold_negmask": use_scaffold_negmask,
            "seed": seed,
            "node_targets": node_targets,
            "num_examples": num_examples,
        }
    )

    # Shared forward pass for both train and validation steps.
    def _batch_forward(batch_examples: list) -> dict[str, torch.Tensor]:
        graph_batch = GraphBatch.from_graphs([ex.graph for ex in batch_examples])
        graph_batch = dataclasses.replace(
            graph_batch,
            atomic_numbers=graph_batch.atomic_numbers.to(device),
            edge_index=graph_batch.edge_index.to(device),
            edge_attr=graph_batch.edge_attr.to(device),
            batch=graph_batch.batch.to(device),
            ptr=graph_batch.ptr.to(device),
        )
        model_output = model(graph_batch)
        supervised = _supervised_loss_for_batch(model_output, batch_examples, normalization)

        contrastive = torch.zeros((), dtype=supervised.dtype, device=supervised.device)
        teacher_term = torch.zeros((), dtype=supervised.dtype, device=supervised.device)
        usable = [
            (pos, ex)
            for pos, ex in enumerate(batch_examples)
            if ex.conformer_coords and ex.conformer_node_targets is not None
        ]
        if len(usable) >= 2:
            coords_index = [pos for pos, _ in usable]
            usable_examples = [ex for _, ex in usable]
            batch_mask: torch.Tensor | None = None
            if use_scaffold_negmask:
                keys = [_example_scaffold_key(ex) for ex in usable_examples]
                batch_mask = scaffold_mask_from_keys(keys).to(supervised.device)
                if batch_mask.all(dim=1).any():
                    warnings.warn(
                        "scaffold negmask: at least one molecule has all "
                        "negatives masked in this batch",
                        stacklevel=2,
                    )
            conformer_batch = ConformerEncoderBatch.from_molecule_conformers(
                [ex.graph for ex in usable_examples],
                [ex.conformer_coords for ex in usable_examples],
                conformer_energies=[ex.conformer_energies for ex in usable_examples],
            )
            conformer_batch = dataclasses.replace(
                conformer_batch,
                atomic_numbers=conformer_batch.atomic_numbers.to(device),
                edge_index=conformer_batch.edge_index.to(device),
                positions=conformer_batch.positions.to(device),
                node_conformer_index=conformer_batch.node_conformer_index.to(device),
                conformer_molecule_index=conformer_batch.conformer_molecule_index.to(device),
                conformer_energy=conformer_batch.conformer_energy.to(device) if conformer_batch.conformer_energy is not None else None,
            )
            node_states_3d, conformer_embeddings = encoder3d.forward_with_nodes(
                conformer_batch.atomic_numbers,
                conformer_batch.edge_index,
                conformer_batch.positions,
                conformer_batch.node_conformer_index,
                conformer_batch.num_conformers,
            )

            if teacher_weight:
                node_pred, edge_pred, graph_pred = teacher(
                    node_states_3d, conformer_batch.edge_index, conformer_embeddings
                )
                node_t, edge_t, graph_t, _ = assemble_conformer_targets(usable_examples)
                node_t = node_t.to(device)
                edge_t = edge_t.to(device)
                graph_t = graph_t.to(device)
                node_t, edge_t, graph_t = normalize_targets(
                    node_t, edge_t, graph_t, normalization
                )
                teacher_term = teacher_loss(
                    node_pred, edge_pred, graph_pred, node_t, edge_t, graph_t
                )

            if contrastive_weight:
                molecule_3d = _boltzmann_pool_molecules(
                    conformer_embeddings,
                    conformer_batch.conformer_molecule_index,
                    conformer_batch.conformer_energy,
                    conformer_batch.num_molecules,
                    temperature=energy_temperature,
                    mode=conformer_pool_mode,
                )
                molecule_2d = model_output.mol_embedding[coords_index]
                view_2d = proj_2d(molecule_2d)
                view_3d = proj_3d(molecule_3d)
                if contrastive_loss == "vicreg":
                    contrastive = vicreg_loss(
                        view_2d,
                        view_3d,
                        sim_weight=vicreg_sim_weight,
                        var_weight=vicreg_var_weight,
                        cov_weight=vicreg_cov_weight,
                    )
                elif contrastive_loss == "infonce":
                    contrastive = info_nce_contrastive_loss(
                        view_2d, view_3d, temperature=temperature,
                        negative_mask=batch_mask,
                    )
                else:
                    raise ValueError(f"unknown contrastive_loss: {contrastive_loss!r}")

        total = (
            supervised_weight * supervised
            + contrastive_weight * contrastive
            + teacher_weight * teacher_term
        )
        return {
            "total": total,
            "supervised": supervised,
            "contrastive": contrastive,
            "teacher": teacher_term,
        }

    loss_history: list[float] = []
    contrastive_loss_history: list[float] = []
    start_step = 0
    cycle_order = torch.randperm(num_examples)
    cycle_start = 0

    if do_resume:
        payload = _load_checkpoint(checkpoint_path)
        _validate_fingerprint(payload["config_fingerprint"], current_fingerprint)
        model.load_state_dict(payload["model_state_dict"])
        encoder3d.load_state_dict(payload["encoder3d_state_dict"])
        teacher.load_state_dict(payload["teacher_state_dict"])
        proj_2d.load_state_dict(payload["proj_2d_state_dict"])
        proj_3d.load_state_dict(payload["proj_3d_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        torch.set_rng_state(payload["rng_state"])
        loss_history = list(payload["loss_history"])
        contrastive_loss_history = list(payload["contrastive_loss_history"])
        start_step = int(payload["step"])
        cycle_order = torch.tensor(payload["cycle_order"])
        cycle_start = int(payload["cycle_start"])
        if total_steps <= start_step:
            warnings.warn(
                f"resume: checkpoint step ({start_step}) >= requested total_steps "
                f"({total_steps}); no training will run.",
                stacklevel=2,
            )

    def _write_checkpoint(completed_steps: int, co: torch.Tensor, cs: int) -> None:
        _atomic_save_checkpoint(
            checkpoint_path,
            {
                "version": PRETRAIN_CHECKPOINT_VERSION,
                "step": completed_steps,
                "cycle_order": co.tolist(),
                "cycle_start": cs,
                "model_state_dict": model.state_dict(),
                "encoder3d_state_dict": encoder3d.state_dict(),
                "teacher_state_dict": teacher.state_dict(),
                "proj_2d_state_dict": proj_2d.state_dict(),
                "proj_3d_state_dict": proj_3d.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "loss_history": list(loss_history),
                "contrastive_loss_history": list(contrastive_loss_history),
                "config_fingerprint": current_fingerprint,
            },
        )

    _TERMS = ("total", "supervised", "contrastive", "teacher")
    step = start_step
    recent = {term: 0.0 for term in _TERMS}
    recent_n = 0
    val_examples = val_dataset.examples if val_dataset is not None else []
    _modules = (model, encoder3d, teacher, proj_2d, proj_3d)
    train_start = time.perf_counter()

    _resume_note = f" (resuming from step {start_step})" if do_resume else ""
    _val_note = f", val={len(val_examples)}" if val_examples else ""
    print(
        f"Training start{_resume_note}: "
        f"train={num_examples} molecules{_val_note} | "
        f"steps={start_step}→{total_steps} | "
        f"batch={batch_size} | "
        f"log_every={log_every} | "
        f"loss={contrastive_loss} | "
        f"device={device} | "
        f"seed={seed}"
    )
    if checkpoint_path is not None:
        print(f"  checkpoint → {checkpoint_path}  (every {checkpoint_every} steps)")
    if metrics_path is not None:
        print(f"  metrics    → {metrics_path}")
    print()

    while step < total_steps:
        if cycle_start >= num_examples:
            cycle_start = 0
            cycle_order = torch.randperm(num_examples)

        end = min(cycle_start + batch_size, num_examples)
        batch_indices = cycle_order[cycle_start:end].tolist()
        cycle_start = end
        batch_examples = [examples[i] for i in batch_indices]

        out = _batch_forward(batch_examples)
        total_loss = out["total"]
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        step += 1
        loss_history.append(float(total_loss.item()))
        contrastive_loss_history.append(float(out["contrastive"].item()))
        for term in _TERMS:
            recent[term] += float(out[term].item())
        recent_n += 1

        if step % log_every == 0 or step == total_steps:
            train_avg = {term: recent[term] / recent_n for term in _TERMS}
            recent = {term: 0.0 for term in _TERMS}
            recent_n = 0

            val_avg = None
            if val_examples:
                for mod in _modules:
                    mod.eval()
                val_sum = {term: 0.0 for term in _TERMS}
                val_n = 0
                with torch.no_grad():
                    for vstart in range(0, len(val_examples), batch_size):
                        vout = _batch_forward(val_examples[vstart : vstart + batch_size])
                        for term in _TERMS:
                            val_sum[term] += float(vout[term].item())
                        val_n += 1
                for mod in _modules:
                    mod.train()
                val_avg = {term: val_sum[term] / max(val_n, 1) for term in _TERMS}

            line = (
                f"step {step:6d}/{total_steps}"
                f" | loss {train_avg['total']:.4f}"
                f" (sup {train_avg['supervised']:.4f}"
                f" / con {train_avg['contrastive']:.4f}"
                f" / tea {train_avg['teacher']:.4f})"
            )
            if val_avg is not None:
                line += f" | val {val_avg['total']:.4f}"
            print(line)

            if metrics_path is not None:
                wall = time.perf_counter() - train_start
                completed = step - start_step
                record = {
                    "step": step,
                    "total_steps": total_steps,
                    "train_loss": train_avg["total"],
                    "train_supervised": train_avg["supervised"],
                    "train_contrastive": train_avg["contrastive"],
                    "train_teacher": train_avg["teacher"],
                    "steps_per_sec": completed / wall if wall > 0 else 0.0,
                    "wall_seconds": wall,
                }
                if val_avg is not None:
                    record["val_loss"] = val_avg["total"]
                    record["val_supervised"] = val_avg["supervised"]
                    record["val_contrastive"] = val_avg["contrastive"]
                    record["val_teacher"] = val_avg["teacher"]
                _append_metrics_line(metrics_path, record)

        if checkpoint_path is not None and (
            step % checkpoint_every == 0 or step == total_steps
        ):
            _write_checkpoint(step, cycle_order, cycle_start)

    model.eval()
    with torch.no_grad():
        full_batch = GraphBatch.from_graphs([ex.graph for ex in examples])
        full_batch = dataclasses.replace(
            full_batch,
            atomic_numbers=full_batch.atomic_numbers.to(device),
            edge_index=full_batch.edge_index.to(device),
            edge_attr=full_batch.edge_attr.to(device),
            batch=full_batch.batch.to(device),
            ptr=full_batch.ptr.to(device),
        )
        embeddings = model.encode_graph_embeddings(full_batch)
    model.train()

    return ContrastivePretrainingResult(
        model=model,
        loss_history=loss_history,
        contrastive_loss_history=contrastive_loss_history,
        embeddings=embeddings,
        target_normalization=normalization,
        optimizer_state_dict=optimizer.state_dict(),
        global_step=step,
        teacher=teacher,
        encoder3d=encoder3d,
    )


def run_contrastive_ablation(
    dataset: MinimalQuantumDataset,
    *,
    hidden_dim: int = 32,
    total_steps: int = 200,
    batch_size: int = 8,
    contrastive_weight: float = 1.0,
    seed: int = 0,
) -> dict[str, dict]:
    from .eval import run_linear_probe
    from .splits import scaffold_or_random_split
    labels = np.stack(
        [example.graph_target.detach().cpu().numpy() for example in dataset.examples], axis=0
    )
    split = scaffold_or_random_split(
        [example.mol_id for example in dataset.examples], seed=seed
    )

    report: dict[str, dict] = {}
    for arm_name, weight in (("supervised_only", 0.0), ("with_contrastive", contrastive_weight)):
        result = contrastive_pretrain_on_dataset(
            dataset,
            hidden_dim=hidden_dim,
            total_steps=total_steps,
            batch_size=batch_size,
            contrastive_weight=weight,
            seed=seed,
        )
        embeddings = result.embeddings.detach().cpu().numpy()
        report[arm_name] = run_linear_probe(embeddings, labels, split)
    return report
