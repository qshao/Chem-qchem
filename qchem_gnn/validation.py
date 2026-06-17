from __future__ import annotations

import torch

from .conformer import ConformerEncoderBatch
from .minimal import MinimalQuantumDataset
from .teacher_heads import assemble_conformer_targets


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
