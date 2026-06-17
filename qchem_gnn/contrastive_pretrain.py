from __future__ import annotations

from dataclasses import dataclass

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
    epoch: int
    global_step: int
    teacher: nn.Module | None = None
    encoder3d: nn.Module | None = None


def _supervised_loss_for_batch(model_output, examples, normalization) -> torch.Tensor:
    node_target = torch.cat([example.node_target for example in examples], dim=0)
    edge_target = torch.cat([example.edge_target for example in examples], dim=0)
    graph_target = torch.stack([example.graph_target for example in examples], dim=0)
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
    epochs: int = 200,
    batch_size: int = 8,
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
    seed: int = 0,
) -> ContrastivePretrainingResult:
    torch.manual_seed(seed)
    examples = dataset.examples
    normalization = compute_target_normalization(dataset)

    node_targets = int(examples[0].node_target.shape[-1])
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

    params = list(model.parameters()) + list(encoder3d.parameters())
    params += list(teacher.parameters())
    params += list(proj_2d.parameters()) + list(proj_3d.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    loss_history: list[float] = []
    contrastive_loss_history: list[float] = []
    num_examples = len(examples)

    for _ in range(epochs):
        order = torch.randperm(num_examples)
        epoch_total = 0.0
        epoch_contrastive = 0.0
        num_batches = 0

        for start in range(0, num_examples, batch_size):
            batch_indices = order[start : start + batch_size].tolist()
            batch_examples = [examples[i] for i in batch_indices]

            graph_batch = GraphBatch.from_graphs([ex.graph for ex in batch_examples])
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
                conformer_batch = ConformerEncoderBatch.from_molecule_conformers(
                    [ex.graph for ex in usable_examples],
                    [ex.conformer_coords for ex in usable_examples],
                    conformer_energies=[ex.conformer_energies for ex in usable_examples],
                )
                node_states_3d, conformer_embeddings = encoder3d.forward_with_nodes(
                    conformer_batch.atomic_numbers,
                    conformer_batch.edge_index,
                    conformer_batch.positions,
                    conformer_batch.node_conformer_index,
                    conformer_batch.num_conformers,
                )

                # Teacher: per-conformer quantum regression.
                if teacher_weight:
                    node_pred, edge_pred, graph_pred = teacher(
                        node_states_3d, conformer_batch.edge_index, conformer_embeddings
                    )
                    node_t, edge_t, graph_t, _ = assemble_conformer_targets(usable_examples)
                    teacher_term = teacher_loss(
                        node_pred, edge_pred, graph_pred, node_t, edge_t, graph_t
                    )

                # Contrastive: align 2D molecule embedding to Boltzmann-pooled 3D embedding.
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
                    else:
                        contrastive = info_nce_contrastive_loss(
                            view_2d, view_3d, temperature=temperature
                        )

            total = (
                supervised_weight * supervised
                + contrastive_weight * contrastive
                + teacher_weight * teacher_term
            )
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            epoch_total += float(total.item())
            epoch_contrastive += float(contrastive.item())
            num_batches += 1

        loss_history.append(epoch_total / max(num_batches, 1))
        contrastive_loss_history.append(epoch_contrastive / max(num_batches, 1))

    model.eval()
    with torch.no_grad():
        full_batch = GraphBatch.from_graphs([ex.graph for ex in examples])
        embeddings = model.encode_graph_embeddings(full_batch)
    model.train()

    return ContrastivePretrainingResult(
        model=model,
        loss_history=loss_history,
        contrastive_loss_history=contrastive_loss_history,
        embeddings=embeddings,
        target_normalization=normalization,
        optimizer_state_dict=optimizer.state_dict(),
        epoch=epochs,
        global_step=epochs * ((num_examples + batch_size - 1) // batch_size),
        teacher=teacher,
        encoder3d=encoder3d,
    )


def run_contrastive_ablation(
    dataset: MinimalQuantumDataset,
    *,
    hidden_dim: int = 32,
    epochs: int = 200,
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
            epochs=epochs,
            batch_size=batch_size,
            contrastive_weight=weight,
            seed=seed,
        )
        embeddings = result.embeddings.detach().cpu().numpy()
        report[arm_name] = run_linear_probe(embeddings, labels, split)
    return report
