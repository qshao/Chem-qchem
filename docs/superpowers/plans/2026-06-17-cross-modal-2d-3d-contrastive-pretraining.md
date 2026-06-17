# Cross-Modal 2D↔3D Contrastive Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distill the existing 3D/quantum view (conformer geometries + DFT outputs) into the 2D GNN encoder via a cross-modal contrastive objective, so exported 2D embeddings transfer better downstream while inference stays 2D-only.

**Architecture:** A training-only 3D "teacher" encoder consumes conformer coordinates (featurized as rotation-invariant interatomic distances via a Gaussian radial basis) and produces a molecule-level 3D embedding (energy-weighted over the conformer ensemble). A symmetric InfoNCE loss aligns each molecule's 2D embedding with its 3D embedding using in-batch negatives. The existing supervised quantum-regression loss remains the anchor: `total = w_sup·supervised + w_con·contrastive`. At inference, the 3D encoder and projection heads are dropped; `MolecularQuantumGNN.encode_graph_embeddings` is unchanged.

**Tech Stack:** Python 3.13+, PyTorch, RDKit, NumPy/Pandas, PyYAML, pytest. No new third-party dependencies.

## Global Constraints

- Python 3.13+; no new third-party dependencies beyond the existing set (numpy, pandas, torch, rdkit, pyyaml, pytest, optional h5py).
- Inference must remain 2D-only: do NOT change the `MolecularQuantumGNN` interface or the `export-embeddings`/`eval` code paths. Checkpoints saved by the new command must load with the existing `_load_model_from_checkpoint`.
- `atom_vocab_size=128`, `bond_vocab_size=8` are the fixed vocab sizes used throughout the codebase. Atomic numbers index the atom embedding directly.
- Follow existing module style: `from __future__ import annotations`, frozen dataclasses, `nn.SiLU` activations, `nn.LayerNorm` residual blocks, MSE for regression.
- Conformer coordinates in the geometry pickles are ordered to match the RDKit atom order produced by `build_graph_from_smiles(smiles, add_hs=True)` (heavy atoms + explicit H). The 3D encoder reuses the molecule's existing `edge_index`; coordinates align with node indices.
- Tests live in `tests/`, run with `python -m pytest tests -q`. Match existing test conventions (synthetic SMILES like `"C"`, `"CC"`; `tmp_path` fixtures for IO).
- Commit after every task with a `feat:`/`test:`/`docs:` prefixed message ending with the project's `Co-Authored-By` trailer.

---

## File Structure

**New files:**
- `qchem_gnn/encoder3d.py` — `GaussianRBF`, `Residual3DMessagePassingBlock`, `Conformer3DEncoder` (training-only 3D teacher).
- `qchem_gnn/contrastive_pretrain.py` — `ProjectionHead`, `ContrastivePretrainingResult`, `contrastive_pretrain_on_dataset` (minibatched joint loop).
- `configs/minimal_contrastive_pretrain.yaml` — runnable minimal example.
- `tests/test_encoder3d.py`, `tests/test_contrastive_loss.py`, `tests/test_conformer_encoder_batch.py`, `tests/test_contrastive_pretrain.py`, `tests/test_contrastive_ablation.py`.

**Modified files:**
- `qchem_gnn/losses.py` — add `info_nce_contrastive_loss`.
- `qchem_gnn/conformer.py` — add `ConformerEncoderBatch` and `pool_conformers_to_molecules`.
- `qchem_gnn/quantum_data.py` — surface per-conformer coordinates and energies on `MinimalQuantumExample`.
- `qchem_gnn/minimal.py` — add `conformer_coords` / `conformer_energies` fields to `MinimalQuantumExample`.
- `qchem_gnn/config.py` — register the `contrastive-pretrain` command and `contrastive` config section.
- `qchem_gnn/cli.py` — add parser, config mapping, and `run_contrastive_pretrain`.

---

# Phase 1 — Pure model + loss units (synthetic tensors only)

### Task 1: Gaussian RBF distance expansion

**Files:**
- Create: `qchem_gnn/encoder3d.py`
- Test: `tests/test_encoder3d.py`

**Interfaces:**
- Produces: `GaussianRBF(num_rbf: int = 16, cutoff: float = 5.0)`; `forward(distances: Tensor[E]) -> Tensor[E, num_rbf]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_encoder3d.py
import torch

from qchem_gnn.encoder3d import GaussianRBF


def test_gaussian_rbf_shapes_and_range():
    rbf = GaussianRBF(num_rbf=8, cutoff=5.0)
    distances = torch.tensor([0.0, 1.0, 2.5, 5.0], dtype=torch.float32)
    expanded = rbf(distances)

    assert expanded.shape == (4, 8)
    assert torch.isfinite(expanded).all()
    # Gaussian activations are bounded by 1.0 (peak when distance equals a center).
    assert (expanded.max(dim=-1).values <= 1.0 + 1e-6).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_encoder3d.py::test_gaussian_rbf_shapes_and_range -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qchem_gnn.encoder3d'`.

- [ ] **Step 3: Write minimal implementation**

```python
# qchem_gnn/encoder3d.py
from __future__ import annotations

import torch
from torch import nn


class GaussianRBF(nn.Module):
    def __init__(self, num_rbf: int = 16, cutoff: float = 5.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        width = cutoff / max(num_rbf - 1, 1)
        self.coeff = -0.5 / (width ** 2)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(self.coeff * diff.pow(2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_encoder3d.py::test_gaussian_rbf_shapes_and_range -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/encoder3d.py tests/test_encoder3d.py
git commit -m "feat: add Gaussian RBF distance expansion for 3D encoder"
```

---

### Task 2: 3D conformer encoder with rotation invariance

**Files:**
- Modify: `qchem_gnn/encoder3d.py`
- Test: `tests/test_encoder3d.py`

**Interfaces:**
- Consumes: `GaussianRBF` (Task 1).
- Produces:
  - `Residual3DMessagePassingBlock(hidden_dim: int, num_rbf: int)`; `forward(node_states[N,H], edge_index[2,E], edge_rbf[E,num_rbf]) -> Tensor[N,H]`.
  - `Conformer3DEncoder(atom_vocab_size: int, hidden_dim: int = 64, num_rbf: int = 16, cutoff: float = 5.0, num_message_passing_steps: int = 3)`; `forward(atomic_numbers[N], edge_index[2,E], positions[N,3], node_conformer_index[N], num_conformers: int) -> Tensor[num_conformers, hidden_dim]`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_encoder3d.py
from qchem_gnn.encoder3d import Conformer3DEncoder


def _two_atom_conformer():
    atomic_numbers = torch.tensor([6, 8], dtype=torch.long)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float32)
    node_conformer_index = torch.tensor([0, 0], dtype=torch.long)
    return atomic_numbers, edge_index, positions, node_conformer_index


def test_conformer_encoder_output_shape():
    encoder = Conformer3DEncoder(atom_vocab_size=128, hidden_dim=16, num_rbf=8, num_message_passing_steps=2)
    atomic_numbers, edge_index, positions, node_conformer_index = _two_atom_conformer()

    embedding = encoder(atomic_numbers, edge_index, positions, node_conformer_index, num_conformers=1)

    assert embedding.shape == (1, 16)
    assert torch.isfinite(embedding).all()


def test_conformer_encoder_is_rotation_and_translation_invariant():
    encoder = Conformer3DEncoder(atom_vocab_size=128, hidden_dim=16, num_rbf=8, num_message_passing_steps=2)
    encoder.eval()
    atomic_numbers, edge_index, positions, node_conformer_index = _two_atom_conformer()

    theta = torch.tensor(0.7)
    rotation = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta), torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = torch.tensor([3.0, -2.0, 1.0])
    moved_positions = positions @ rotation.t() + translation

    with torch.no_grad():
        base = encoder(atomic_numbers, edge_index, positions, node_conformer_index, num_conformers=1)
        moved = encoder(atomic_numbers, edge_index, moved_positions, node_conformer_index, num_conformers=1)

    assert torch.allclose(base, moved, atol=1e-5)


def test_conformer_encoder_pools_multiple_conformers():
    encoder = Conformer3DEncoder(atom_vocab_size=128, hidden_dim=16, num_rbf=8, num_message_passing_steps=2)
    # Two conformers of the same 2-atom molecule, packed back to back.
    atomic_numbers = torch.tensor([6, 8, 6, 8], dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        dtype=torch.float32,
    )
    node_conformer_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    embeddings = encoder(atomic_numbers, edge_index, positions, node_conformer_index, num_conformers=2)

    assert embeddings.shape == (2, 16)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_encoder3d.py -v`
Expected: the three new tests FAIL with `ImportError: cannot import name 'Conformer3DEncoder'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to qchem_gnn/encoder3d.py
class Residual3DMessagePassingBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rbf: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        messages = self.message_mlp(torch.cat([node_states[src], edge_rbf], dim=-1))

        aggregated = torch.zeros_like(node_states)
        aggregated.index_add_(0, dst, messages)

        updated = self.update_mlp(torch.cat([node_states, aggregated], dim=-1))
        return self.norm(node_states + updated)


class Conformer3DEncoder(nn.Module):
    def __init__(
        self,
        atom_vocab_size: int,
        hidden_dim: int = 64,
        num_rbf: int = 16,
        cutoff: float = 5.0,
        num_message_passing_steps: int = 3,
    ):
        super().__init__()
        self.atom_encoder = nn.Embedding(atom_vocab_size, hidden_dim)
        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=cutoff)
        self.blocks = nn.ModuleList(
            Residual3DMessagePassingBlock(hidden_dim, num_rbf)
            for _ in range(num_message_passing_steps)
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        positions: torch.Tensor,
        node_conformer_index: torch.Tensor,
        num_conformers: int,
    ) -> torch.Tensor:
        node_states = self.atom_encoder(atomic_numbers)
        src, dst = edge_index
        distances = (positions[src] - positions[dst]).norm(dim=-1)
        edge_rbf = self.rbf(distances)

        for block in self.blocks:
            node_states = block(node_states, edge_index, edge_rbf)

        pooled = torch.zeros(
            num_conformers,
            node_states.shape[-1],
            dtype=node_states.dtype,
            device=node_states.device,
        )
        pooled.index_add_(0, node_conformer_index, node_states)
        counts = torch.bincount(node_conformer_index, minlength=num_conformers).clamp_min(1).unsqueeze(-1)
        return self.embedding_head(pooled / counts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_encoder3d.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/encoder3d.py tests/test_encoder3d.py
git commit -m "feat: add rotation-invariant 3D conformer teacher encoder"
```

---

### Task 3: Symmetric InfoNCE contrastive loss

**Files:**
- Modify: `qchem_gnn/losses.py`
- Test: `tests/test_contrastive_loss.py`

**Interfaces:**
- Produces: `info_nce_contrastive_loss(z_a: Tensor[B,D], z_b: Tensor[B,D], temperature: float = 0.1) -> Tensor[scalar]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_contrastive_loss.py
import torch

from qchem_gnn.losses import info_nce_contrastive_loss


def test_info_nce_returns_scalar_and_flows_gradient():
    z_a = torch.randn(8, 16, requires_grad=True)
    z_b = torch.randn(8, 16)

    loss = info_nce_contrastive_loss(z_a, z_b, temperature=0.1)

    assert loss.ndim == 0
    loss.backward()
    assert z_a.grad is not None
    assert torch.isfinite(z_a.grad).all()


def test_info_nce_is_symmetric():
    torch.manual_seed(0)
    z_a = torch.randn(6, 16)
    z_b = torch.randn(6, 16)

    forward = info_nce_contrastive_loss(z_a, z_b, temperature=0.2)
    swapped = info_nce_contrastive_loss(z_b, z_a, temperature=0.2)

    assert torch.allclose(forward, swapped, atol=1e-6)


def test_info_nce_rewards_aligned_pairs():
    base = torch.randn(8, 16)
    aligned = info_nce_contrastive_loss(base, base.clone(), temperature=0.1)
    shuffled = info_nce_contrastive_loss(base, base[torch.randperm(8)], temperature=0.1)

    assert aligned < shuffled
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_contrastive_loss.py -v`
Expected: FAIL with `ImportError: cannot import name 'info_nce_contrastive_loss'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to qchem_gnn/losses.py
def info_nce_contrastive_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    if z_a.shape != z_b.shape:
        raise ValueError("z_a and z_b must have the same shape")
    if z_a.shape[0] < 2:
        raise ValueError("contrastive loss needs at least 2 examples for in-batch negatives")

    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = (z_a @ z_b.t()) / temperature
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_ab + loss_ba)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_contrastive_loss.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/losses.py tests/test_contrastive_loss.py
git commit -m "feat: add symmetric InfoNCE contrastive loss"
```

---

### Task 4: Conformer batching + molecule-level pooling

**Files:**
- Modify: `qchem_gnn/conformer.py`
- Test: `tests/test_conformer_encoder_batch.py`

**Interfaces:**
- Consumes: `GraphData` (from `qchem_gnn.graph`), existing `pool_conformer_embeddings`.
- Produces:
  - `ConformerEncoderBatch` (frozen dataclass) with fields: `atomic_numbers[Ntot]`, `edge_index[2,Etot]`, `positions[Ntot,3]`, `node_conformer_index[Ntot]`, `conformer_molecule_index[C]`, `conformer_energy: Tensor[C] | None`, `num_conformers: int`, `num_molecules: int`.
  - `ConformerEncoderBatch.from_molecule_conformers(graphs: list[GraphData], conformer_coords: list[list[Tensor]], conformer_energies: list[Tensor] | None = None) -> ConformerEncoderBatch`.
  - `pool_conformers_to_molecules(conformer_embeddings: Tensor[C,H], conformer_molecule_index: Tensor[C], conformer_energy: Tensor[C] | None, num_molecules: int, mode: str = "mean") -> Tensor[num_molecules, H]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_conformer_encoder_batch.py
import torch

from qchem_gnn.conformer import ConformerEncoderBatch, pool_conformers_to_molecules
from qchem_gnn.graph import build_graph_from_smiles


def test_from_molecule_conformers_packs_offsets():
    graphs = [build_graph_from_smiles("C"), build_graph_from_smiles("CC")]
    coords = [
        [torch.zeros(graphs[0].num_nodes, 3), torch.ones(graphs[0].num_nodes, 3)],
        [torch.zeros(graphs[1].num_nodes, 3)],
    ]

    batch = ConformerEncoderBatch.from_molecule_conformers(graphs, coords)

    assert batch.num_molecules == 2
    assert batch.num_conformers == 3
    assert batch.conformer_molecule_index.tolist() == [0, 0, 1]
    assert batch.positions.shape[0] == batch.atomic_numbers.shape[0]
    # node_conformer_index has one entry per packed atom, spanning 3 conformers.
    assert int(batch.node_conformer_index.max().item()) == 2
    # Second conformer's edges are offset past the first conformer's atoms.
    first_conf_atoms = graphs[0].num_nodes
    assert int(batch.edge_index[:, graphs[0].num_edges:].min().item()) >= first_conf_atoms


def test_pool_conformers_to_molecules_mean():
    conformer_embeddings = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]], dtype=torch.float32)
    conformer_molecule_index = torch.tensor([0, 0, 1], dtype=torch.long)

    pooled = pool_conformers_to_molecules(
        conformer_embeddings,
        conformer_molecule_index,
        conformer_energy=None,
        num_molecules=2,
        mode="mean",
    )

    assert pooled.shape == (2, 2)
    assert torch.allclose(pooled[0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(pooled[1], torch.tensor([5.0, 5.0]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_conformer_encoder_batch.py -v`
Expected: FAIL with `ImportError: cannot import name 'ConformerEncoderBatch'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to qchem_gnn/conformer.py
from .graph import GraphData


@dataclass(frozen=True)
class ConformerEncoderBatch:
    atomic_numbers: torch.LongTensor
    edge_index: torch.LongTensor
    positions: torch.Tensor
    node_conformer_index: torch.LongTensor
    conformer_molecule_index: torch.LongTensor
    conformer_energy: torch.Tensor | None
    num_conformers: int
    num_molecules: int

    @classmethod
    def from_molecule_conformers(
        cls,
        graphs: list[GraphData],
        conformer_coords: list[list[torch.Tensor]],
        conformer_energies: list[torch.Tensor] | None = None,
    ) -> "ConformerEncoderBatch":
        atomic_numbers: list[torch.Tensor] = []
        edge_indices: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        node_conformer_index: list[torch.Tensor] = []
        conformer_molecule_index: list[int] = []
        energies: list[torch.Tensor] = []

        node_offset = 0
        conformer_id = 0
        for molecule_id, (graph, coords_list) in enumerate(zip(graphs, conformer_coords)):
            for conformer_position, coords in enumerate(coords_list):
                atomic_numbers.append(graph.atomic_numbers)
                edge_indices.append(graph.edge_index + node_offset)
                positions.append(coords.to(torch.float32))
                node_conformer_index.append(
                    torch.full((graph.num_nodes,), conformer_id, dtype=torch.long)
                )
                conformer_molecule_index.append(molecule_id)
                if conformer_energies is not None:
                    energies.append(conformer_energies[molecule_id][conformer_position])
                node_offset += graph.num_nodes
                conformer_id += 1

        energy_tensor = torch.stack(energies, dim=0) if energies else None
        return cls(
            atomic_numbers=torch.cat(atomic_numbers, dim=0),
            edge_index=torch.cat(edge_indices, dim=1),
            positions=torch.cat(positions, dim=0),
            node_conformer_index=torch.cat(node_conformer_index, dim=0),
            conformer_molecule_index=torch.tensor(conformer_molecule_index, dtype=torch.long),
            conformer_energy=energy_tensor,
            num_conformers=conformer_id,
            num_molecules=len(graphs),
        )


def pool_conformers_to_molecules(
    conformer_embeddings: torch.Tensor,
    conformer_molecule_index: torch.Tensor,
    conformer_energy: torch.Tensor | None,
    num_molecules: int,
    mode: str = "mean",
) -> torch.Tensor:
    pooled = []
    for molecule_id in range(num_molecules):
        mask = conformer_molecule_index == molecule_id
        molecule_embeddings = conformer_embeddings[mask]
        molecule_energy = conformer_energy[mask] if conformer_energy is not None else None
        pooled.append(
            pool_conformer_embeddings(molecule_embeddings, conformer_energy=molecule_energy, mode=mode)
        )
    return torch.stack(pooled, dim=0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_conformer_encoder_batch.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full Phase 1 suite (no regressions)**

Run: `python -m pytest tests/test_encoder3d.py tests/test_contrastive_loss.py tests/test_conformer_encoder_batch.py tests/test_conformer.py -q`
Expected: all PASS (existing `test_conformer.py` unaffected — `ConformerBatch` is untouched).

- [ ] **Step 6: Commit**

```bash
git add qchem_gnn/conformer.py tests/test_conformer_encoder_batch.py
git commit -m "feat: add conformer encoder batching and molecule-level pooling"
```

---

# Phase 2 — Data pipeline, joint loop, config, CLI

### Task 5: Surface conformer coordinates and energies on dataset examples

**Files:**
- Modify: `qchem_gnn/minimal.py:14-24` (add fields to `MinimalQuantumExample`)
- Modify: `qchem_gnn/quantum_data.py` (populate the new fields)
- Test: `tests/test_quantum_dataset.py` (add a test)

**Interfaces:**
- Produces: `MinimalQuantumExample.conformer_coords: list[torch.Tensor] | None` (each `[n,3]`, atom order matches `example.graph`), `MinimalQuantumExample.conformer_energies: torch.Tensor | None` (`[k]`, or `None` when energies are unavailable).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_quantum_dataset.py
import pickle

import numpy as np
import pandas as pd
import torch

from qchem_gnn.minimal import load_minimal_zinc_dataset


def test_dataset_surfaces_conformer_coordinates(tmp_path):
    csv_path = tmp_path / "subset_000.csv"
    geo_path = tmp_path / "coords_000.pkl"
    pd.DataFrame([{"smiles": "C", "logP": 0.1, "qed": 0.2, "SAS": 0.3}]).to_csv(csv_path, index=False)
    pickle.dump(
        {
            "subset_0_idx_0": {
                "smiles": "C",
                "charge": 0,
                "atomic_nums": [6, 1, 1, 1, 1],
                "conformers": [np.zeros((5, 3), dtype=np.float32), np.ones((5, 3), dtype=np.float32)],
            }
        },
        geo_path.open("wb"),
    )

    dataset = load_minimal_zinc_dataset(csv_path, geo_path, limit=1)
    example = dataset.examples[0]

    assert example.conformer_coords is not None
    assert len(example.conformer_coords) == 2
    assert example.conformer_coords[0].shape == (example.graph.num_nodes, 3)
    assert isinstance(example.conformer_coords[0], torch.Tensor)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_quantum_dataset.py::test_dataset_surfaces_conformer_coordinates -v`
Expected: FAIL with `AttributeError: 'MinimalQuantumExample' object has no attribute 'conformer_coords'`.

- [ ] **Step 3: Add fields to the dataclass**

In `qchem_gnn/minimal.py`, extend `MinimalQuantumExample` (after `aux_target`):

```python
@dataclass(frozen=True)
class MinimalQuantumExample:
    mol_id: str
    smiles: str
    graph: GraphData
    charge: int
    conformer_count: int
    node_target: torch.Tensor
    edge_target: torch.Tensor
    graph_target: torch.Tensor
    aux_target: torch.Tensor | None = None
    conformer_coords: list[torch.Tensor] | None = None
    conformer_energies: torch.Tensor | None = None
```

- [ ] **Step 4: Populate the fields in the loader**

In `qchem_gnn/quantum_data.py`, add a helper near the top (after `_as_tensor`):

```python
def _conformer_coords_from_geometry(geometry: dict, num_nodes: int) -> list[torch.Tensor] | None:
    conformers = geometry.get("conformers")
    if not conformers:
        return None
    coords = []
    for conformer in conformers:
        tensor = _as_tensor(conformer)
        if tensor.ndim != 2 or tensor.shape[0] != num_nodes or tensor.shape[1] != 3:
            return None
        coords.append(tensor)
    return coords or None
```

Then, in `load_quantum_zinc_dataset`, where each `MinimalQuantumExample` is constructed (inside the `for mol_id, row ...` loop), build the coords/energies just before `examples.append(...)` and pass them in:

```python
                conformer_coords = _conformer_coords_from_geometry(geometry, graph.num_nodes)
                conformer_energies = None  # energies require HDF5 results; mean pooling is used otherwise
                examples.append(
                    MinimalQuantumExample(
                        mol_id=mol_id,
                        smiles=smiles,
                        graph=graph,
                        charge=int(geometry.get("charge", 0)),
                        conformer_count=conformer_count,
                        node_target=node_target,
                        edge_target=edge_target,
                        graph_target=graph_target,
                        aux_target=aux_target,
                        conformer_coords=conformer_coords,
                        conformer_energies=conformer_energies,
                    )
                )
```

(Energy-weighted pooling from HDF5 is a follow-up; mean pooling is the default and correct fallback when `conformer_energies is None`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_quantum_dataset.py::test_dataset_surfaces_conformer_coordinates -v`
Expected: PASS.

- [ ] **Step 6: Run existing dataset/pretraining tests (no regressions)**

Run: `python -m pytest tests/test_quantum_dataset.py tests/test_pretraining.py tests/test_minimal_dataset.py -q`
Expected: all PASS (new fields default to `None`).

- [ ] **Step 7: Commit**

```bash
git add qchem_gnn/minimal.py qchem_gnn/quantum_data.py tests/test_quantum_dataset.py
git commit -m "feat: surface conformer coordinates on quantum dataset examples"
```

---

### Task 6: Projection head + joint minibatched contrastive pretraining loop

**Files:**
- Create: `qchem_gnn/contrastive_pretrain.py`
- Test: `tests/test_contrastive_pretrain.py`

**Interfaces:**
- Consumes: `MolecularQuantumGNN` (`model.py`), `Conformer3DEncoder` (`encoder3d.py`), `info_nce_contrastive_loss` (`losses.py`), `ConformerEncoderBatch` + `pool_conformers_to_molecules` (`conformer.py`), `GraphBatch` (`graph.py`), `compute_multitask_loss` (`losses.py`), `compute_target_normalization`/`normalize_targets` (`quantum_data.py`), `MinimalQuantumDataset` (`minimal.py`).
- Produces:
  - `ProjectionHead(input_dim: int, output_dim: int)`; `forward(x) -> Tensor`.
  - `ContrastivePretrainingResult` (frozen dataclass): `model: MolecularQuantumGNN`, `loss_history: list[float]`, `contrastive_loss_history: list[float]`, `embeddings: Tensor`, `target_normalization: dict[str, Tensor]`, `optimizer_state_dict: dict`, `epoch: int`, `global_step: int`.
  - `contrastive_pretrain_on_dataset(dataset, *, hidden_dim=32, num_message_passing_steps=2, hidden_dim_3d=32, num_rbf=16, cutoff=5.0, num_message_passing_steps_3d=2, epochs=200, batch_size=8, learning_rate=0.01, supervised_weight=1.0, contrastive_weight=1.0, temperature=0.1, conformer_pool_mode="mean", seed=0) -> ContrastivePretrainingResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contrastive_pretrain.py
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qchem_gnn.minimal import load_minimal_zinc_dataset
from qchem_gnn.contrastive_pretrain import contrastive_pretrain_on_dataset


def _write_subset_with_conformers(tmp_path: Path):
    csv_path = tmp_path / "subset_000.csv"
    geo_path = tmp_path / "coords_000.pkl"
    pd.DataFrame(
        [
            {"smiles": "C", "logP": 0.1, "qed": 0.2, "SAS": 0.3},
            {"smiles": "CC", "logP": 0.4, "qed": 0.5, "SAS": 0.6},
            {"smiles": "CCC", "logP": 0.2, "qed": 0.3, "SAS": 0.4},
            {"smiles": "CCO", "logP": 0.5, "qed": 0.6, "SAS": 0.7},
        ]
    ).to_csv(csv_path, index=False)

    def _coords(n):
        rng = np.random.default_rng(n)
        return [rng.standard_normal((n, 3)).astype(np.float32)]

    pickle.dump(
        {
            "subset_0_idx_0": {"smiles": "C", "charge": 0, "atomic_nums": [6, 1, 1, 1, 1], "conformers": _coords(5)},
            "subset_0_idx_1": {"smiles": "CC", "charge": 0, "atomic_nums": [6, 6] + [1] * 6, "conformers": _coords(8)},
            "subset_0_idx_2": {"smiles": "CCC", "charge": 0, "atomic_nums": [6, 6, 6] + [1] * 8, "conformers": _coords(11)},
            "subset_0_idx_3": {"smiles": "CCO", "charge": 0, "atomic_nums": [6, 6, 8] + [1] * 6, "conformers": _coords(9)},
        },
        geo_path.open("wb"),
    )
    return csv_path, geo_path


def test_contrastive_pretrain_runs_and_reduces_loss(tmp_path: Path):
    csv_path, geo_path = _write_subset_with_conformers(tmp_path)
    dataset = load_minimal_zinc_dataset(csv_path, geo_path, limit=4)

    result = contrastive_pretrain_on_dataset(
        dataset,
        hidden_dim=16,
        num_message_passing_steps=2,
        hidden_dim_3d=16,
        epochs=120,
        batch_size=4,
        learning_rate=0.01,
        contrastive_weight=1.0,
        temperature=0.1,
        seed=0,
    )

    assert result.loss_history[-1] < result.loss_history[0]
    assert len(result.contrastive_loss_history) == len(result.loss_history)
    assert result.embeddings.shape == (4, 16)
    assert torch.isfinite(result.embeddings).all()
    # Inference checkpoint must be loadable by the existing 2D model loader.
    from qchem_gnn.model import MolecularQuantumGNN

    clone = MolecularQuantumGNN(atom_vocab_size=128, bond_vocab_size=8, hidden_dim=16, num_message_passing_steps=2, graph_targets=2)
    clone.load_state_dict(result.model.state_dict())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_contrastive_pretrain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qchem_gnn.contrastive_pretrain'`.

- [ ] **Step 3: Write minimal implementation**

```python
# qchem_gnn/contrastive_pretrain.py
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .conformer import ConformerEncoderBatch, pool_conformers_to_molecules
from .encoder3d import Conformer3DEncoder
from .graph import GraphBatch
from .losses import compute_multitask_loss, info_nce_contrastive_loss
from .minimal import MinimalQuantumDataset
from .model import MolecularQuantumGNN
from .quantum_data import compute_target_normalization, normalize_targets


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
    conformer_pool_mode: str = "mean",
    seed: int = 0,
) -> ContrastivePretrainingResult:
    torch.manual_seed(seed)
    examples = dataset.examples
    normalization = compute_target_normalization(dataset)

    model = MolecularQuantumGNN(
        atom_vocab_size=128,
        bond_vocab_size=8,
        hidden_dim=hidden_dim,
        num_message_passing_steps=num_message_passing_steps,
        graph_targets=2,
    )
    encoder3d = Conformer3DEncoder(
        atom_vocab_size=128,
        hidden_dim=hidden_dim_3d,
        num_rbf=num_rbf,
        cutoff=cutoff,
        num_message_passing_steps=num_message_passing_steps_3d,
    )
    proj_2d = ProjectionHead(hidden_dim, hidden_dim)
    proj_3d = ProjectionHead(hidden_dim_3d, hidden_dim)

    params = list(model.parameters()) + list(encoder3d.parameters())
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

            contrastive = torch.zeros((), dtype=supervised.dtype)
            with_coords = [ex for ex in batch_examples if ex.conformer_coords]
            if contrastive_weight and len(with_coords) >= 2:
                coords_index = [
                    pos for pos, ex in enumerate(batch_examples) if ex.conformer_coords
                ]
                conformer_batch = ConformerEncoderBatch.from_molecule_conformers(
                    [ex.graph for ex in with_coords],
                    [ex.conformer_coords for ex in with_coords],
                    conformer_energies=None,
                )
                conformer_embeddings = encoder3d(
                    conformer_batch.atomic_numbers,
                    conformer_batch.edge_index,
                    conformer_batch.positions,
                    conformer_batch.node_conformer_index,
                    conformer_batch.num_conformers,
                )
                molecule_3d = pool_conformers_to_molecules(
                    conformer_embeddings,
                    conformer_batch.conformer_molecule_index,
                    conformer_batch.conformer_energy,
                    conformer_batch.num_molecules,
                    mode=conformer_pool_mode,
                )
                molecule_2d = model_output.mol_embedding[coords_index]
                contrastive = info_nce_contrastive_loss(
                    proj_2d(molecule_2d), proj_3d(molecule_3d), temperature=temperature
                )

            total = supervised_weight * supervised + contrastive_weight * contrastive
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            epoch_total += float(total.item())
            epoch_contrastive += float(contrastive.item())
            num_batches += 1

        loss_history.append(epoch_total / max(num_batches, 1))
        contrastive_loss_history.append(epoch_contrastive / max(num_batches, 1))

    with torch.no_grad():
        full_batch = GraphBatch.from_graphs([ex.graph for ex in examples])
        embeddings = model.encode_graph_embeddings(full_batch)

    return ContrastivePretrainingResult(
        model=model,
        loss_history=loss_history,
        contrastive_loss_history=contrastive_loss_history,
        embeddings=embeddings,
        target_normalization=normalization,
        optimizer_state_dict=optimizer.state_dict(),
        epoch=epochs,
        global_step=epochs * ((num_examples + batch_size - 1) // batch_size),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_contrastive_pretrain.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/contrastive_pretrain.py tests/test_contrastive_pretrain.py
git commit -m "feat: add joint minibatched contrastive pretraining loop"
```

---

### Task 7: Register the `contrastive-pretrain` config command

**Files:**
- Modify: `qchem_gnn/config.py` (multiple sections noted below)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `resolve_config` accepts `command == "contrastive-pretrain"` with a `contrastive` section; `config_to_namespace` returns a `Namespace` carrying `batch_size`, `contrastive_weight`, `supervised_weight`, `temperature`, `hidden_dim_3d`, `num_rbf`, `cutoff`, `message_passing_steps_3d`, `conformer_pool_mode`, `seed`, plus the existing dataset/model/training/output fields.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_config.py
from qchem_gnn.config import config_to_namespace, resolve_config


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py::test_contrastive_pretrain_config_resolves -v`
Expected: FAIL with `ConfigError: command must be one of [...]`.

- [ ] **Step 3: Edit `qchem_gnn/config.py`**

(a) Add the command and section to the valid sets:

```python
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
```

(b) Add the section's valid keys to `VALID_SECTION_KEYS`:

```python
    "contrastive": {
        "batch_size",
        "supervised_weight",
        "contrastive_weight",
        "temperature",
        "hidden_dim_3d",
        "num_rbf",
        "cutoff",
        "message_passing_steps_3d",
        "conformer_pool_mode",
        "seed",
    },
```

(c) Add the section defaults to `DEFAULT_CONFIG` (after the `"training"` block):

```python
    "contrastive": {
        "batch_size": 8,
        "supervised_weight": 1.0,
        "contrastive_weight": 1.0,
        "temperature": 0.1,
        "hidden_dim_3d": 32,
        "num_rbf": 16,
        "cutoff": 5.0,
        "message_passing_steps_3d": 2,
        "conformer_pool_mode": "mean",
        "seed": 0,
    },
```

(d) In `_validate_config`, add `"contrastive"` to the section-iteration tuple:

```python
    for section_name in ("dataset", "model", "training", "contrastive", "outputs", "inference", "downstream"):
```

(e) In `_validate_config`, after the `training[...]` coercions, add contrastive coercions:

```python
    contrastive = config["contrastive"]
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
    if contrastive["conformer_pool_mode"] not in {"mean", "weighted", "energy"}:
        raise ConfigError("contrastive.conformer_pool_mode must be one of mean, weighted, energy")
```

(f) In `_validate_config`, extend the dataset-mode and checkpoint requirements to include the new command. Change:

```python
    if command in {"train", "pretrain"}:
        if has_csv == has_dataset_root:
```
to:
```python
    if command in {"train", "pretrain", "contrastive-pretrain"}:
        if has_csv == has_dataset_root:
```
and:
```python
    if command in {"train", "pretrain"} and not outputs.get("checkpoint"):
        raise ConfigError("outputs.checkpoint is required for training commands")
```
to:
```python
    if command in {"train", "pretrain", "contrastive-pretrain"} and not outputs.get("checkpoint"):
        raise ConfigError("outputs.checkpoint is required for training commands")
```

(g) In `config_to_namespace`, change the train/pretrain branch guard and append contrastive fields. Change:

```python
    if command in {"train", "pretrain"}:
        values: ConfigDict = {
```
to:
```python
    if command in {"train", "pretrain", "contrastive-pretrain"}:
        values: ConfigDict = {
```
and, immediately before `if command == "pretrain":`, add:

```python
        if command == "contrastive-pretrain":
            contrastive = config["contrastive"]
            values.update(
                {
                    "batch_size": _coerce_int(contrastive["batch_size"], "contrastive.batch_size"),
                    "supervised_weight": _coerce_float(contrastive["supervised_weight"], "contrastive.supervised_weight"),
                    "contrastive_weight": _coerce_float(contrastive["contrastive_weight"], "contrastive.contrastive_weight"),
                    "temperature": _coerce_float(contrastive["temperature"], "contrastive.temperature"),
                    "hidden_dim_3d": _coerce_int(contrastive["hidden_dim_3d"], "contrastive.hidden_dim_3d"),
                    "num_rbf": _coerce_int(contrastive["num_rbf"], "contrastive.num_rbf"),
                    "cutoff": _coerce_float(contrastive["cutoff"], "contrastive.cutoff"),
                    "message_passing_steps_3d": _coerce_int(contrastive["message_passing_steps_3d"], "contrastive.message_passing_steps_3d"),
                    "conformer_pool_mode": contrastive["conformer_pool_mode"],
                    "seed": _coerce_int(contrastive["seed"], "contrastive.seed"),
                }
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -q`
Expected: all PASS (existing config tests unaffected — the new section has defaults).

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/config.py tests/test_config.py
git commit -m "feat: register contrastive-pretrain command in config"
```

---

### Task 8: Wire the `contrastive-pretrain` CLI command

**Files:**
- Modify: `qchem_gnn/cli.py` (parser in `build_parser`, mapping in `_config_from_args`, new `run_contrastive_pretrain`, dispatch in `main`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `contrastive_pretrain_on_dataset` (Task 6), existing `load_quantum_zinc_dataset` / `load_quantum_zinc_subset_range`, `build_checkpoint_state`, `save_checkpoint`, `normalize_dataset_config`.
- Produces: `run_contrastive_pretrain(args) -> int`; writes a checkpoint loadable by `_load_model_from_checkpoint` (i.e. with `model_config`, `model_state_dict`, `dataset_config`, `target_normalization`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_cli.py
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from qchem_gnn.cli import main


def _write_contrastive_inputs(tmp_path: Path):
    csv_path = tmp_path / "subset_000.csv"
    geo_path = tmp_path / "coords_000.pkl"
    pd.DataFrame(
        [
            {"smiles": "C", "logP": 0.1, "qed": 0.2, "SAS": 0.3},
            {"smiles": "CC", "logP": 0.4, "qed": 0.5, "SAS": 0.6},
            {"smiles": "CCC", "logP": 0.2, "qed": 0.3, "SAS": 0.4},
            {"smiles": "CCO", "logP": 0.5, "qed": 0.6, "SAS": 0.7},
        ]
    ).to_csv(csv_path, index=False)
    pickle.dump(
        {
            "subset_0_idx_0": {"smiles": "C", "charge": 0, "atomic_nums": [6, 1, 1, 1, 1], "conformers": [np.zeros((5, 3), dtype=np.float32)]},
            "subset_0_idx_1": {"smiles": "CC", "charge": 0, "atomic_nums": [6, 6] + [1] * 6, "conformers": [np.zeros((8, 3), dtype=np.float32)]},
            "subset_0_idx_2": {"smiles": "CCC", "charge": 0, "atomic_nums": [6, 6, 6] + [1] * 8, "conformers": [np.zeros((11, 3), dtype=np.float32)]},
            "subset_0_idx_3": {"smiles": "CCO", "charge": 0, "atomic_nums": [6, 6, 8] + [1] * 6, "conformers": [np.zeros((9, 3), dtype=np.float32)]},
        },
        geo_path.open("wb"),
    )
    return csv_path, geo_path


def test_cli_contrastive_pretrain_then_export(tmp_path: Path):
    csv_path, geo_path = _write_contrastive_inputs(tmp_path)
    checkpoint = tmp_path / "contrastive.pt"
    embeddings = tmp_path / "emb.pt"

    code = main(
        [
            "contrastive-pretrain",
            "--csv", str(csv_path),
            "--geometry", str(geo_path),
            "--limit", "4",
            "--epochs", "20",
            "--hidden-dim", "16",
            "--batch-size", "4",
            "--output", str(checkpoint),
        ]
    )
    assert code == 0
    assert checkpoint.exists()

    # The checkpoint must work with the unchanged export-embeddings path.
    code = main(["export-embeddings", "--checkpoint", str(checkpoint), "--output", str(embeddings)])
    assert code == 0
    assert embeddings.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py::test_cli_contrastive_pretrain_then_export -v`
Expected: FAIL — `contrastive-pretrain` is not a known subcommand (argparse error / SystemExit).

- [ ] **Step 3: Add the parser in `build_parser`**

After the `pretrain` parser block (before the `export` parser), add:

```python
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
    contrastive.add_argument("--epochs", type=int, help="Training epochs")
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
    contrastive.add_argument("--conformer-pool-mode", help="Conformer pooling mode: mean, weighted, or energy")
    contrastive.add_argument("--seed", type=int, help="Random seed")
    contrastive.add_argument("--output", help="Checkpoint output path")
```

- [ ] **Step 4: Map CLI args in `_config_from_args`**

In `_config_from_args`, change the dataset/model/training/outputs guard from `args.command in {"train", "pretrain"}` to include the new command, and add a `contrastive` section. Replace the opening guard:

```python
    if args.command in {"train", "pretrain", "contrastive-pretrain"}:
```

Then, after the `training` block and before the `outputs` block, add:

```python
        if args.command == "contrastive-pretrain":
            contrastive: dict[str, object] = {}
            for arg_name, key in (
                ("hidden_dim_3d", "hidden_dim_3d"),
                ("message_passing_steps_3d", "message_passing_steps_3d"),
                ("num_rbf", "num_rbf"),
                ("cutoff", "cutoff"),
                ("batch_size", "batch_size"),
                ("supervised_weight", "supervised_weight"),
                ("contrastive_weight", "contrastive_weight"),
                ("temperature", "temperature"),
                ("conformer_pool_mode", "conformer_pool_mode"),
                ("seed", "seed"),
            ):
                if hasattr(args, arg_name):
                    contrastive[key] = getattr(args, arg_name)
            if contrastive:
                config["contrastive"] = contrastive
```

- [ ] **Step 5: Add `run_contrastive_pretrain`**

After `run_pretrain`, add:

```python
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
        hidden_dim=args.hidden_dim,
        num_message_passing_steps=args.message_passing_steps,
        hidden_dim_3d=args.hidden_dim_3d,
        num_rbf=args.num_rbf,
        cutoff=args.cutoff,
        num_message_passing_steps_3d=args.message_passing_steps_3d,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        supervised_weight=args.supervised_weight,
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature,
        conformer_pool_mode=args.conformer_pool_mode,
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
            "graph_targets": 2,
        },
        run_metadata={
            "num_examples": len(dataset),
            "num_skipped": len(dataset.skipped_mol_ids),
            "epochs": args.epochs,
            "contrastive_weight": args.contrastive_weight,
            "contrastive_loss_history": result.contrastive_loss_history,
            **run_metadata,
            "resolved_config": getattr(args, "resolved_config", None),
        },
    )
    save_checkpoint(Path(args.output), checkpoint_payload)
    return 0
```

- [ ] **Step 6: Dispatch in `main`**

In `main`, after the `pretrain` dispatch line, add:

```python
    if args.command == "contrastive-pretrain":
        return run_contrastive_pretrain(args)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py::test_cli_contrastive_pretrain_then_export -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add qchem_gnn/cli.py tests/test_cli.py
git commit -m "feat: add contrastive-pretrain CLI command"
```

---

### Task 9: Minimal runnable YAML config

**Files:**
- Create: `configs/minimal_contrastive_pretrain.yaml`
- Test: `tests/test_config.py` (add a load-from-file assertion)

**Interfaces:**
- Consumes: the `contrastive-pretrain` config schema (Task 7).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_config.py
from pathlib import Path

from qchem_gnn.config import load_yaml_config, resolve_config


def test_minimal_contrastive_yaml_resolves():
    path = Path("configs/minimal_contrastive_pretrain.yaml")
    resolved = resolve_config(load_yaml_config(path))

    assert resolved["command"] == "contrastive-pretrain"
    assert resolved["contrastive"]["batch_size"] >= 2
    assert resolved["outputs"]["checkpoint"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py::test_minimal_contrastive_yaml_resolves -v`
Expected: FAIL with `ConfigError: Failed to read YAML config configs/minimal_contrastive_pretrain.yaml`.

- [ ] **Step 3: Create the YAML**

```yaml
# configs/minimal_contrastive_pretrain.yaml
command: contrastive-pretrain

dataset:
  dataset_root: zinc-250k
  subset_ids: [0]
  limit_per_shard: 16
  use_results: false

model:
  hidden_dim: 32
  message_passing_steps: 2

training:
  epochs: 100
  learning_rate: 0.01

contrastive:
  batch_size: 8
  supervised_weight: 1.0
  contrastive_weight: 1.0
  temperature: 0.1
  hidden_dim_3d: 32
  num_rbf: 16
  cutoff: 5.0
  message_passing_steps_3d: 2
  conformer_pool_mode: mean
  seed: 0

outputs:
  checkpoint: runs/minimal_contrastive.pt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py::test_minimal_contrastive_yaml_resolves -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/minimal_contrastive_pretrain.yaml tests/test_config.py
git commit -m "feat: add minimal contrastive-pretrain YAML config"
```

---

# Phase 3 — Ablation (does contrastive actually help?)

### Task 10: Ablation harness comparing supervised-only vs +contrastive

**Files:**
- Modify: `qchem_gnn/contrastive_pretrain.py` (add `run_contrastive_ablation`)
- Test: `tests/test_contrastive_ablation.py`

**Interfaces:**
- Consumes: `contrastive_pretrain_on_dataset` (Task 6), `run_linear_probe` (`eval.py`), `scaffold_or_random_split` (`splits.py`).
- Produces: `run_contrastive_ablation(dataset, *, hidden_dim=32, epochs=200, batch_size=8, contrastive_weight=1.0, seed=0) -> dict` with keys `supervised_only` and `with_contrastive`, each a linear-probe metrics dict from `run_linear_probe`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contrastive_ablation.py
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from qchem_gnn.minimal import load_minimal_zinc_dataset
from qchem_gnn.contrastive_pretrain import run_contrastive_ablation


def _write_inputs(tmp_path: Path):
    csv_path = tmp_path / "subset_000.csv"
    geo_path = tmp_path / "coords_000.pkl"
    rows = []
    geometry = {}
    smiles_list = ["C", "CC", "CCC", "CCO", "CCN", "CCCC", "CCCO", "CCCN"]
    atom_counts = [5, 8, 11, 9, 9, 14, 12, 12]
    for idx, (smiles, n_atoms) in enumerate(zip(smiles_list, atom_counts)):
        rows.append({"smiles": smiles, "logP": 0.1 * idx, "qed": 0.05 * idx, "SAS": 0.2 * idx})
        rng = np.random.default_rng(idx)
        geometry[f"subset_0_idx_{idx}"] = {
            "smiles": smiles,
            "charge": 0,
            "atomic_nums": [6] * n_atoms,
            "conformers": [rng.standard_normal((n_atoms, 3)).astype(np.float32)],
        }
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    pickle.dump(geometry, geo_path.open("wb"))
    return csv_path, geo_path


def test_run_contrastive_ablation_returns_both_arms(tmp_path: Path):
    csv_path, geo_path = _write_inputs(tmp_path)
    dataset = load_minimal_zinc_dataset(csv_path, geo_path, limit=8)

    report = run_contrastive_ablation(
        dataset, hidden_dim=16, epochs=30, batch_size=4, contrastive_weight=1.0, seed=0
    )

    assert set(report) == {"supervised_only", "with_contrastive"}
    for arm in report.values():
        assert isinstance(arm, dict)
        assert arm  # non-empty metrics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_contrastive_ablation.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_contrastive_ablation'`.

- [ ] **Step 3: Add the ablation function**

```python
# append to qchem_gnn/contrastive_pretrain.py
import numpy as np

from .eval import run_linear_probe
from .splits import scaffold_or_random_split


def run_contrastive_ablation(
    dataset: MinimalQuantumDataset,
    *,
    hidden_dim: int = 32,
    epochs: int = 200,
    batch_size: int = 8,
    contrastive_weight: float = 1.0,
    seed: int = 0,
) -> dict[str, dict]:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_contrastive_ablation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/contrastive_pretrain.py tests/test_contrastive_ablation.py
git commit -m "feat: add supervised-vs-contrastive ablation harness"
```

---

### Task 11: Full suite green + README documentation

**Files:**
- Modify: `README.md` (add a "Contrastive pretraining" subsection under Training)
- Verify: entire test suite

- [ ] **Step 1: Run the full suite**

Run: `python -m pytest tests -q`
Expected: all PASS, including every test added in Tasks 1–10 and all pre-existing tests.

- [ ] **Step 2: Document the new command in `README.md`**

Add, after the "Minimal pretraining example" block:

```markdown
### Cross-modal 2D/3D contrastive pretraining

Distills the 3D/quantum view (conformer geometries) into the 2D encoder via a
cross-modal contrastive objective. Inference stays 2D-only — checkpoints work
with `export-embeddings` and `eval` unchanged.

```bash
python -m qchem_gnn.cli contrastive-pretrain --config configs/minimal_contrastive_pretrain.yaml
```

To compare against the supervised-only baseline, run the ablation via
`qchem_gnn.contrastive_pretrain.run_contrastive_ablation`, which reports linear-probe
metrics for both arms on the same split.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document contrastive-pretrain command and ablation"
```

---

## Notes for the implementer

- **TDD order matters:** Phases must be done in order. Phase 1 tasks are pure and need no data files. Phase 2 depends on Phase 1 interfaces; Phase 3 depends on Phase 2.
- **In-batch negatives need `batch_size >= 2`** with conformer coordinates present; `info_nce_contrastive_loss` raises otherwise. The loop guards this (`len(with_coords) >= 2`) and silently skips the contrastive term for degenerate batches — that is intended, not a bug.
- **Energy-weighted pooling** is wired through (`conformer_pool_mode`) but defaults to `mean` because the minimal/CSV path has no per-conformer energies. Populating `conformer_energies` from HDF5 results is a deliberate follow-up, out of scope here.
- **Do not touch** `MolecularQuantumGNN`, `ConformerBatch`, or the `export-embeddings`/`eval` code paths — the 2D-only inference contract depends on them staying stable.
- After completing all tasks, the success criterion from the spec is met: `run_contrastive_ablation` produces a falsifiable supervised-only vs +contrastive comparison on the existing downstream harness.
```
