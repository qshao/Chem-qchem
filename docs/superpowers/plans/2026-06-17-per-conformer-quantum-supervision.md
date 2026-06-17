# Per-Conformer Quantum Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Supervise a geometry-aware teacher on per-conformer DFT quantum labels and distill a Boltzmann-ensemble prediction into the 2D student, so the contrastive pretraining actually exploits multiple conformers with quantum labels while keeping SMILES-only inference.

**Architecture:** A new Boltzmann module provides energy-weighting helpers. The data layer (`quantum_data.py`) stops mean-collapsing labels, loads per-conformer chelpg/wbi/energy/polarizability plus conformer energies and coordinates straight from the HDF5, and exposes both per-conformer tensors and a Boltzmann-weighted molecule-level target. The 3D encoder exposes per-atom states; a new `QuantumTeacherHeads` module predicts per-conformer properties from them. The contrastive trainer adds a teacher regression term and switches conformer pooling to energy-weighted.

**Tech Stack:** Python 3.13+, PyTorch, h5py, RDKit, pandas, numpy, pytest.

## Global Constraints

- Inference contract is unchanged: prediction stays 2D/SMILES-only. The teacher is a pretraining-only auxiliary, discarded after training.
- Real training data is the smallest real DFT shard: `zinc-250k/results/results_044.h5` (325 molecules). Do not depend on the 7 GB shards.
- Per-conformer target set: node = chelpg `[N,1]`; edge = wbi gathered at graph edges `[E,1]`; graph = `[energy, isotropic_polarizability]` `[2]`. `iao_pops` is per-orbital and is NOT used.
- Isotropic polarizability = `trace(alpha_3x3)/3`.
- Boltzmann weights use energies in Hartree with `k_B = 3.166811563e-6` Hartree/K, default `T = 298.15 K`; missing/degenerate/single energies fall back to uniform weights without raising.
- The proxy path (`use_results: false`) must keep working unchanged; the existing 120 tests must still pass.
- `build_graph_from_smiles` adds explicit H by default, so graph node count equals the DFT atom count — rely on this for chelpg/coords alignment and validate it.

---

### Task 1: Boltzmann energy-weighting helpers

**Files:**
- Create: `qchem_gnn/boltzmann.py`
- Test: `tests/test_boltzmann.py`

**Interfaces:**
- Produces:
  - `BOLTZMANN_HARTREE_PER_K: float = 3.166811563e-6`
  - `boltzmann_weights(energies: torch.Tensor, temperature: float = 298.15) -> torch.Tensor` — input `[C]` energies in Hartree, returns `[C]` float32 weights summing to 1.
  - `boltzmann_average(values: torch.Tensor, energies: torch.Tensor, temperature: float = 298.15) -> torch.Tensor` — input `values` `[C, ...]`, returns `[...]` (the leading conformer axis weighted and summed).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_boltzmann.py
import math

import pytest
import torch

from qchem_gnn.boltzmann import (
    BOLTZMANN_HARTREE_PER_K,
    boltzmann_average,
    boltzmann_weights,
)


def test_weights_sum_to_one_and_favor_lower_energy():
    energies = torch.tensor([-115.0000, -115.0005])
    w = boltzmann_weights(energies)
    assert w.dtype == torch.float32
    assert math.isclose(float(w.sum()), 1.0, rel_tol=1e-6)
    assert w[0] > w[1]  # lower energy gets more weight


def test_single_conformer_weight_is_one():
    w = boltzmann_weights(torch.tensor([-42.0]))
    assert w.shape == (1,)
    assert math.isclose(float(w[0]), 1.0, rel_tol=1e-6)


def test_degenerate_energies_are_uniform():
    w = boltzmann_weights(torch.tensor([-5.0, -5.0, -5.0]))
    assert torch.allclose(w, torch.full((3,), 1.0 / 3.0), atol=1e-6)


def test_weights_match_manual_softmax():
    energies = torch.tensor([-10.0, -10.001, -9.999])
    kt = BOLTZMANN_HARTREE_PER_K * 298.15
    manual = torch.softmax(-(energies - energies.min()) / kt, dim=0)
    assert torch.allclose(boltzmann_weights(energies), manual.to(torch.float32), atol=1e-6)


def test_average_weights_leading_axis():
    energies = torch.tensor([-1.0, -1.0])  # uniform -> plain mean
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = boltzmann_average(values, energies)
    assert torch.allclose(out, torch.tensor([2.0, 3.0]), atol=1e-6)


def test_average_handles_3d_values():
    energies = torch.tensor([-1.0, -1.0])
    values = torch.ones(2, 4, 1)
    out = boltzmann_average(values, energies)
    assert out.shape == (4, 1)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        boltzmann_weights(torch.empty(0))
    with pytest.raises(ValueError):
        boltzmann_weights(torch.tensor([-1.0]), temperature=0.0)
    with pytest.raises(ValueError):
        boltzmann_average(torch.ones(3, 2), torch.tensor([-1.0, -1.0]))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_boltzmann.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'qchem_gnn.boltzmann'`

- [ ] **Step 3: Write the implementation**

```python
# qchem_gnn/boltzmann.py
from __future__ import annotations

import torch

# Boltzmann constant in Hartree per Kelvin.
BOLTZMANN_HARTREE_PER_K: float = 3.166811563e-6


def boltzmann_weights(
    energies: torch.Tensor, temperature: float = 298.15
) -> torch.Tensor:
    """Boltzmann weights over conformer energies (Hartree). Returns float32 [C]."""
    if energies.ndim != 1:
        raise ValueError("energies must be a 1D tensor")
    if energies.numel() == 0:
        raise ValueError("energies must contain at least one conformer")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    e = energies.to(torch.float64)
    kt = BOLTZMANN_HARTREE_PER_K * temperature
    shifted = -(e - e.min()) / kt
    return torch.softmax(shifted, dim=0).to(torch.float32)


def boltzmann_average(
    values: torch.Tensor, energies: torch.Tensor, temperature: float = 298.15
) -> torch.Tensor:
    """Boltzmann-weighted average of ``values`` over their leading conformer axis."""
    weights = boltzmann_weights(energies, temperature)
    if values.shape[0] != weights.shape[0]:
        raise ValueError("values and energies must share the leading conformer dimension")
    weight_shape = [weights.shape[0]] + [1] * (values.ndim - 1)
    return (values.to(torch.float32) * weights.reshape(weight_shape)).sum(dim=0)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_boltzmann.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/boltzmann.py tests/test_boltzmann.py
git commit -m "feat(quantum): Boltzmann energy-weighting helpers"
```

---

### Task 2: Per-conformer target extraction from HDF5

**Files:**
- Modify: `qchem_gnn/quantum_data.py` (add helpers near `_aggregate_targets`, line ~97)
- Create: `tests/_quantum_fixtures.py`
- Test: `tests/test_per_conformer_targets.py`

**Interfaces:**
- Consumes: `build_graph_from_smiles` (graph with explicit H), h5py molecule group.
- Produces (in `quantum_data.py`):
  - `_isotropic_polarizability(value) -> float`
  - `@dataclass(frozen=True) PerConformerTargets` with fields `coords: list[torch.Tensor]` (each `[N,3]`), `node_targets: torch.Tensor` `[C,N,1]`, `edge_targets: torch.Tensor` `[C,E,1]`, `graph_targets: torch.Tensor` `[C,2]`, `energies: torch.Tensor` `[C]`.
  - `extract_per_conformer_targets(graph, mol_group) -> PerConformerTargets` — raises `ValueError` if no conformer carries the full chelpg/wbi/coords set, or if an atom count mismatches the graph.
- Produces (in `tests/_quantum_fixtures.py`):
  - `write_synthetic_results_h5(path, *, mol_id="subset_44_idx_0", smiles="CO", n_atoms=6, n_basis=10, energies=(-115.0, -115.0005), seed=0) -> Path`

- [ ] **Step 1: Write the shared synthetic-HDF5 fixture builder**

```python
# tests/_quantum_fixtures.py
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_synthetic_results_h5(
    path,
    *,
    mol_id: str = "subset_44_idx_0",
    smiles: str = "CO",
    n_atoms: int = 6,
    n_basis: int = 10,
    energies=(-115.0, -115.0005),
    seed: int = 0,
) -> Path:
    """Write one molecule group mirroring the real results_044.h5 schema.

    ``CO`` (methanol) parses to 2 heavy atoms + 4 H = 6 atoms, matching n_atoms.
    """
    import h5py

    rng = np.random.default_rng(seed)
    path = Path(path)
    with h5py.File(path, "w") as handle:
        group = handle.create_group(mol_id)
        group.attrs["smiles"] = smiles
        for conf_idx, energy in enumerate(energies):
            conf = group.create_group(f"conf_{conf_idx}")
            conf.attrs["energy"] = float(energy)
            conf.attrs["polarizability"] = (np.eye(3) * (10.0 + conf_idx)).astype(np.float64)
            conf.create_dataset("chelpg", data=rng.standard_normal(n_atoms))
            wbi = rng.standard_normal((n_atoms, n_atoms))
            conf.create_dataset("wbi", data=(wbi + wbi.T) / 2.0)
            conf.create_dataset("coords", data=rng.standard_normal((n_atoms, 3)))
            conf.create_dataset("iao_pops", data=rng.standard_normal(n_basis + 3))
            conf.create_dataset("fukui_p", data=rng.standard_normal(n_basis))
            conf.create_dataset("dm", data=rng.standard_normal((n_basis, n_basis)))
    return path
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_per_conformer_targets.py
from pathlib import Path

import h5py
import pytest
import torch

from qchem_gnn.graph import build_graph_from_smiles
from qchem_gnn.quantum_data import (
    PerConformerTargets,
    _isotropic_polarizability,
    extract_per_conformer_targets,
)
from tests._quantum_fixtures import write_synthetic_results_h5

REAL_SHARD = Path("zinc-250k/results/results_044.h5")


def test_isotropic_polarizability_is_trace_over_three():
    alpha = [[3.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 9.0]]
    assert _isotropic_polarizability(alpha) == pytest.approx(6.0)


def test_isotropic_polarizability_scalar_passthrough():
    assert _isotropic_polarizability(5.0) == pytest.approx(5.0)


def test_extract_shapes_from_synthetic(tmp_path):
    h5_path = write_synthetic_results_h5(tmp_path / "synthetic.h5")
    graph = build_graph_from_smiles("CO")
    with h5py.File(h5_path, "r") as handle:
        pct = extract_per_conformer_targets(graph, handle["subset_44_idx_0"])

    assert isinstance(pct, PerConformerTargets)
    n, e = graph.num_nodes, graph.num_edges
    assert pct.node_targets.shape == (2, n, 1)
    assert pct.edge_targets.shape == (2, e, 1)
    assert pct.graph_targets.shape == (2, 2)
    assert pct.energies.shape == (2,)
    assert len(pct.coords) == 2 and pct.coords[0].shape == (n, 3)
    # graph target column 1 is isotropic polarizability = trace/3 of (10/11 * I)
    assert pct.graph_targets[0, 1].item() == pytest.approx(10.0, rel=1e-5)
    assert pct.graph_targets[1, 1].item() == pytest.approx(11.0, rel=1e-5)


def test_extract_rejects_atom_count_mismatch(tmp_path):
    # n_atoms deliberately wrong for "CO" (6) -> mismatch must raise
    h5_path = write_synthetic_results_h5(tmp_path / "bad.h5", n_atoms=5)
    graph = build_graph_from_smiles("CO")
    with h5py.File(h5_path, "r") as handle:
        with pytest.raises(ValueError):
            extract_per_conformer_targets(graph, handle["subset_44_idx_0"])


@pytest.mark.skipif(not REAL_SHARD.exists(), reason="real DFT shard not present")
def test_extract_from_real_shard():
    with h5py.File(REAL_SHARD, "r") as handle:
        mol_id = next(iter(handle.keys()))
        smiles = str(handle[mol_id].attrs["smiles"])
        graph = build_graph_from_smiles(smiles)
        pct = extract_per_conformer_targets(graph, handle[mol_id])
    assert pct.node_targets.shape[1] == graph.num_nodes
    assert pct.node_targets.shape[2] == 1
    assert pct.graph_targets.shape[1] == 2
    assert pct.energies.numel() == pct.node_targets.shape[0]
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest tests/test_per_conformer_targets.py -q`
Expected: FAIL with `ImportError: cannot import name 'PerConformerTargets'`

- [ ] **Step 4: Implement the helpers in `quantum_data.py`**

Add `from dataclasses import dataclass` to the imports at the top of the file (it is not yet imported). Then add, immediately after `_aggregate_targets` (after line 156):

```python
@dataclass(frozen=True)
class PerConformerTargets:
    coords: list[torch.Tensor]
    node_targets: torch.Tensor   # [C, N, 1]
    edge_targets: torch.Tensor   # [C, E, 1]
    graph_targets: torch.Tensor  # [C, 2]
    energies: torch.Tensor       # [C]


def _isotropic_polarizability(value) -> float:
    tensor = _as_tensor(value)
    if tensor.numel() == 1:
        return float(tensor.item())
    if tensor.shape == (3, 3):
        return float(torch.diagonal(tensor).mean())
    return float(tensor.mean())


def _conformer_polarizability(conf_group) -> float:
    attrs = getattr(conf_group, "attrs", {})
    if "polarizability" in attrs:
        return _isotropic_polarizability(attrs["polarizability"])
    if "alpha" in attrs:
        return _isotropic_polarizability(attrs["alpha"])
    dataset = _collect_conf_value(conf_group, ("polarizability", "alpha"))
    return _isotropic_polarizability(dataset) if dataset is not None else 0.0


def extract_per_conformer_targets(graph, mol_group) -> PerConformerTargets:
    conf_groups = _conformer_groups(mol_group)
    if not conf_groups:
        raise ValueError("Result group contains no conformer groups")

    coords: list[torch.Tensor] = []
    node_targets: list[torch.Tensor] = []
    edge_targets: list[torch.Tensor] = []
    graph_targets: list[torch.Tensor] = []
    energies: list[float] = []

    src, dst = graph.edge_index[0], graph.edge_index[1]
    for conf_group in conf_groups:
        chelpg = _collect_conf_value(conf_group, ("chelpg",))
        wbi = _collect_conf_value(conf_group, ("wbi",))
        conf_coords = _collect_conf_value(conf_group, ("coords",))
        if chelpg is None or wbi is None or conf_coords is None:
            continue
        if chelpg.shape[0] != graph.num_nodes or conf_coords.shape[0] != graph.num_nodes:
            raise ValueError(
                f"Atom count mismatch: graph has {graph.num_nodes}, "
                f"conformer has chelpg {tuple(chelpg.shape)}"
            )

        node_targets.append(chelpg.unsqueeze(-1).to(torch.float32))
        edge_targets.append(wbi[src, dst].unsqueeze(-1).to(torch.float32))
        coords.append(conf_coords.to(torch.float32))

        attrs = getattr(conf_group, "attrs", {})
        energy = float(attrs["energy"]) if "energy" in attrs else 0.0
        energies.append(energy)
        graph_targets.append(
            torch.tensor([energy, _conformer_polarizability(conf_group)], dtype=torch.float32)
        )

    if not node_targets:
        raise ValueError("No conformer carried a complete chelpg/wbi/coords set")

    return PerConformerTargets(
        coords=coords,
        node_targets=torch.stack(node_targets, dim=0),
        edge_targets=torch.stack(edge_targets, dim=0),
        graph_targets=torch.stack(graph_targets, dim=0),
        energies=torch.tensor(energies, dtype=torch.float32),
    )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_per_conformer_targets.py -q`
Expected: PASS (the real-shard test runs if `zinc-250k/results/results_044.h5` is present, else skipped)

- [ ] **Step 6: Commit**

```bash
git add qchem_gnn/quantum_data.py tests/_quantum_fixtures.py tests/test_per_conformer_targets.py
git commit -m "feat(quantum): extract per-conformer chelpg/wbi/energy/polarizability from HDF5"
```

---

### Task 3: Per-conformer dataset fields and Boltzmann-averaged molecule targets

**Files:**
- Modify: `qchem_gnn/minimal.py:14-26` (add fields to `MinimalQuantumExample`)
- Modify: `qchem_gnn/quantum_data.py` (import `boltzmann_average`; use `extract_per_conformer_targets` in `load_quantum_zinc_dataset`, lines ~224-262)
- Test: `tests/test_quantum_dataset_loading.py`

**Interfaces:**
- Consumes: `extract_per_conformer_targets`, `PerConformerTargets` (Task 2); `boltzmann_average` (Task 1).
- Produces: `MinimalQuantumExample` gains `conformer_node_targets: torch.Tensor | None = None`, `conformer_edge_targets: torch.Tensor | None = None`, `conformer_graph_targets: torch.Tensor | None = None` (the `conformer_energies` field already exists). When `use_results=True` and the HDF5 group is present, `node_target`/`edge_target`/`graph_target` become the Boltzmann-weighted average over conformers, `conformer_coords` come from the HDF5 `coords`, and molecules with no DFT group are skipped. The proxy path (`use_results=False`) is unchanged and leaves the three new fields `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quantum_dataset_loading.py
from pathlib import Path

import torch

from qchem_gnn.boltzmann import boltzmann_average
from qchem_gnn.graph import build_graph_from_smiles
from qchem_gnn.quantum_data import extract_per_conformer_targets, load_quantum_zinc_dataset
from tests._quantum_fixtures import write_synthetic_results_h5
import h5py


def _write_csv(path: Path, smiles: str) -> Path:
    path.write_text("smiles,logP,qed,SAS\n" + f"{smiles},0.0,0.5,2.0\n")
    return path


def test_use_results_populates_per_conformer_fields(tmp_path):
    csv_path = _write_csv(tmp_path / "subset_044.csv", "CO")
    h5_path = write_synthetic_results_h5(tmp_path / "results_044.h5")

    with h5py.File(h5_path, "r") as handle:
        dataset = load_quantum_zinc_dataset(
            csv_path,
            geometry_path=tmp_path / "coords_044.pkl",  # absent -> no pickle coords
            limit=1,
            results_handle=handle,
            use_results=True,
        )

    assert len(dataset.examples) == 1
    ex = dataset.examples[0]
    assert ex.conformer_node_targets is not None
    assert ex.conformer_node_targets.shape[0] == 2  # two conformers
    assert ex.conformer_graph_targets.shape == (2, 2)
    assert ex.conformer_energies.shape == (2,)
    # molecule-level target equals the Boltzmann average of the per-conformer targets
    expected = boltzmann_average(ex.conformer_node_targets, ex.conformer_energies)
    assert torch.allclose(ex.node_target, expected, atol=1e-5)
    # coords used for the 3D path come from the HDF5 (one per conformer)
    assert ex.conformer_coords is not None and len(ex.conformer_coords) == 2


def test_proxy_path_leaves_per_conformer_fields_none(tmp_path):
    csv_path = _write_csv(tmp_path / "subset_044.csv", "CO")
    dataset = load_quantum_zinc_dataset(
        csv_path,
        geometry_path=tmp_path / "coords_044.pkl",
        limit=1,
        use_results=False,
    )
    ex = dataset.examples[0]
    assert ex.conformer_node_targets is None
    assert ex.conformer_graph_targets is None


def test_molecules_without_dft_group_are_skipped(tmp_path):
    # CSV has two rows, HDF5 only has subset_44_idx_0 -> second row skipped
    (tmp_path / "subset_044.csv").write_text(
        "smiles,logP,qed,SAS\nCO,0.0,0.5,2.0\nCCO,0.1,0.6,2.1\n"
    )
    h5_path = write_synthetic_results_h5(tmp_path / "results_044.h5")
    with h5py.File(h5_path, "r") as handle:
        dataset = load_quantum_zinc_dataset(
            tmp_path / "subset_044.csv",
            geometry_path=tmp_path / "coords_044.pkl",
            limit=2,
            results_handle=handle,
            use_results=True,
        )
    assert [ex.mol_id for ex in dataset.examples] == ["subset_44_idx_0"]
    assert "subset_44_idx_1" in dataset.skipped_mol_ids
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_quantum_dataset_loading.py -q`
Expected: FAIL — `MinimalQuantumExample` has no `conformer_node_targets` field / molecule target is not the Boltzmann average.

- [ ] **Step 3: Add the new fields to `MinimalQuantumExample`**

In `qchem_gnn/minimal.py`, extend the dataclass (currently ending at `conformer_energies: torch.Tensor | None = None`):

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
    conformer_node_targets: torch.Tensor | None = None
    conformer_edge_targets: torch.Tensor | None = None
    conformer_graph_targets: torch.Tensor | None = None
```

- [ ] **Step 4: Wire the per-conformer path into `load_quantum_zinc_dataset`**

In `qchem_gnn/quantum_data.py`, add `from .boltzmann import boltzmann_average` to the imports. Replace the per-molecule body in the loop (lines ~226-262, from `smiles = str(row.smiles).strip()` through the `examples.append(...)` call) with:

```python
                smiles = str(row.smiles).strip()
                graph = build_graph_from_smiles(smiles)
                geometry = geometry_lookup.get(mol_id, {})
                mol_group = _result_group(h5_handle, mol_id)

                conformer_node_targets = None
                conformer_edge_targets = None
                conformer_graph_targets = None
                conformer_energies = None

                if use_results and h5_handle is not None and mol_group is None:
                    # real-data mode: require DFT results, skip molecules without them
                    skipped_mol_ids.append(mol_id)
                    continue

                if mol_group is None:
                    node_target, edge_target, graph_target = _build_proxy_targets(graph)
                    conformer_count = len(geometry.get("conformers", []))
                    conformer_coords = _conformer_coords_from_geometry(geometry, graph.num_nodes)
                else:
                    pct = extract_per_conformer_targets(graph, mol_group)
                    conformer_node_targets = pct.node_targets
                    conformer_edge_targets = pct.edge_targets
                    conformer_graph_targets = pct.graph_targets
                    conformer_energies = pct.energies
                    conformer_coords = pct.coords
                    conformer_count = len(pct.coords)
                    node_target = boltzmann_average(pct.node_targets, pct.energies)
                    edge_target = boltzmann_average(pct.edge_targets, pct.energies)
                    graph_target = boltzmann_average(pct.graph_targets, pct.energies)

                aux_target = None
                if aux_available:
                    aux_target = torch.tensor(
                        [float(row.logP), float(row.qed), float(row.SAS)], dtype=torch.float32
                    )
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
                        conformer_node_targets=conformer_node_targets,
                        conformer_edge_targets=conformer_edge_targets,
                        conformer_graph_targets=conformer_graph_targets,
                    )
                )
```

Note: this replaces the previous `try/except` that fell back to proxy targets when `_aggregate_targets` failed. In real-data mode a molecule whose extraction raises now propagates to the outer `except Exception` (line ~263) and is recorded in `skipped_mol_ids`, which is the intended behaviour.

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_quantum_dataset_loading.py -q`
Expected: PASS (3 passed)

- [ ] **Step 6: Run the existing quantum/minimal tests to confirm the proxy path is intact**

Run: `python -m pytest tests/ -q -k "quantum or minimal or dataset"`
Expected: PASS (no regressions)

- [ ] **Step 7: Commit**

```bash
git add qchem_gnn/minimal.py qchem_gnn/quantum_data.py tests/test_quantum_dataset_loading.py
git commit -m "feat(quantum): per-conformer dataset fields + Boltzmann-averaged molecule targets"
```

---

### Task 4: Expose per-atom node states from the 3D encoder

**Files:**
- Modify: `qchem_gnn/encoder3d.py:73-97`
- Test: `tests/test_encoder3d_nodes.py`

**Interfaces:**
- Produces: `Conformer3DEncoder.forward_with_nodes(atomic_numbers, edge_index, positions, node_conformer_index, num_conformers) -> tuple[torch.Tensor, torch.Tensor]` returning `(node_states [total_nodes, hidden_dim], conformer_embedding [num_conformers, hidden_dim])`. `forward(...)` keeps its current signature and return (the pooled conformer embedding) for back-compat.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_encoder3d_nodes.py
import torch

from qchem_gnn.encoder3d import Conformer3DEncoder


def _toy_inputs():
    atomic_numbers = torch.tensor([6, 8, 6, 8])  # 2 conformers, 2 atoms each
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    positions = torch.tensor(
        [[0.0, 0, 0], [1.0, 0, 0], [0.0, 0, 0], [1.2, 0, 0]], dtype=torch.float32
    )
    node_conformer_index = torch.tensor([0, 0, 1, 1])
    return atomic_numbers, edge_index, positions, node_conformer_index, 2


def test_forward_with_nodes_shapes_and_consistency():
    enc = Conformer3DEncoder(atom_vocab_size=128, hidden_dim=16, num_message_passing_steps=2)
    enc.eval()
    args = _toy_inputs()
    with torch.no_grad():
        node_states, conf_emb = enc.forward_with_nodes(*args)
        legacy = enc(*args)
    assert node_states.shape == (4, 16)
    assert conf_emb.shape == (2, 16)
    # the pooled embedding from forward_with_nodes matches the legacy forward()
    assert torch.allclose(conf_emb, legacy, atol=1e-6)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_encoder3d_nodes.py -q`
Expected: FAIL with `AttributeError: 'Conformer3DEncoder' object has no attribute 'forward_with_nodes'`

- [ ] **Step 3: Refactor `Conformer3DEncoder` to expose node states**

Replace the `forward` method (lines 73-97) with:

```python
    def forward_with_nodes(
        self,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        positions: torch.Tensor,
        node_conformer_index: torch.Tensor,
        num_conformers: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        counts = (
            torch.bincount(node_conformer_index, minlength=num_conformers)
            .clamp_min(1)
            .unsqueeze(-1)
            .to(pooled.device)
        )
        conformer_embedding = self.embedding_head(pooled / counts)
        return node_states, conformer_embedding

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        positions: torch.Tensor,
        node_conformer_index: torch.Tensor,
        num_conformers: int,
    ) -> torch.Tensor:
        _, conformer_embedding = self.forward_with_nodes(
            atomic_numbers, edge_index, positions, node_conformer_index, num_conformers
        )
        return conformer_embedding
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_encoder3d_nodes.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Run existing encoder/contrastive tests for no regressions**

Run: `python -m pytest tests/ -q -k "encoder or contrastive or conformer"`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add qchem_gnn/encoder3d.py tests/test_encoder3d_nodes.py
git commit -m "feat(quantum): expose per-atom node states from 3D conformer encoder"
```

---

### Task 5: Quantum teacher heads and per-conformer target assembly

**Files:**
- Create: `qchem_gnn/teacher_heads.py`
- Test: `tests/test_teacher_heads.py`

**Interfaces:**
- Consumes: `Conformer3DEncoder.forward_with_nodes` (Task 4); `MinimalQuantumExample.conformer_node_targets/_edge_targets/_graph_targets/_energies` (Task 3).
- Produces:
  - `class QuantumTeacherHeads(nn.Module)` — `__init__(hidden_dim: int)`; `forward(node_states, edge_index, conformer_embedding) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]` returning `(node_pred [total_nodes,1], edge_pred [total_edges,1], graph_pred [num_conformers,2])`.
  - `teacher_loss(node_pred, edge_pred, graph_pred, node_target, edge_target, graph_target) -> torch.Tensor` — sum of the three MSEs.
  - `assemble_conformer_targets(examples: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]` returning `(node_target [ΣC·N,1], edge_target [ΣC·E,1], graph_target [ΣC,2], energies [ΣC])`, concatenated in the same molecule-then-conformer order that `ConformerEncoderBatch.from_molecule_conformers` uses.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_teacher_heads.py
import torch

from qchem_gnn.teacher_heads import (
    QuantumTeacherHeads,
    assemble_conformer_targets,
    teacher_loss,
)


class _FakeExample:
    def __init__(self, c, n, e):
        self.conformer_node_targets = torch.randn(c, n, 1)
        self.conformer_edge_targets = torch.randn(c, e, 1)
        self.conformer_graph_targets = torch.randn(c, 2)
        self.conformer_energies = torch.randn(c)


def test_assemble_orders_molecule_then_conformer():
    ex0 = _FakeExample(c=2, n=3, e=4)
    ex1 = _FakeExample(c=1, n=3, e=4)
    node_t, edge_t, graph_t, energies = assemble_conformer_targets([ex0, ex1])
    assert node_t.shape == (2 * 3 + 1 * 3, 1)
    assert edge_t.shape == (2 * 4 + 1 * 4, 1)
    assert graph_t.shape == (3, 2)
    assert energies.shape == (3,)
    # first molecule's first conformer node block matches its source
    assert torch.allclose(node_t[:3], ex0.conformer_node_targets[0])


def test_heads_output_shapes():
    heads = QuantumTeacherHeads(hidden_dim=16)
    node_states = torch.randn(6, 16)            # 2 conformers x 3 atoms
    edge_index = torch.tensor([[0, 1, 3, 4], [1, 0, 4, 3]])
    conf_emb = torch.randn(2, 16)
    node_pred, edge_pred, graph_pred = heads(node_states, edge_index, conf_emb)
    assert node_pred.shape == (6, 1)
    assert edge_pred.shape == (4, 1)
    assert graph_pred.shape == (2, 2)


def test_teacher_loss_decreases_on_overfit():
    torch.manual_seed(0)
    heads = QuantumTeacherHeads(hidden_dim=16)
    node_states = torch.randn(6, 16)
    edge_index = torch.tensor([[0, 1, 3, 4], [1, 0, 4, 3]])
    conf_emb = torch.randn(2, 16)
    node_t = torch.randn(6, 1)
    edge_t = torch.randn(4, 1)
    graph_t = torch.randn(2, 2)
    opt = torch.optim.Adam(heads.parameters(), lr=0.05)

    def step():
        opt.zero_grad()
        np_, ep_, gp_ = heads(node_states, edge_index, conf_emb)
        loss = teacher_loss(np_, ep_, gp_, node_t, edge_t, graph_t)
        loss.backward()
        opt.step()
        return float(loss)

    first = step()
    for _ in range(200):
        last = step()
    assert last < first * 0.5
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_teacher_heads.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'qchem_gnn.teacher_heads'`

- [ ] **Step 3: Implement `teacher_heads.py`**

```python
# qchem_gnn/teacher_heads.py
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class QuantumTeacherHeads(nn.Module):
    """Per-conformer quantum prediction heads on top of the 3D encoder."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.node_head = nn.Linear(hidden_dim, 1)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        conformer_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_pred = self.node_head(node_states)
        src, dst = edge_index
        edge_pred = self.edge_head(
            torch.cat([node_states[src], node_states[dst]], dim=-1)
        )
        graph_pred = self.graph_head(conformer_embedding)
        return node_pred, edge_pred, graph_pred


def teacher_loss(
    node_pred: torch.Tensor,
    edge_pred: torch.Tensor,
    graph_pred: torch.Tensor,
    node_target: torch.Tensor,
    edge_target: torch.Tensor,
    graph_target: torch.Tensor,
) -> torch.Tensor:
    return (
        F.mse_loss(node_pred, node_target)
        + F.mse_loss(edge_pred, edge_target)
        + F.mse_loss(graph_pred, graph_target)
    )


def assemble_conformer_targets(
    examples: list,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate per-conformer targets in molecule-then-conformer order.

    Matches the node/edge/conformer ordering produced by
    ``ConformerEncoderBatch.from_molecule_conformers``.
    """
    node_targets: list[torch.Tensor] = []
    edge_targets: list[torch.Tensor] = []
    graph_targets: list[torch.Tensor] = []
    energies: list[torch.Tensor] = []

    for example in examples:
        node = example.conformer_node_targets  # [C, N, 1]
        edge = example.conformer_edge_targets  # [C, E, 1]
        node_targets.append(node.reshape(-1, node.shape[-1]))
        edge_targets.append(edge.reshape(-1, edge.shape[-1]))
        graph_targets.append(example.conformer_graph_targets)
        energies.append(example.conformer_energies)

    return (
        torch.cat(node_targets, dim=0),
        torch.cat(edge_targets, dim=0),
        torch.cat(graph_targets, dim=0),
        torch.cat(energies, dim=0),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_teacher_heads.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/teacher_heads.py tests/test_teacher_heads.py
git commit -m "feat(quantum): per-conformer teacher heads, loss, and target assembly"
```

---

### Task 6: Wire the teacher into contrastive pretraining + real-data config

**Files:**
- Modify: `qchem_gnn/contrastive_pretrain.py` (signature + training loop, lines 53-171)
- Create: `configs/contrastive_pretrain_quantum.yaml`
- Test: `tests/test_contrastive_quantum.py`

**Interfaces:**
- Consumes: `QuantumTeacherHeads`, `teacher_loss`, `assemble_conformer_targets` (Task 5); `Conformer3DEncoder.forward_with_nodes` (Task 4); `boltzmann_average` (Task 1); per-conformer example fields (Task 3).
- Produces: `contrastive_pretrain_on_dataset(...)` gains keyword args `teacher_weight: float = 1.0` and `energy_temperature: float = 298.15`. The 2D model's `node_targets` is inferred from the data (`dataset.examples[0].node_target.shape[-1]`) so chelpg-only (`1`) and proxy (`2`) both work. When examples carry per-conformer targets, the loop adds `teacher_weight * teacher_loss(...)` and pools the 3D embedding with Boltzmann weights.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contrastive_quantum.py
from pathlib import Path

import h5py
import torch

from qchem_gnn.contrastive_pretrain import contrastive_pretrain_on_dataset
from qchem_gnn.quantum_data import load_quantum_zinc_dataset
from tests._quantum_fixtures import write_synthetic_results_h5


def _make_dataset(tmp_path):
    (tmp_path / "subset_044.csv").write_text(
        "smiles,logP,qed,SAS\n"
        "CO,0.0,0.5,2.0\nCCO,0.1,0.6,2.1\nCCN,0.2,0.7,2.2\nCCC,0.3,0.4,1.9\n"
    )
    # one molecule per group; reuse the same schema for 4 rows
    h5_path = tmp_path / "results_044.h5"
    with h5py.File(h5_path, "w") as handle:
        for idx, smiles, n in [
            (0, "CO", 6), (1, "CCO", 9), (2, "CCN", 10), (3, "CCC", 11)
        ]:
            tmp_single = tmp_path / f"_tmp_{idx}.h5"
            write_synthetic_results_h5(
                tmp_single, mol_id=f"subset_44_idx_{idx}", smiles=smiles, n_atoms=n, seed=idx
            )
            with h5py.File(tmp_single, "r") as src:
                src.copy(src[f"subset_44_idx_{idx}"], handle, f"subset_44_idx_{idx}")
    with h5py.File(h5_path, "r") as handle:
        return load_quantum_zinc_dataset(
            tmp_path / "subset_044.csv",
            geometry_path=tmp_path / "coords_044.pkl",
            limit=4,
            results_handle=handle,
            use_results=True,
        )


def test_contrastive_with_teacher_runs(tmp_path):
    dataset = _make_dataset(tmp_path)
    assert len(dataset.examples) == 4
    result = contrastive_pretrain_on_dataset(
        dataset,
        hidden_dim=16,
        hidden_dim_3d=16,
        epochs=3,
        batch_size=4,
        teacher_weight=1.0,
        conformer_pool_mode="energy",
        seed=0,
    )
    assert len(result.loss_history) == 3
    assert all(torch.isfinite(torch.tensor(v)) for v in result.loss_history)
    # the 2D student head matches chelpg-only target dim (1)
    assert result.model.atom_head.out_features == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_contrastive_quantum.py -q`
Expected: FAIL — `contrastive_pretrain_on_dataset()` has no `teacher_weight` argument.

- [ ] **Step 3: Update the imports and signature in `contrastive_pretrain.py`**

At the top, add:

```python
from .boltzmann import boltzmann_average
from .teacher_heads import QuantumTeacherHeads, assemble_conformer_targets, teacher_loss
```

Change the function signature to add the two new keyword args (insert after `temperature: float = 0.1,`):

```python
    teacher_weight: float = 1.0,
    energy_temperature: float = 298.15,
```

- [ ] **Step 4: Infer the student node-target dim and build the teacher**

Replace the model construction block (lines 75-94) with:

```python
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
```

- [ ] **Step 5: Replace the contrastive block in the training loop with teacher + Boltzmann pooling**

Replace the block that builds `with_coords`, computes `contrastive`, and forms `total` (lines 114-144, through the original `total = supervised_weight * supervised + contrastive_weight * contrastive` line) with:

```python
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
                    contrastive = info_nce_contrastive_loss(
                        proj_2d(molecule_2d), proj_3d(molecule_3d), temperature=temperature
                    )

            total = (
                supervised_weight * supervised
                + contrastive_weight * contrastive
                + teacher_weight * teacher_term
            )
```

Add this helper near the top of the module (after the imports, before `ProjectionHead`):

```python
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
```

Also update the epoch bookkeeping line that records `epoch_contrastive` to keep using `float(contrastive.item())` (unchanged), and leave the rest of the function (final embedding computation, return value) as-is.

- [ ] **Step 6: Run the new test to verify it passes**

Run: `python -m pytest tests/test_contrastive_quantum.py -q`
Expected: PASS (1 passed)

- [ ] **Step 7: Create the real-data config**

```yaml
# configs/contrastive_pretrain_quantum.yaml
command: contrastive-pretrain

dataset:
  dataset_root: zinc-250k
  subset_ids: [44]
  limit_per_shard: 64
  use_results: true

model:
  hidden_dim: 64
  message_passing_steps: 3

training:
  epochs: 100
  learning_rate: 0.005

contrastive:
  batch_size: 16
  supervised_weight: 1.0
  contrastive_weight: 1.0
  teacher_weight: 1.0
  temperature: 0.1
  energy_temperature: 298.15
  hidden_dim_3d: 64
  num_rbf: 16
  cutoff: 5.0
  message_passing_steps_3d: 3
  conformer_pool_mode: energy
  seed: 0

outputs:
  checkpoint: runs/contrastive_quantum.pt
```

- [ ] **Step 8: Verify the CLI accepts the new contrastive keys**

Check `qchem_gnn/cli.py` around lines 176-200 where `contrastive_cfg` is assembled from the YAML `contrastive` block. Confirm `teacher_weight` and `energy_temperature` are forwarded to `contrastive_pretrain_on_dataset`. If the handler copies a fixed key list, add `("teacher_weight", "teacher_weight")` and `("energy_temperature", "energy_temperature")` to it. Then run:

Run: `python -m qchem_gnn.cli contrastive-pretrain --config configs/contrastive_pretrain_quantum.yaml`
Expected: training runs to completion on `results_044.h5` and writes `runs/contrastive_quantum.pt`. (Skip/abbreviate by lowering `limit_per_shard`/`epochs` if wall-clock is a concern; the goal is a clean end-to-end run on real data.)

- [ ] **Step 9: Run the full suite for no regressions**

Run: `python -m pytest tests -q`
Expected: PASS (previous 120 + the new tests from Tasks 1-6)

- [ ] **Step 10: Commit**

```bash
git add qchem_gnn/contrastive_pretrain.py qchem_gnn/cli.py configs/contrastive_pretrain_quantum.yaml tests/test_contrastive_quantum.py
git commit -m "feat(quantum): per-conformer teacher + energy-weighted pooling in contrastive pretraining"
```

---

## Notes for the implementer

- After Task 6, the shipped `example_contrastive.pt` is unchanged. Regenerating a quantum-trained backbone and re-running the downstream solubility benchmarks is a follow-up, not part of this plan.
- Conformer counts vary (1–5). The teacher path requires `len(usable) >= 2` for in-batch negatives; molecules with a single conformer still contribute teacher and supervised signal but are only included when the batch has at least two usable molecules. This matches the existing contrastive guard.
- Do not load the whole 2.4 GB shard into memory. h5py reads lazily; the loader opens groups one molecule at a time.
