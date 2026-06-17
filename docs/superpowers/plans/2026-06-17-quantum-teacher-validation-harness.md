# Quantum-Teacher Validation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-command harness that measures whether the per-conformer quantum teacher improves the 2D backbone — extrinsically (ESOL transfer via a matched ablation) and intrinsically (does the teacher predict held-out conformer quantum labels).

**Architecture:** A single new module `qchem_gnn/validation.py` orchestrates and reports only — it splits the dataset into a fixed pretrain/holdout, trains backbones for two matched arms × N seeds via the existing `contrastive_pretrain_on_dataset`, evaluates the trained teacher on the holdout, runs the existing adapt subsystem as the downstream probe, and aggregates everything into a `report.{json,md}`. One small enabling change exposes the trained `teacher`/`encoder3d` on the pretraining result.

**Tech Stack:** Python 3.13+, PyTorch, h5py, RDKit, pandas, numpy, PyYAML, pytest.

## Global Constraints

- Matched ablation: the two arms differ in EXACTLY `teacher_weight` (baseline `0.0` vs quantum `1.0`) and `conformer_pool_mode` (baseline `mean` vs quantum `energy`); every other pretraining setting is identical and both use `use_results: true`.
- Training data is all 325 complete molecules in `zinc-250k/results/results_044.h5`; `limit_per_shard` must be ≥ 325. Do not depend on the 7 GB shards.
- Default seeds are `[0, 1, 2]` (3 per arm); the seed list is configurable.
- Probes: `mlp_head` (frozen backbone — decisive) and `finetune` (headline only); adapt split/training seed is `42`.
- Verdict keys off frozen `mlp_head` MAE: the teacher *helps* iff `mean_baseline − mean_quantum > sqrt(std_baseline² + std_quantum²)`; `n < 2` successful seeds in either arm → `insufficient seeds`; `n == 0` → `n/a`. This is a stated heuristic for small n, NOT a significance test.
- Intrinsic metric: Pearson r and MAE for `chelpg`, `energy`, `iso_polarizability` (and `wbi`) on a holdout split that is computed ONCE and reused across all arms and seeds.
- `ContrastivePretrainingResult` gains in-memory fields `teacher` and `encoder3d` (default `None`); they are NEVER serialized — the checkpoint and inference contracts are unchanged.
- The holdout split is deterministic from its own seed, independent of the pretraining seed.
- Backbone checkpoints are saved as a torch dict `{"model_config": {...}, "model_state_dict": {...}}` so the adapt `load_backbone` can load them; `model_config` includes `node_targets` inferred from the data (`int(pretrain_ds.examples[0].node_target.shape[-1])`).
- Skip-if-exists caching, per-cell `try/except` isolation, a non-finite backbone guard, and an `overwrite` flag are required.
- The proxy path and the existing test suite (~140 tests) must still pass. Reuse `tests/_quantum_fixtures.py`.

---

### Task 1: Expose the trained teacher and encoder on the pretraining result

**Files:**
- Modify: `qchem_gnn/contrastive_pretrain.py:31-40` (add fields) and `:213-222` (populate them)
- Create: `tests/_validation_fixtures.py`
- Test: `tests/test_validation_result_fields.py`

**Interfaces:**
- Consumes: `contrastive_pretrain_on_dataset` (existing), `write_synthetic_results_h5` (from `tests/_quantum_fixtures.py`), `load_quantum_zinc_dataset` (existing).
- Produces:
  - `ContrastivePretrainingResult.teacher: nn.Module | None = None` and `.encoder3d: nn.Module | None = None`, populated with the trained `QuantumTeacherHeads` and `Conformer3DEncoder`.
  - `tests/_validation_fixtures.py::make_tiny_quantum_dataset(tmp_path, mols=None) -> MinimalQuantumDataset` — a small real-schema dataset for all validation tests.

- [ ] **Step 1: Write the shared tiny-dataset fixture builder**

```python
# tests/_validation_fixtures.py
from __future__ import annotations

from pathlib import Path

import h5py

from qchem_gnn.quantum_data import load_quantum_zinc_dataset
from tests._quantum_fixtures import write_synthetic_results_h5


def make_tiny_quantum_dataset(tmp_path, mols=None):
    """Build a small MinimalQuantumDataset from synthetic real-schema HDF5.

    Each (smiles, n_atoms) pair becomes one molecule group with two conformers.
    ``CO`` -> 6 atoms, ``CCO`` -> 9, ``CCN`` -> 10, ``CCC`` -> 11 (explicit H).
    """
    mols = mols or [("CO", 6), ("CCO", 9), ("CCN", 10), ("CCC", 11)]
    tmp_path = Path(tmp_path)
    csv_path = tmp_path / "subset_044.csv"
    csv_path.write_text(
        "smiles,logP,qed,SAS\n"
        + "".join(f"{smiles},0.0,0.5,2.0\n" for smiles, _ in mols)
    )
    h5_path = tmp_path / "results_044.h5"
    with h5py.File(h5_path, "w") as handle:
        for idx, (smiles, n_atoms) in enumerate(mols):
            tmp_single = tmp_path / f"_tmp_{idx}.h5"
            write_synthetic_results_h5(
                tmp_single,
                mol_id=f"subset_44_idx_{idx}",
                smiles=smiles,
                n_atoms=n_atoms,
                seed=idx,
            )
            with h5py.File(tmp_single, "r") as src:
                src.copy(src[f"subset_44_idx_{idx}"], handle, f"subset_44_idx_{idx}")
    with h5py.File(h5_path, "r") as handle:
        return load_quantum_zinc_dataset(
            csv_path,
            geometry_path=tmp_path / "coords_044.pkl",
            limit=len(mols),
            results_handle=handle,
            use_results=True,
        )
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_validation_result_fields.py
from qchem_gnn.contrastive_pretrain import contrastive_pretrain_on_dataset
from qchem_gnn.encoder3d import Conformer3DEncoder
from qchem_gnn.teacher_heads import QuantumTeacherHeads
from tests._validation_fixtures import make_tiny_quantum_dataset


def test_pretrain_result_exposes_teacher_and_encoder(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    result = contrastive_pretrain_on_dataset(
        dataset,
        hidden_dim=16,
        hidden_dim_3d=16,
        epochs=2,
        batch_size=4,
        teacher_weight=1.0,
        conformer_pool_mode="energy",
        seed=0,
    )
    assert isinstance(result.teacher, QuantumTeacherHeads)
    assert isinstance(result.encoder3d, Conformer3DEncoder)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest tests/test_validation_result_fields.py -q`
Expected: FAIL with `AttributeError: 'ContrastivePretrainingResult' object has no attribute 'teacher'`

- [ ] **Step 4: Add the two fields to the result dataclass**

In `qchem_gnn/contrastive_pretrain.py`, extend `ContrastivePretrainingResult` (currently ending at `global_step: int`):

```python
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
```

- [ ] **Step 5: Populate the fields in the return statement**

In the same file, in the `return ContrastivePretrainingResult(...)` at the end of `contrastive_pretrain_on_dataset`, add the two fields:

```python
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
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `python -m pytest tests/test_validation_result_fields.py -q`
Expected: PASS (1 passed)

- [ ] **Step 7: Run the contrastive suite for no regressions**

Run: `python -m pytest tests/ -q -k "contrastive or checkpoint"`
Expected: PASS — the new fields default to `None` and are not serialized, so checkpoint tests are unaffected.

- [ ] **Step 8: Commit**

```bash
git add qchem_gnn/contrastive_pretrain.py tests/_validation_fixtures.py tests/test_validation_result_fields.py
git commit -m "feat(validation): expose trained teacher and encoder3d on pretraining result"
```

---

### Task 2: Deterministic pretrain/holdout split

**Files:**
- Create: `qchem_gnn/validation.py`
- Test: `tests/test_validation_split.py`

**Interfaces:**
- Consumes: `MinimalQuantumDataset` (from `qchem_gnn.minimal`).
- Produces: `split_holdout(dataset, fraction: float, seed: int) -> tuple[MinimalQuantumDataset, MinimalQuantumDataset]` returning `(pretrain_dataset, holdout_dataset)`. The holdout has `max(1, round(n * fraction))` molecules; the split is deterministic for a fixed seed and the two subsets are disjoint and cover all examples.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation_split.py
import pytest

from qchem_gnn.validation import split_holdout
from tests._validation_fixtures import make_tiny_quantum_dataset


def test_split_is_deterministic_disjoint_and_sized(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)  # 4 molecules
    pre_a, hold_a = split_holdout(dataset, fraction=0.5, seed=7)
    pre_b, hold_b = split_holdout(dataset, fraction=0.5, seed=7)

    ids_pre = {ex.mol_id for ex in pre_a.examples}
    ids_hold = {ex.mol_id for ex in hold_a.examples}

    # deterministic
    assert [ex.mol_id for ex in hold_a.examples] == [ex.mol_id for ex in hold_b.examples]
    assert [ex.mol_id for ex in pre_a.examples] == [ex.mol_id for ex in pre_b.examples]
    # disjoint and complete
    assert ids_pre.isdisjoint(ids_hold)
    assert ids_pre | ids_hold == {ex.mol_id for ex in dataset.examples}
    # sized: round(4 * 0.5) == 2
    assert len(hold_a.examples) == 2
    assert len(pre_a.examples) == 2


def test_different_seed_changes_holdout(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    _, hold_0 = split_holdout(dataset, fraction=0.5, seed=0)
    _, hold_1 = split_holdout(dataset, fraction=0.5, seed=999)
    # at least the ordering/content differs for some seed pair
    assert [e.mol_id for e in hold_0.examples] != [e.mol_id for e in hold_1.examples]


def test_holdout_at_least_one(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    _, hold = split_holdout(dataset, fraction=0.01, seed=0)
    assert len(hold.examples) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_validation_split.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'qchem_gnn.validation'`

- [ ] **Step 3: Create `validation.py` with the split**

```python
# qchem_gnn/validation.py
from __future__ import annotations

import torch

from .minimal import MinimalQuantumDataset


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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_validation_split.py -q`
Expected: PASS (3 passed). If `test_different_seed_changes_holdout` is flaky for a degenerate seed pair, the seeds `0`/`999` chosen here produce different holdouts for the 4-molecule fixture.

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/validation.py tests/test_validation_split.py
git commit -m "feat(validation): deterministic pretrain/holdout split"
```

---

### Task 3: Intrinsic teacher evaluation

**Files:**
- Modify: `qchem_gnn/validation.py` (add imports + `_pearson_mae` + `evaluate_teacher`)
- Test: `tests/test_validation_teacher_eval.py`

**Interfaces:**
- Consumes: `ConformerEncoderBatch` (from `qchem_gnn.conformer`), `Conformer3DEncoder.forward_with_nodes` (existing), `QuantumTeacherHeads` + `assemble_conformer_targets` (from `qchem_gnn.teacher_heads`), `split_holdout` (Task 2).
- Produces: `evaluate_teacher(teacher, encoder3d, holdout_examples: list) -> dict` returning
  `{"chelpg": {"r": float, "mae": float}, "energy": {...}, "iso_polarizability": {...}, "wbi": {...}}`.
  Skips examples lacking per-conformer targets; raises `ValueError` if NONE are usable.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation_teacher_eval.py
import pytest
import torch

from qchem_gnn.conformer import ConformerEncoderBatch
from qchem_gnn.encoder3d import Conformer3DEncoder
from qchem_gnn.teacher_heads import (
    QuantumTeacherHeads,
    assemble_conformer_targets,
    teacher_loss,
)
from qchem_gnn.validation import evaluate_teacher, split_holdout
from tests._validation_fixtures import make_tiny_quantum_dataset


def test_evaluate_teacher_returns_finite_metrics(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    _, holdout = split_holdout(dataset, fraction=0.5, seed=0)
    enc = Conformer3DEncoder(atom_vocab_size=128, hidden_dim=16, num_message_passing_steps=2)
    teacher = QuantumTeacherHeads(hidden_dim=16)
    metrics = evaluate_teacher(teacher, enc, holdout.examples)
    for prop in ("chelpg", "energy", "iso_polarizability", "wbi"):
        assert torch.isfinite(torch.tensor(metrics[prop]["r"]))
        assert torch.isfinite(torch.tensor(metrics[prop]["mae"]))
        assert -1.0 <= metrics[prop]["r"] <= 1.0


def test_metric_reflects_learning(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    _, holdout = split_holdout(dataset, fraction=0.5, seed=0)
    examples = holdout.examples
    enc = Conformer3DEncoder(atom_vocab_size=128, hidden_dim=16, num_message_passing_steps=2)
    teacher = QuantumTeacherHeads(hidden_dim=16)

    random_metrics = evaluate_teacher(teacher, enc, examples)

    # Fit the teacher heads on the holdout targets with the encoder frozen.
    batch = ConformerEncoderBatch.from_molecule_conformers(
        [ex.graph for ex in examples],
        [ex.conformer_coords for ex in examples],
        conformer_energies=[ex.conformer_energies for ex in examples],
    )
    with torch.no_grad():
        node_states, conf_emb = enc.forward_with_nodes(
            batch.atomic_numbers, batch.edge_index, batch.positions,
            batch.node_conformer_index, batch.num_conformers,
        )
    node_t, edge_t, graph_t, _ = assemble_conformer_targets(examples)
    opt = torch.optim.Adam(teacher.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        np_, ep_, gp_ = teacher(node_states, batch.edge_index, conf_emb)
        teacher_loss(np_, ep_, gp_, node_t, edge_t, graph_t).backward()
        opt.step()

    fit_metrics = evaluate_teacher(teacher, enc, examples)
    assert fit_metrics["chelpg"]["mae"] < random_metrics["chelpg"]["mae"]


def test_raises_when_no_usable_examples():
    with pytest.raises(ValueError):
        enc = Conformer3DEncoder(atom_vocab_size=128, hidden_dim=16, num_message_passing_steps=2)
        evaluate_teacher(QuantumTeacherHeads(hidden_dim=16), enc, [])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_validation_teacher_eval.py -q`
Expected: FAIL with `ImportError: cannot import name 'evaluate_teacher'`

- [ ] **Step 3: Implement `_pearson_mae` and `evaluate_teacher`**

Add to the top imports of `qchem_gnn/validation.py`:

```python
from .conformer import ConformerEncoderBatch
from .teacher_heads import assemble_conformer_targets
```

Then add:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_validation_teacher_eval.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/validation.py tests/test_validation_teacher_eval.py
git commit -m "feat(validation): intrinsic teacher evaluation (Pearson r + MAE)"
```

---

### Task 4: Aggregation and report rendering

**Files:**
- Modify: `qchem_gnn/validation.py` (add `_mean_std`, `_verdict`, `aggregate_results`, `render_report`)
- Test: `tests/test_validation_aggregate.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (operates on plain row dicts).
- Produces:
  - `aggregate_results(extrinsic_rows: list[dict], intrinsic_rows: list[dict]) -> dict` with shape
    `{"extrinsic": {method: {"baseline": {...}, "quantum": {...}}}, "verdict": {...}, "intrinsic": {arm: {prop: {...}}}}`.
  - `render_report(aggregate: dict) -> str` (markdown).
  - Extrinsic row schema: `{"arm": str, "seed": int, "method": str, "status": "ok"|"failed", "mae": float|None, "r2": float|None}`.
  - Intrinsic row schema: `{"arm": str, "seed": int, "status": "ok"|"failed", "properties": {prop: {"r": float, "mae": float}}}`.
  - Verdict: `{"method": "mlp_head", "metric": "mae", "delta": float|None, "combined_std": float|None, "result": "helps"|"within noise"|"insufficient seeds"|"n/a"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation_aggregate.py
from qchem_gnn.validation import aggregate_results, render_report


def _ex(arm, seed, method, mae, r2):
    return {"arm": arm, "seed": seed, "method": method, "status": "ok", "mae": mae, "r2": r2}


def test_mean_std_and_helps_verdict():
    # baseline mlp_head MAE ~1.20, quantum ~1.00, tight spread -> delta beats combined std
    rows = [
        _ex("baseline", 0, "mlp_head", 1.20, 0.40),
        _ex("baseline", 1, "mlp_head", 1.22, 0.39),
        _ex("baseline", 2, "mlp_head", 1.18, 0.41),
        _ex("quantum", 0, "mlp_head", 1.00, 0.52),
        _ex("quantum", 1, "mlp_head", 1.01, 0.51),
        _ex("quantum", 2, "mlp_head", 0.99, 0.53),
    ]
    agg = aggregate_results(rows, [])
    base = agg["extrinsic"]["mlp_head"]["baseline"]
    assert abs(base["mae_mean"] - 1.20) < 1e-6
    assert base["n"] == 3
    assert agg["verdict"]["result"] == "helps"
    assert agg["verdict"]["delta"] > 0


def test_within_noise_verdict():
    rows = [
        _ex("baseline", 0, "mlp_head", 1.20, 0.40),
        _ex("baseline", 1, "mlp_head", 0.90, 0.55),  # wide spread
        _ex("baseline", 2, "mlp_head", 1.50, 0.20),
        _ex("quantum", 0, "mlp_head", 1.15, 0.42),
        _ex("quantum", 1, "mlp_head", 0.95, 0.50),
        _ex("quantum", 2, "mlp_head", 1.45, 0.25),
    ]
    agg = aggregate_results(rows, [])
    assert agg["verdict"]["result"] == "within noise"


def test_insufficient_seeds_verdict():
    rows = [
        _ex("baseline", 0, "mlp_head", 1.2, 0.4),
        _ex("quantum", 0, "mlp_head", 1.0, 0.5),
    ]
    agg = aggregate_results(rows, [])
    assert agg["verdict"]["result"] == "insufficient seeds"


def test_failed_rows_excluded_and_na_when_empty():
    rows = [
        {"arm": "baseline", "seed": 0, "method": "mlp_head",
         "status": "failed", "mae": None, "r2": None},
        _ex("quantum", 0, "mlp_head", 1.0, 0.5),
        _ex("quantum", 1, "mlp_head", 1.0, 0.5),
    ]
    agg = aggregate_results(rows, [])
    assert agg["extrinsic"]["mlp_head"]["baseline"]["n"] == 0
    assert agg["verdict"]["result"] == "n/a"


def test_render_report_contains_sections():
    rows = [_ex("baseline", 0, "mlp_head", 1.2, 0.4), _ex("quantum", 0, "mlp_head", 1.0, 0.5)]
    intr = [{"arm": "quantum", "seed": 0, "status": "ok",
             "properties": {"chelpg": {"r": 0.7, "mae": 0.1},
                            "energy": {"r": 0.8, "mae": 0.2},
                            "iso_polarizability": {"r": 0.6, "mae": 0.3},
                            "wbi": {"r": 0.5, "mae": 0.4}}}]
    text = render_report(aggregate_results(rows, intr))
    assert "Extrinsic" in text
    assert "Intrinsic" in text
    assert "Verdict" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_validation_aggregate.py -q`
Expected: FAIL with `ImportError: cannot import name 'aggregate_results'`

- [ ] **Step 3: Implement aggregation and rendering**

Add `import statistics` to the top of `qchem_gnn/validation.py`, then add:

```python
ARMS = ("baseline", "quantum")
DECISIVE_METHOD = "mlp_head"
DECISIVE_METRIC = "mae"
INTRINSIC_PROPERTIES = ("chelpg", "energy", "iso_polarizability", "wbi")


def _mean_std(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "n": 0}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n >= 2 else None
    return {"mean": mean, "std": std, "n": n}


def _verdict(baseline: dict, quantum: dict) -> dict:
    # Lower MAE is better; the teacher "helps" if baseline_mean - quantum_mean
    # exceeds the combined seed noise sqrt(std_b^2 + std_q^2).
    out = {"method": DECISIVE_METHOD, "metric": DECISIVE_METRIC,
           "delta": None, "combined_std": None, "result": "n/a"}
    if baseline["n"] == 0 or quantum["n"] == 0:
        out["result"] = "n/a"
        return out
    if baseline["std"] is None or quantum["std"] is None:
        out["result"] = "insufficient seeds"
        return out
    delta = baseline["mean"] - quantum["mean"]
    combined = (baseline["std"] ** 2 + quantum["std"] ** 2) ** 0.5
    out["delta"] = delta
    out["combined_std"] = combined
    out["result"] = "helps" if delta > combined else "within noise"
    return out


def aggregate_results(extrinsic_rows: list[dict], intrinsic_rows: list[dict]) -> dict:
    methods = sorted({r["method"] for r in extrinsic_rows})
    extrinsic: dict = {}
    for method in methods:
        extrinsic[method] = {}
        for arm in ARMS:
            ok = [r for r in extrinsic_rows
                  if r["method"] == method and r["arm"] == arm and r["status"] == "ok"]
            mae = _mean_std([r["mae"] for r in ok])
            r2 = _mean_std([r["r2"] for r in ok])
            extrinsic[method][arm] = {
                "mae_mean": mae["mean"], "mae_std": mae["std"],
                "r2_mean": r2["mean"], "r2_std": r2["std"], "n": mae["n"],
            }

    if DECISIVE_METHOD in extrinsic:
        decisive = extrinsic[DECISIVE_METHOD]
        verdict = _verdict(
            {"mean": decisive["baseline"]["mae_mean"], "std": decisive["baseline"]["mae_std"],
             "n": decisive["baseline"]["n"]},
            {"mean": decisive["quantum"]["mae_mean"], "std": decisive["quantum"]["mae_std"],
             "n": decisive["quantum"]["n"]},
        )
    else:
        verdict = {"method": DECISIVE_METHOD, "metric": DECISIVE_METRIC,
                   "delta": None, "combined_std": None, "result": "n/a"}

    intrinsic: dict = {}
    for arm in ARMS:
        ok = [r for r in intrinsic_rows if r["arm"] == arm and r["status"] == "ok"]
        intrinsic[arm] = {}
        for prop in INTRINSIC_PROPERTIES:
            rs = _mean_std([r["properties"][prop]["r"] for r in ok if prop in r["properties"]])
            ms = _mean_std([r["properties"][prop]["mae"] for r in ok if prop in r["properties"]])
            intrinsic[arm][prop] = {"r_mean": rs["mean"], "mae_mean": ms["mean"], "n": rs["n"]}

    return {"extrinsic": extrinsic, "verdict": verdict, "intrinsic": intrinsic}


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_report(aggregate: dict) -> str:
    lines: list[str] = ["# Quantum-Teacher Validation Report", ""]

    verdict = aggregate["verdict"]
    lines += ["## Verdict", "",
              f"- Decisive probe: `{verdict['method']}` {verdict['metric'].upper()}",
              f"- delta (baseline - quantum): {_fmt(verdict['delta'])}",
              f"- combined seed std: {_fmt(verdict['combined_std'])}",
              f"- **Result: {verdict['result']}**",
              "", "_Heuristic, not a significance test: 'helps' iff delta > combined std._", ""]

    lines += ["## Extrinsic (ESOL transfer)", "",
              "| Method | Arm | MAE (mean) | MAE (std) | R2 (mean) | n |",
              "|---|---|---|---|---|---|"]
    for method, arms in aggregate["extrinsic"].items():
        for arm in ARMS:
            s = arms[arm]
            lines.append(
                f"| {method} | {arm} | {_fmt(s['mae_mean'])} | {_fmt(s['mae_std'])} "
                f"| {_fmt(s['r2_mean'])} | {s['n']} |"
            )
    lines.append("")

    lines += ["## Intrinsic (teacher on held-out conformers)", "",
              "| Property | Arm | r (mean) | MAE (mean) | n |",
              "|---|---|---|---|---|"]
    for arm in ARMS:
        for prop in INTRINSIC_PROPERTIES:
            s = aggregate["intrinsic"][arm][prop]
            lines.append(
                f"| {prop} | {arm} | {_fmt(s['r_mean'])} | {_fmt(s['mae_mean'])} | {s['n']} |"
            )
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_validation_aggregate.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/validation.py tests/test_validation_aggregate.py
git commit -m "feat(validation): result aggregation, verdict logic, and report rendering"
```

---

### Task 5: Single-cell runner (train, save, intrinsic, probes) with caching

**Files:**
- Modify: `qchem_gnn/validation.py` (add imports + `_save_backbone` + `_run_probe` + `run_one_cell`)
- Test: `tests/test_validation_cell.py`

**Interfaces:**
- Consumes: `contrastive_pretrain_on_dataset` (Task 1 fields), `evaluate_teacher` (Task 3), the adapt subsystem (`resolve_adapt_config`, `run`).
- Produces: `run_one_cell(arm_name, arm_overrides, seed, pretrain_ds, holdout_examples, pretrain_cfg, probes, adapt_cfg, out_dir, overwrite=False) -> dict` returning `{"extrinsic": [extrinsic_row, ...], "intrinsic": intrinsic_row}` using the Task-4 row schemas. A backbone is saved to `out_dir/{arm_name}_s{seed}.pt` as `{"model_config": {...}, "model_state_dict": {...}}`; the intrinsic metrics cache to `out_dir/{arm_name}_s{seed}_intrinsic.json`; each probe's adapt summary caches to `out_dir/{arm_name}_s{seed}_{method}.json`. Existing artifacts are reused unless `overwrite`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation_cell.py
import json

import qchem_gnn.validation as validation
from qchem_gnn.validation import run_one_cell, split_holdout
from tests._validation_fixtures import make_tiny_quantum_dataset


def _tiny_adapt_cfg(tmp_path):
    csv = tmp_path / "esol_tiny.csv"
    rows = ["smiles,y"]
    for i, smi in enumerate(["CO", "CCO", "CCN", "CCC", "CCCC", "CCCO",
                             "CCCN", "CCCCO", "c1ccccc1", "CC", "CCCl", "CCBr"]):
        rows.append(f"{smi},{0.1 * i:.2f}")
    csv.write_text("\n".join(rows) + "\n")
    return {
        "dataset": {"csv": str(csv), "smiles_col": "smiles", "targets": ["y"]},
        "task": "regression",
        "adapter": {"hidden_dims": [8], "dropout": 0.0},
        "training": {"epochs": 2, "lr": 1.0e-3, "batch_size": 4, "patience": 5, "seed": 42},
        "split": {"test_frac": 0.25, "val_frac": 0.25, "seed": 42, "stratify": False},
    }


def _pretrain_cfg():
    return {"hidden_dim": 16, "message_passing_steps": 2, "epochs": 2,
            "learning_rate": 0.01, "batch_size": 4, "hidden_dim_3d": 16,
            "message_passing_steps_3d": 2}


def test_run_one_cell_produces_rows_and_artifacts(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    pretrain_ds, holdout = split_holdout(dataset, fraction=0.5, seed=0)
    out_dir = tmp_path / "out"
    cell = run_one_cell(
        "quantum", {"teacher_weight": 1.0, "conformer_pool_mode": "energy"}, 0,
        pretrain_ds, holdout.examples, _pretrain_cfg(),
        [{"method": "mlp_head"}], _tiny_adapt_cfg(tmp_path), out_dir,
    )
    assert (out_dir / "quantum_s0.pt").exists()
    assert (out_dir / "quantum_s0_intrinsic.json").exists()
    assert (out_dir / "quantum_s0_mlp_head.json").exists()
    assert cell["intrinsic"]["status"] == "ok"
    assert cell["extrinsic"][0]["method"] == "mlp_head"
    assert cell["extrinsic"][0]["status"] == "ok"
    assert cell["extrinsic"][0]["mae"] is not None


def test_caching_skips_retraining(tmp_path, monkeypatch):
    dataset = make_tiny_quantum_dataset(tmp_path)
    pretrain_ds, holdout = split_holdout(dataset, fraction=0.5, seed=0)
    out_dir = tmp_path / "out"
    args = ("quantum", {"teacher_weight": 1.0, "conformer_pool_mode": "energy"}, 0,
            pretrain_ds, holdout.examples, _pretrain_cfg(),
            [{"method": "mlp_head"}], _tiny_adapt_cfg(tmp_path), out_dir)

    run_one_cell(*args)  # first run trains + caches

    calls = {"n": 0}
    real = validation.contrastive_pretrain_on_dataset

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(validation, "contrastive_pretrain_on_dataset", spy)
    run_one_cell(*args)  # second run should reuse cache
    assert calls["n"] == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_validation_cell.py -q`
Expected: FAIL with `ImportError: cannot import name 'run_one_cell'`

- [ ] **Step 3: Implement the cell runner**

Add these imports to the top of `qchem_gnn/validation.py` (note: `import torch` is already present from Task 2 — do NOT add a second one):

```python
import json
import math
from pathlib import Path

from .adapt import resolve_adapt_config
from .adapt import run as adapt_run
from .contrastive_pretrain import contrastive_pretrain_on_dataset
```

Then add:

```python
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
        try:
            result = contrastive_pretrain_on_dataset(
                pretrain_ds, **_pretrain_kwargs(pretrain_cfg, arm_overrides, seed)
            )
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

    for probe in probes:
        method = probe["method"]
        report_path = out_dir / f"{arm_name}_s{seed}_{method}.json"
        try:
            if report_path.exists() and not overwrite:
                metrics = json.loads(report_path.read_text())["test_metrics"]
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_validation_cell.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/validation.py tests/test_validation_cell.py
git commit -m "feat(validation): single-cell runner with skip-if-exists caching and isolation"
```

---

### Task 6: Full driver, config parsing, CLI entry point, and the experiment config

**Files:**
- Modify: `qchem_gnn/validation.py` (add `_load_dataset`, `run_validation`, `load_validation_config`, `main`, `__main__` guard)
- Create: `configs/validate_quantum_teacher.yaml`
- Test: `tests/test_validation_end_to_end.py`

**Interfaces:**
- Consumes: `run_one_cell` (Task 5), `aggregate_results` + `render_report` (Task 4), `split_holdout` (Task 2), `load_quantum_zinc_subset_range` (from `qchem_gnn.quantum_data`).
- Produces:
  - `run_validation(cfg: dict, dataset=None, overwrite=False) -> dict` — loads the dataset (unless `dataset` is supplied), splits the holdout once, runs every `arm × seed` cell, writes `{outputs.report}.json` (raw rows + aggregate) and `{outputs.report}.md`, and returns the aggregate.
  - `load_validation_config(path) -> dict` — `yaml.safe_load` plus a check that `pretrain`, `arms`, `seeds`, `holdout`, `probes`, `adapt`, `outputs` are present.
  - `main(argv=None) -> int` — argparse `--config` (required) and `--overwrite` (flag); the `python -m qchem_gnn.validation` entry point.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation_end_to_end.py
import json

from qchem_gnn.validation import run_validation
from tests._validation_fixtures import make_tiny_quantum_dataset


def _tiny_esol_csv(tmp_path):
    csv = tmp_path / "esol_tiny.csv"
    rows = ["smiles,y"]
    for i, smi in enumerate(["CO", "CCO", "CCN", "CCC", "CCCC", "CCCO",
                             "CCCN", "CCCCO", "c1ccccc1", "CC", "CCCl", "CCBr"]):
        rows.append(f"{smi},{0.1 * i:.2f}")
    csv.write_text("\n".join(rows) + "\n")
    return csv


def test_run_validation_writes_report(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    out_dir = tmp_path / "validate"
    cfg = {
        "pretrain": {"hidden_dim": 16, "message_passing_steps": 2, "epochs": 2,
                     "learning_rate": 0.01, "batch_size": 4, "hidden_dim_3d": 16,
                     "message_passing_steps_3d": 2},
        "arms": {"baseline": {"teacher_weight": 0.0, "conformer_pool_mode": "mean"},
                 "quantum": {"teacher_weight": 1.0, "conformer_pool_mode": "energy"}},
        "seeds": [0],
        "holdout": {"fraction": 0.25, "seed": 1234},
        "probes": [{"method": "mlp_head"}],
        "adapt": {"dataset": {"csv": str(_tiny_esol_csv(tmp_path)),
                              "smiles_col": "smiles", "targets": ["y"]},
                  "task": "regression",
                  "adapter": {"hidden_dims": [8], "dropout": 0.0},
                  "training": {"epochs": 2, "lr": 1.0e-3, "batch_size": 4,
                               "patience": 5, "seed": 42},
                  "split": {"test_frac": 0.25, "val_frac": 0.25, "seed": 42,
                            "stratify": False}},
        "outputs": {"dir": str(out_dir), "report": str(out_dir / "report")},
    }
    aggregate = run_validation(cfg, dataset=dataset, overwrite=True)

    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.md").exists()
    assert "verdict" in aggregate
    saved = json.loads((out_dir / "report.json").read_text())
    assert "rows" in saved and "aggregate" in saved
    # one cell per arm at seed 0 -> two backbones written
    assert (out_dir / "baseline_s0.pt").exists()
    assert (out_dir / "quantum_s0.pt").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_validation_end_to_end.py -q`
Expected: FAIL with `ImportError: cannot import name 'run_validation'`

- [ ] **Step 3: Implement the driver, config loader, and CLI**

Add `import argparse` and `import sys` to the top of `qchem_gnn/validation.py`, then add:

```python
def _load_dataset(pretrain_cfg: dict):
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
    pretrain_ds, holdout = split_holdout(
        dataset, fraction=holdout_cfg["fraction"], seed=holdout_cfg["seed"]
    )

    out_dir = Path(cfg["outputs"]["dir"])
    extrinsic_rows: list[dict] = []
    intrinsic_rows: list[dict] = []
    for arm_name in ARMS:
        arm_overrides = cfg["arms"][arm_name]
        for seed in cfg["seeds"]:
            cell = run_one_cell(
                arm_name, arm_overrides, seed, pretrain_ds, holdout.examples,
                cfg["pretrain"], cfg["probes"], cfg["adapt"], out_dir, overwrite=overwrite,
            )
            extrinsic_rows.extend(cell["extrinsic"])
            intrinsic_rows.append(cell["intrinsic"])

    aggregate = aggregate_results(extrinsic_rows, intrinsic_rows)

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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_validation_end_to_end.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Create the experiment config**

```yaml
# configs/validate_quantum_teacher.yaml
command: validate-quantum-teacher

pretrain:
  dataset_root: zinc-250k
  subset_ids: [44]
  limit_per_shard: 400          # >= 325 so all complete molecules load
  hidden_dim: 64
  message_passing_steps: 3
  hidden_dim_3d: 64
  message_passing_steps_3d: 3
  num_rbf: 16
  cutoff: 5.0
  epochs: 100
  batch_size: 16
  learning_rate: 0.005
  supervised_weight: 1.0
  contrastive_weight: 1.0
  temperature: 0.1

arms:
  baseline: { teacher_weight: 0.0, conformer_pool_mode: mean }
  quantum:  { teacher_weight: 1.0, conformer_pool_mode: energy, energy_temperature: 298.15 }

seeds: [0, 1, 2]

holdout:
  fraction: 0.15
  seed: 1234

probes:
  - method: mlp_head            # frozen backbone — decisive
  - method: finetune            # headline

adapt:
  dataset: { csv: data/delaney-processed.csv, smiles_col: auto, targets: auto }
  task: regression
  adapter: { hidden_dims: [128, 64], dropout: 0.1 }
  training: { epochs: 300, lr: 1.0e-3, batch_size: 128, patience: 40, seed: 42 }
  split: { test_frac: 0.2, val_frac: 0.25, seed: 42, stratify: true }

outputs:
  dir: runs/validate
  report: runs/validate/report
```

- [ ] **Step 6: Smoke-test the CLI entry point on the synthetic path**

The real run is `python -m qchem_gnn.validation --config configs/validate_quantum_teacher.yaml`
(requires `zinc-250k/results/results_044.h5`; it trains 6 backbones + ~12 adapt runs and writes
`runs/validate/report.{json,md}`). If the shard is absent, this step is satisfied by the passing
end-to-end test in Step 4, which exercises the same `run_validation` path on synthetic data.

Run: `python -c "import qchem_gnn.validation as v; print(v.main.__doc__ or 'main present'); print(hasattr(v, 'run_validation'))"`
Expected: prints `True` (module imports cleanly and exposes the entry points).

- [ ] **Step 7: Run the full suite for no regressions**

Run: `python -m pytest tests -q`
Expected: PASS (previous ~140 + the new validation tests from Tasks 1-6)

- [ ] **Step 8: Commit**

```bash
git add qchem_gnn/validation.py configs/validate_quantum_teacher.yaml tests/test_validation_end_to_end.py
git commit -m "feat(validation): full driver, config parsing, CLI entry, and experiment config"
```

---

## Notes for the implementer

- The harness only orchestrates; it adds no training or model logic. If you find yourself editing the contrastive trainer or a model beyond Task 1's two result fields, stop and report it.
- `run_validation(cfg, dataset=...)` accepts an in-memory dataset specifically so tests avoid the 2.4 GB shard. The real run omits `dataset` and loads from `zinc-250k`.
- The adapt `training` block uses `lr` (matching the shipped ESOL configs), not `head_lr`. Pass the `adapt` block through verbatim.
- Backbone checkpoints intentionally save only `model_config` + `model_state_dict` — the minimum `load_backbone` reads. Do not serialize the teacher or encoder3d; the intrinsic metrics are cached as JSON instead.
- The verdict is a stated heuristic for small n, not a significance test. Keep the disclaimer line in the rendered report.
