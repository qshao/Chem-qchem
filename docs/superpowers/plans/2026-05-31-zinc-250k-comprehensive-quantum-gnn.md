# ZINC-250K Comprehensive Quantum GNN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a comprehensive 2D molecular graph neural network for ZINC-250K that learns from conformer-resolved quantum descriptors, exports reusable molecular embeddings, and transfers to downstream property prediction.

**Architecture:** Keep the current fast path as the baseline, then extend it in layers: shard-aware data loading, conformer-aware batching, a stronger multi-head encoder, normalized multi-task pretraining, checkpointed training, and downstream evaluation. The inference model stays 2D-only; conformer information is used only during training and target aggregation.

**Tech Stack:** Python 3.13, RDKit, PyTorch, pandas, numpy, scikit-learn, pyyaml, tqdm, optional `h5py`, optional `torch_geometric`.

---

### Task 1: Make The Data Layer The Single Source Of Truth

**Files:**
- Create: `qchem_gnn/dataset_index.py`
- Modify: `qchem_gnn/quantum_data.py`
- Modify: `qchem_gnn/minimal.py`
- Modify: `qchem_gnn/cli.py`
- Test: `tests/test_quantum_dataset.py`
- Test: `tests/test_minimal_dataset.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing tests**

Add tests that pin down the loader behavior for both fast and real-target modes:

```python
def test_load_quantum_zinc_dataset_uses_proxy_targets_when_results_are_disabled(tmp_path):
    ...

def test_load_quantum_zinc_dataset_aggregates_fake_hdf5_targets(tmp_path):
    ...

def test_train_cli_accepts_use_results_and_dataset_root(tmp_path):
    ...
```

The new `dataset_index.py` tests should verify a shard index API that can enumerate:

```python
DatasetIndex(
    csv_path=Path(...),
    geometry_path=Path(...),
    results_path=Path(...),
    mol_ids=[...],
)
```

**Step 2: Run the tests to verify they fail for the right reason**

Run:

```bash
python -m pytest tests/test_quantum_dataset.py tests/test_minimal_dataset.py tests/test_cli.py -q
```

Expected: failures or import errors for missing `dataset_index.py` helpers or missing real-loader wiring.

**Step 3: Implement the loader and index**

Add `qchem_gnn/dataset_index.py` with shard discovery and lazy per-shard metadata:

```python
@dataclass(frozen=True)
class DatasetIndex:
    subset_id: int
    csv_path: Path
    geometry_path: Path
    results_path: Path | None
    mol_ids: list[str]
```

Extend `qchem_gnn/quantum_data.py` so it:

- loads a single shard or a small shard range
- uses `--use-results` to opt into HDF5-backed targets
- falls back to proxy targets when results are missing or `h5py` is unavailable
- computes target normalization from the loaded dataset

**Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_quantum_dataset.py tests/test_minimal_dataset.py tests/test_cli.py -q
```

Expected: all tests pass, including both proxy fallback and opt-in real-target behavior.

**Step 5: Commit**

```bash
git add qchem_gnn/dataset_index.py qchem_gnn/quantum_data.py qchem_gnn/minimal.py qchem_gnn/cli.py tests/test_quantum_dataset.py tests/test_minimal_dataset.py tests/test_cli.py
git commit -m "feat: unify zinc-250k dataset loading"
```

---

### Task 2: Add Conformer-Aware Batching And Pooling

**Files:**
- Create: `qchem_gnn/conformer.py`
- Modify: `qchem_gnn/graph.py`
- Modify: `qchem_gnn/model.py`
- Test: `tests/test_conformer.py`
- Test: `tests/test_model.py`

**Step 1: Write the failing tests**

Add tests for a conformer batch container and conformer pooling:

```python
def test_conformer_batch_preserves_graph_and_conformer_boundaries():
    ...

def test_pool_conformer_embeddings_is_permutation_invariant():
    ...
```

The target API should look like:

```python
@dataclass(frozen=True)
class ConformerBatch:
    graph_batch: GraphBatch
    conformer_index: torch.LongTensor
    conformer_energy: torch.Tensor
```

**Step 2: Run the tests to verify they fail for the right reason**

Run:

```bash
python -m pytest tests/test_conformer.py tests/test_model.py -q
```

Expected: failures due to missing conformer batching helpers or missing pooling functions.

**Step 3: Implement conformer-aware pooling**

Add `qchem_gnn/conformer.py` with:

```python
def pool_conformer_embeddings(
    conformer_embeddings: torch.Tensor,
    conformer_energy: torch.Tensor | None = None,
    mode: str = "mean",
) -> torch.Tensor:
    ...
```

Extend `qchem_gnn/model.py` so the encoder can return:

```python
node_repr, edge_repr, graph_repr, mol_embedding = model(batch)
```

Use permutation-invariant pooling first, then keep the API ready for energy-weighted pooling later.

**Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_conformer.py tests/test_model.py -q
```

Expected: conformer batching and pooling tests pass.

**Step 5: Commit**

```bash
git add qchem_gnn/conformer.py qchem_gnn/graph.py qchem_gnn/model.py tests/test_conformer.py tests/test_model.py
git commit -m "feat: add conformer-aware batching"
```

---

### Task 3: Upgrade The Model To A Comprehensive Multi-Head Encoder

**Files:**
- Modify: `qchem_gnn/model.py`
- Modify: `qchem_gnn/training.py`
- Create: `qchem_gnn/losses.py`
- Test: `tests/test_model.py`
- Test: `tests/test_training.py`

**Step 1: Write the failing tests**

Add tests that require the model to expose a reusable molecular embedding and all quantum heads:

```python
def test_molecular_quantum_gnn_returns_embedding_and_quantum_heads():
    ...

def test_multitask_loss_supports_weighted_quantum_and_auxiliary_targets():
    ...
```

The model API should support:

```python
node_pred, edge_pred, graph_pred, mol_embedding = model(batch)
```

The loss API should support weighted task groups:

```python
compute_multitask_loss(
    predictions,
    targets,
    weights={
        "atom": 1.0,
        "edge": 1.0,
        "graph": 0.5,
        "aux": 0.1,
        "consistency": 0.1,
    },
)
```

**Step 2: Run the tests to verify they fail for the right reason**

Run:

```bash
python -m pytest tests/test_model.py tests/test_training.py -q
```

Expected: failures due to missing embedding output or missing weighted loss support.

**Step 3: Implement the comprehensive encoder**

In `qchem_gnn/model.py`, refactor the current baseline into:

- atom and bond embeddings
- residual message passing
- graph pooling
- a dedicated molecular embedding head
- separate heads for atom, edge, and graph targets

Keep the inference path 2D-only.

**Step 4: Implement the weighted multitask loss**

Create `qchem_gnn/losses.py` and move the task-specific loss composition there so training can combine:

- atom-level quantum loss
- edge-level quantum loss
- graph-level quantum loss
- auxiliary property loss
- conformer consistency loss

**Step 5: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_model.py tests/test_training.py -q
```

Expected: the model returns a reusable embedding and the weighted loss behaves as expected.

**Step 6: Commit**

```bash
git add qchem_gnn/model.py qchem_gnn/training.py qchem_gnn/losses.py tests/test_model.py tests/test_training.py
git commit -m "feat: upgrade quantum gnn heads and loss"
```

---

### Task 4: Add Checkpointed Training, Resume, And Embedding Export

**Files:**
- Create: `qchem_gnn/checkpoint.py`
- Modify: `qchem_gnn/cli.py`
- Modify: `qchem_gnn/minimal.py`
- Modify: `qchem_gnn/training.py`
- Test: `tests/test_checkpoint.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_training.py`

**Step 1: Write the failing tests**

Add tests that verify checkpoint round-tripping and exported embeddings:

```python
def test_save_and_load_checkpoint_round_trips_model_state(tmp_path):
    ...

def test_train_cli_writes_target_normalization_and_embeddings(tmp_path):
    ...
```

The checkpoint structure should include:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": ...,
    "global_step": ...,
    "target_normalization": ...,
    "dataset_config": ...,
    "split_metadata": ...,
}
```

**Step 2: Run the tests to verify they fail for the right reason**

Run:

```bash
python -m pytest tests/test_checkpoint.py tests/test_cli.py tests/test_training.py -q
```

Expected: failures due to missing checkpoint helpers or missing resume/export wiring.

**Step 3: Implement checkpoint and export helpers**

Create `qchem_gnn/checkpoint.py` with:

```python
def save_checkpoint(path: Path, state: dict) -> None:
    ...

def load_checkpoint(path: Path) -> dict:
    ...
```

Extend the CLI with:

- `train`
- `export-embeddings`
- `eval`

Make sure `train` can resume from an existing checkpoint and `export-embeddings` writes `.pt`, `.npz`, or `.csv` outputs.

**Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_checkpoint.py tests/test_cli.py tests/test_training.py -q
```

Expected: checkpoint round-trip passes and exported embeddings have the expected shapes and metadata.

**Step 5: Commit**

```bash
git add qchem_gnn/checkpoint.py qchem_gnn/cli.py qchem_gnn/minimal.py qchem_gnn/training.py tests/test_checkpoint.py tests/test_cli.py tests/test_training.py
git commit -m "feat: add checkpointed training and embedding export"
```

---

### Task 5: Add Downstream Evaluation, Split Logic, And Full-Scale Run Support

**Files:**
- Create: `qchem_gnn/eval.py`
- Create: `qchem_gnn/splits.py`
- Modify: `qchem_gnn/cli.py`
- Modify: `qchem_gnn/__init__.py`
- Test: `tests/test_eval.py`
- Test: `tests/test_splits.py`

**Step 1: Write the failing tests**

Add tests for split generation and downstream evaluation:

```python
def test_scaffold_or_random_split_is_molecule_level_only():
    ...

def test_linear_probe_runs_on_exported_embeddings():
    ...
```

Evaluation should cover:

- linear probe
- fine-tuning
- Morgan fingerprint baseline
- MAE, RMSE, R2
- sample efficiency at 1%, 5%, 10%, 100%

**Step 2: Run the tests to verify they fail for the right reason**

Run:

```bash
python -m pytest tests/test_eval.py tests/test_splits.py -q
```

Expected: failures due to missing split utilities or evaluation helpers.

**Step 3: Implement split and evaluation modules**

Create `qchem_gnn/splits.py` with molecule-level split utilities that never separate conformers from their parent molecule.

Create `qchem_gnn/eval.py` with:

```python
def run_linear_probe(embeddings, labels, split) -> dict:
    ...

def run_fine_tuning(dataset, checkpoint_path) -> dict:
    ...
```

Wire these into the CLI so full-run experiments are reproducible.

**Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_eval.py tests/test_splits.py -q
```

Expected: downstream evaluation and split logic pass, with molecule-level integrity preserved.

**Step 5: Commit**

```bash
git add qchem_gnn/eval.py qchem_gnn/splits.py qchem_gnn/cli.py qchem_gnn/__init__.py tests/test_eval.py tests/test_splits.py
git commit -m "feat: add downstream evaluation and split utilities"
```

---

## Success Criteria

The comprehensive model work is complete when:

- the loader can run in both proxy and HDF5-backed modes
- conformer-aware batching is permutation-invariant
- the model emits a reusable molecular embedding
- checkpoints can resume training and preserve normalization
- embeddings can be exported for downstream tasks
- downstream evaluation shows whether quantum pretraining improves performance over baselines
