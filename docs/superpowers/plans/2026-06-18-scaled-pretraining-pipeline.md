# Scaled Pretraining Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preprocess the full ZINC dataset into a compact on-disk cache (drop the density matrix, 1.7 TB → ~22 GB), preload it to train the existing contrastive backbone at full scale, and measure downstream MAE vs. data scale.

**Architecture:** A one-time `preprocess` step persists the already-extracted compact examples (the loader never reads `dm`) plus a globally stable `scaffold_key` per molecule. A preload loader concatenates selected compact shards into one in-memory dataset fed to the unchanged trainer. The infeasible global `[N,N]` scaffold mask is replaced by a per-batch mask built from `scaffold_key`, and the holdout becomes a deterministic scaffold-disjoint hash split.

**Tech Stack:** Python 3.13, PyTorch, RDKit (Murcko scaffolds), pandas, h5py, pytest.

## Global Constraints

- `scaffold_key` is a globally stable integer: `int.from_bytes(hashlib.blake2b(scaffold_smiles.encode("utf-8"), digest_size=8).digest(), "big")`. Never use Python's salted `hash()` and never use a per-shard appearance index.
- The Murcko scaffold SMILES is computed exactly as the existing `_infer_scaffolds` does: `MurckoScaffold.MurckoScaffoldSmiles(mol=mol)`, falling back to the raw SMILES when `Chem.MolFromSmiles` returns `None` or the scaffold is empty.
- Compact cache payload format: `{"version": 1, "examples": list[MinimalQuantumExample], "skipped_mol_ids": tuple[str, ...]}`, saved via `torch.save`. The constant `SHARD_CACHE_VERSION = 1` lives in `qchem_gnn/shard_cache.py`.
- Cache file naming: `shard_{subset_id:03d}.pt` (zero-padded to 3 digits), matching the existing `coords_{id:03d}.pkl` / `results_{id:03d}.h5` convention.
- The backbone, projection heads, teacher heads, InfoNCE/`losses.py`, Boltzmann pooling, VICReg arm, adapt subsystem, and checkpoint/inference contract are unchanged. Only data feeding and scaffold-mask construction change.
- `use_scaffold_negmask=False` (the default) must remain byte-for-byte identical to today's behaviour.
- TDD throughout: write the failing test, see it fail, implement, see it pass, commit.

---

### Task 1: Stable scaffold key + per-batch scaffold mask (pure functions)

**Files:**
- Modify: `qchem_gnn/eval.py` (add two functions after `_infer_scaffolds`, which ends at line 172)
- Test: `tests/test_scaffold_mask.py` (append new tests)

**Interfaces:**
- Consumes: `Chem`, `MurckoScaffold` (already imported in `eval.py`), `torch` (already imported).
- Produces:
  - `scaffold_key_from_smiles(smiles: str) -> int`
  - `scaffold_mask_from_keys(keys: Sequence[int]) -> torch.Tensor` returning a `[B, B]` CPU bool tensor, `True` at `[i, j]` iff `i != j` and `keys[i] == keys[j]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scaffold_mask.py`:

```python
def test_scaffold_key_same_for_shared_murcko_scaffold():
    from qchem_gnn.eval import scaffold_key_from_smiles
    # toluene and aniline both reduce to benzene under Murcko
    assert scaffold_key_from_smiles("Cc1ccccc1") == scaffold_key_from_smiles("Nc1ccccc1")


def test_scaffold_key_differs_for_distinct_scaffold():
    from qchem_gnn.eval import scaffold_key_from_smiles
    assert scaffold_key_from_smiles("Cc1ccccc1") != scaffold_key_from_smiles("CCO")


def test_scaffold_key_deterministic_across_calls():
    from qchem_gnn.eval import scaffold_key_from_smiles
    assert scaffold_key_from_smiles("Nc1ccccc1") == scaffold_key_from_smiles("Nc1ccccc1")


def test_scaffold_key_unparseable_falls_back_to_smiles():
    from qchem_gnn.eval import scaffold_key_from_smiles
    # an unparseable string still yields a stable, self-consistent key
    assert scaffold_key_from_smiles("not_a_smiles") == scaffold_key_from_smiles("not_a_smiles")


def test_scaffold_mask_from_keys_groups_equal_keys():
    import torch
    from qchem_gnn.eval import scaffold_mask_from_keys
    mask = scaffold_mask_from_keys([10, 10, 20])
    assert mask.dtype == torch.bool
    assert mask.shape == (3, 3)
    assert mask[0, 1] and mask[1, 0]
    assert not mask[0, 2] and not mask[1, 2]
    assert not mask[0, 0] and not mask[1, 1] and not mask[2, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scaffold_mask.py -k "scaffold_key or from_keys" -v`
Expected: FAIL with `ImportError: cannot import name 'scaffold_key_from_smiles'`.

- [ ] **Step 3: Implement the functions**

In `qchem_gnn/eval.py`, add `import hashlib` to the imports at the top of the file (alongside the existing stdlib imports), and add `from typing import Sequence` if `Sequence` is not already imported (the file already imports `Iterable` from `typing` — extend that import to `from typing import Iterable, Sequence`).

Insert these two functions immediately after `_infer_scaffolds` (after line 172, before `build_scaffold_negative_mask`):

```python
def scaffold_key_from_smiles(smiles: str) -> int:
    """Globally stable integer key for a molecule's Murcko scaffold.

    Two molecules sharing a Murcko scaffold get the same key, deterministically
    across processes and shards (unlike Python's per-process-salted ``hash()``).
    Unparseable SMILES fall back to the raw SMILES, so they only collide with an
    identical SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        scaffold = smiles
    else:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or smiles
    digest = hashlib.blake2b(scaffold.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def scaffold_mask_from_keys(keys: Sequence[int]) -> torch.Tensor:
    """Return [B, B] CPU bool mask; True at [i,j] iff i!=j and keys[i]==keys[j]."""
    t = torch.tensor(list(keys), dtype=torch.int64)
    eq = t[:, None] == t[None, :]
    eq.fill_diagonal_(False)
    return eq
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scaffold_mask.py -v`
Expected: PASS (new tests plus the existing ones).

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/eval.py tests/test_scaffold_mask.py
git commit -m "feat(scale): stable scaffold_key and per-batch scaffold_mask_from_keys"
```

---

### Task 2: Trainer builds per-batch mask; remove global mask

**Files:**
- Modify: `qchem_gnn/contrastive_pretrain.py` (import line 18; `full_mask` block lines 137-140; batch-mask block ~lines 166-173)
- Modify: `qchem_gnn/eval.py` (remove `build_scaffold_negative_mask`, lines 175-193)
- Test: `tests/test_scaffold_mask.py` (remove obsolete global-mask tests), `tests/test_contrastive_pretrain.py` (add per-batch equivalence test)

**Interfaces:**
- Consumes: `scaffold_key_from_smiles`, `scaffold_mask_from_keys` (Task 1).
- Produces: `contrastive_pretrain_on_dataset(..., use_scaffold_negmask: bool = False)` with identical signature, now building the mask per batch from scaffold keys; `build_scaffold_negative_mask` no longer exists.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_contrastive_pretrain.py` (the file already imports `math`, `contrastive_pretrain_on_dataset`, and `make_tiny_quantum_dataset`):

```python
def test_per_batch_scaffold_mask_matches_key_equality():
    # The per-batch mask must equal scaffold-key equality for the batch's molecules.
    import torch
    from qchem_gnn.eval import scaffold_key_from_smiles, scaffold_mask_from_keys

    smiles = ["Cc1ccccc1", "Nc1ccccc1", "CCO"]  # benzene, benzene, ethanol
    keys = [scaffold_key_from_smiles(s) for s in smiles]
    mask = scaffold_mask_from_keys(keys)
    expected = torch.tensor(
        [[False, True, False],
         [True, False, False],
         [False, False, False]]
    )
    assert torch.equal(mask, expected)


def test_build_scaffold_negative_mask_removed():
    import qchem_gnn.eval as ev
    assert not hasattr(ev, "build_scaffold_negative_mask")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_contrastive_pretrain.py -k "per_batch_scaffold_mask or build_scaffold_negative_mask_removed" -v`
Expected: FAIL — `test_build_scaffold_negative_mask_removed` fails because the function still exists.

- [ ] **Step 3: Remove the global mask from `eval.py`**

Delete `build_scaffold_negative_mask` from `qchem_gnn/eval.py` (lines 175-193, the function with docstring `"""Return [N, N] CPU bool tensor; ..."""`).

- [ ] **Step 4: Remove obsolete global-mask tests**

In `tests/test_scaffold_mask.py`, delete every test that calls `build_scaffold_negative_mask` (the original tests from the scaffold-masking feature). Keep the `scaffold_key_*` and `scaffold_mask_from_keys` tests added in Task 1. If this leaves an unused `from qchem_gnn.eval import build_scaffold_negative_mask` import, remove it.

- [ ] **Step 5: Rewire the trainer**

In `qchem_gnn/contrastive_pretrain.py`:

Replace the import on line 18:
```python
from .eval import build_scaffold_negative_mask
```
with:
```python
from .eval import scaffold_key_from_smiles, scaffold_mask_from_keys
```

Add this module-level helper just below the imports (after the existing `from .eval import ...` line):
```python
def _example_scaffold_key(example) -> int:
    key = getattr(example, "scaffold_key", None)
    if key is not None:
        return key
    return scaffold_key_from_smiles(example.smiles)
```

Delete the pre-loop `full_mask` block (lines 137-140):
```python
    full_mask: torch.Tensor | None = None
    if use_scaffold_negmask:
        full_mask = build_scaffold_negative_mask(examples)
```

Inside the batch loop, replace the existing `batch_mask` block (the lines that read `batch_mask: torch.Tensor | None = None`, `if full_mask is not None:`, `global_idx = ...`, `batch_mask = full_mask[global_idx][:, global_idx].to(...)`, and the `warnings.warn` guard) with:
```python
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
```
(`usable_examples` is already defined on the preceding line as `usable_examples = [ex for _, ex in usable]`. Keep the existing `import warnings` at the top of the file.)

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_contrastive_pretrain.py tests/test_scaffold_mask.py -v`
Expected: PASS, including the existing `use_scaffold_negmask=True` end-to-end test and the `False`-default regression.

- [ ] **Step 7: Commit**

```bash
git add qchem_gnn/contrastive_pretrain.py qchem_gnn/eval.py tests/test_scaffold_mask.py tests/test_contrastive_pretrain.py
git commit -m "feat(scale): per-batch scaffold mask in trainer, drop global NxN mask"
```

---

### Task 3: `scaffold_key` field + compact shard cache module

**Files:**
- Modify: `qchem_gnn/minimal.py` (add field to `MinimalQuantumExample`, lines 14-29)
- Create: `qchem_gnn/shard_cache.py`
- Test: `tests/test_shard_cache.py` (new file)

**Interfaces:**
- Consumes: `load_quantum_zinc_subset_range` (`qchem_gnn/quantum_data.py`), `scaffold_key_from_smiles` (Task 1), `MinimalQuantumDataset`/`MinimalQuantumExample` (`qchem_gnn/minimal.py`).
- Produces:
  - `MinimalQuantumExample.scaffold_key: int | None = None`
  - `SHARD_CACHE_VERSION = 1`
  - `preprocess_shard(dataset_root, subset_id, cache_dir, *, overwrite=False) -> Path`
  - `load_compact_shard(path) -> tuple[list[MinimalQuantumExample], tuple[str, ...]]`
  - `load_compact_shards(cache_dir, shard_ids) -> MinimalQuantumDataset`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_shard_cache.py`:

```python
from pathlib import Path

import torch

from qchem_gnn.shard_cache import (
    SHARD_CACHE_VERSION,
    load_compact_shard,
    load_compact_shards,
    preprocess_shard,
)
from qchem_gnn.minimal import MinimalQuantumDataset, MinimalQuantumExample


def _fake_example(mol_id, smiles):
    g = type("G", (), {})()  # placeholder; not inspected by cache I/O
    return MinimalQuantumExample(
        mol_id=mol_id,
        smiles=smiles,
        graph=g,
        charge=0,
        conformer_count=0,
        node_target=torch.zeros(1, 1),
        edge_target=torch.zeros(0, 1),
        graph_target=torch.zeros(2),
    )


def test_example_has_scaffold_key_field_default_none():
    ex = _fake_example("m0", "CCO")
    assert ex.scaffold_key is None


def test_load_compact_shard_roundtrip(tmp_path):
    examples = [_fake_example("m0", "CCO"), _fake_example("m1", "Cc1ccccc1")]
    path = tmp_path / "shard_007.pt"
    torch.save(
        {"version": SHARD_CACHE_VERSION, "examples": examples, "skipped_mol_ids": ("m9",)},
        path,
    )
    loaded, skipped = load_compact_shard(path)
    assert [e.mol_id for e in loaded] == ["m0", "m1"]
    assert skipped == ("m9",)


def test_load_compact_shard_rejects_bad_version(tmp_path):
    path = tmp_path / "shard_000.pt"
    torch.save({"version": 999, "examples": [], "skipped_mol_ids": ()}, path)
    try:
        load_compact_shard(path)
    except ValueError as exc:
        assert "version" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for version mismatch")


def test_load_compact_shards_concatenates(tmp_path):
    for sid, smis in ((0, ["CCO"]), (3, ["CCN", "CCC"])):
        examples = [_fake_example(f"s{sid}_{i}", s) for i, s in enumerate(smis)]
        torch.save(
            {"version": SHARD_CACHE_VERSION, "examples": examples, "skipped_mol_ids": ()},
            tmp_path / f"shard_{sid:03d}.pt",
        )
    ds = load_compact_shards(tmp_path, [0, 3])
    assert isinstance(ds, MinimalQuantumDataset)
    assert len(ds) == 3
    assert [e.mol_id for e in ds.examples] == ["s0_0", "s3_0", "s3_1"]


def test_preprocess_shard_writes_compact_cache_with_scaffold_keys(tmp_path, monkeypatch):
    # Stub the heavy loader so the test stays fast and offline.
    import qchem_gnn.shard_cache as sc

    fake = MinimalQuantumDataset(
        examples=[_fake_example("m0", "Cc1ccccc1"), _fake_example("m1", "Nc1ccccc1")],
        skipped_mol_ids=("m2",),
    )
    monkeypatch.setattr(sc, "load_quantum_zinc_subset_range", lambda *a, **k: fake)

    out = preprocess_shard(tmp_path / "root", 5, tmp_path / "cache")
    assert out == tmp_path / "cache" / "shard_005.pt"

    examples, skipped = load_compact_shard(out)
    assert skipped == ("m2",)
    # toluene + aniline share the benzene Murcko scaffold -> equal keys
    assert examples[0].scaffold_key == examples[1].scaffold_key
    assert examples[0].scaffold_key is not None


def test_preprocess_shard_skips_existing(tmp_path, monkeypatch):
    import qchem_gnn.shard_cache as sc

    calls = {"n": 0}

    def _counting_loader(*a, **k):
        calls["n"] += 1
        return MinimalQuantumDataset(examples=[_fake_example("m0", "CCO")], skipped_mol_ids=())

    monkeypatch.setattr(sc, "load_quantum_zinc_subset_range", _counting_loader)

    preprocess_shard(tmp_path / "root", 1, tmp_path / "cache")
    preprocess_shard(tmp_path / "root", 1, tmp_path / "cache")  # should skip
    assert calls["n"] == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_shard_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qchem_gnn.shard_cache'`.

- [ ] **Step 3: Add the `scaffold_key` field**

In `qchem_gnn/minimal.py`, add a field to `MinimalQuantumExample` as the last field (after `conformer_graph_targets` on line 29):

```python
    conformer_graph_targets: torch.Tensor | None = None
    scaffold_key: int | None = None
```

- [ ] **Step 4: Create the cache module**

Create `qchem_gnn/shard_cache.py`:

```python
from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from .eval import scaffold_key_from_smiles
from .minimal import MinimalQuantumDataset
from .quantum_data import load_quantum_zinc_subset_range

SHARD_CACHE_VERSION = 1


def _cache_path(cache_dir: Path, subset_id: int) -> Path:
    return Path(cache_dir) / f"shard_{subset_id:03d}.pt"


def _is_valid_cache(path: Path) -> bool:
    try:
        payload = torch.load(path, weights_only=False)
    except Exception:
        return False
    return (
        isinstance(payload, dict)
        and payload.get("version") == SHARD_CACHE_VERSION
        and "examples" in payload
        and "skipped_mol_ids" in payload
    )


def preprocess_shard(
    dataset_root, subset_id: int, cache_dir, *, overwrite: bool = False
) -> Path:
    """Extract one shard into a compact, scaffold-keyed cache file.

    Loads the shard through the existing extractor (which already ignores the
    density matrix), attaches a globally stable ``scaffold_key`` to each example,
    and serializes a versioned payload. Skips work if a valid cache exists.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = _cache_path(cache_dir, subset_id)
    if out.exists() and not overwrite and _is_valid_cache(out):
        return out

    dataset = load_quantum_zinc_subset_range(
        dataset_root,
        subset_ids=[subset_id],
        limit_per_shard=None,
        use_results=True,
    )
    examples = [
        dataclasses.replace(ex, scaffold_key=scaffold_key_from_smiles(ex.smiles))
        for ex in dataset.examples
    ]
    torch.save(
        {
            "version": SHARD_CACHE_VERSION,
            "examples": examples,
            "skipped_mol_ids": tuple(dataset.skipped_mol_ids),
        },
        out,
    )
    return out


def load_compact_shard(path):
    """Load one compact shard cache, validating its format version."""
    payload = torch.load(Path(path), weights_only=False)
    if not isinstance(payload, dict) or payload.get("version") != SHARD_CACHE_VERSION:
        got = payload.get("version") if isinstance(payload, dict) else None
        raise ValueError(
            f"shard cache version mismatch at {path}: expected "
            f"{SHARD_CACHE_VERSION}, got {got}"
        )
    return payload["examples"], tuple(payload["skipped_mol_ids"])


def load_compact_shards(cache_dir, shard_ids) -> MinimalQuantumDataset:
    """Preload a set of compact shard caches into one in-memory dataset."""
    examples = []
    skipped: list[str] = []
    for subset_id in shard_ids:
        exs, sk = load_compact_shard(_cache_path(Path(cache_dir), subset_id))
        examples.extend(exs)
        skipped.extend(sk)
    return MinimalQuantumDataset(examples=examples, skipped_mol_ids=tuple(skipped))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_shard_cache.py -v`
Expected: PASS (all 7 tests).

- [ ] **Step 6: Commit**

```bash
git add qchem_gnn/minimal.py qchem_gnn/shard_cache.py tests/test_shard_cache.py
git commit -m "feat(scale): scaffold_key field and compact shard cache module"
```

---

### Task 4: Scaffold-disjoint hash holdout

**Files:**
- Modify: `qchem_gnn/validation.py` (add `scaffold_hash_holdout` near `split_holdout` at lines 25-40; branch in `run_validation` at lines 371-374)
- Test: `tests/test_validation_holdout.py` (new file)

**Interfaces:**
- Consumes: `scaffold_key_from_smiles` (Task 1), `MinimalQuantumDataset`, `MinimalQuantumExample`.
- Produces: `scaffold_hash_holdout(dataset, k: int) -> tuple[MinimalQuantumDataset, MinimalQuantumDataset]`. In `run_validation`, when `cfg["holdout"]` contains `"k"`, the holdout uses `scaffold_hash_holdout`; otherwise the existing `split_holdout` path is used unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validation_holdout.py`:

```python
import torch

from qchem_gnn.minimal import MinimalQuantumDataset, MinimalQuantumExample
from qchem_gnn.validation import scaffold_hash_holdout


def _ex(mol_id, smiles, key):
    return MinimalQuantumExample(
        mol_id=mol_id,
        smiles=smiles,
        graph=type("G", (), {})(),
        charge=0,
        conformer_count=0,
        node_target=torch.zeros(1, 1),
        edge_target=torch.zeros(0, 1),
        graph_target=torch.zeros(2),
        scaffold_key=key,
    )


def test_scaffold_hash_holdout_is_deterministic_and_disjoint():
    # keys chosen so k=3 sends key%3==0 to holdout
    examples = [_ex(f"m{i}", "CCO", key=i) for i in range(6)]
    ds = MinimalQuantumDataset(examples=examples)
    pre1, hold1 = scaffold_hash_holdout(ds, k=3)
    pre2, hold2 = scaffold_hash_holdout(ds, k=3)
    assert [e.mol_id for e in hold1.examples] == [e.mol_id for e in hold2.examples]
    holdout_keys = {e.scaffold_key for e in hold1.examples}
    pretrain_keys = {e.scaffold_key for e in pre1.examples}
    assert holdout_keys.isdisjoint(pretrain_keys)  # scaffold-disjoint
    assert holdout_keys == {0, 3}


def test_scaffold_hash_holdout_falls_back_to_smiles_when_key_missing():
    # scaffold_key None -> computed from smiles; identical scaffolds share a bucket
    a = _ex("a", "Cc1ccccc1", key=None)
    b = _ex("b", "Nc1ccccc1", key=None)  # same benzene scaffold as a
    ds = MinimalQuantumDataset(examples=[a, b])
    pre, hold = scaffold_hash_holdout(ds, k=1)  # k=1 -> everything holdout
    assert len(hold.examples) == 2 and len(pre.examples) == 0 or (
        len(pre.examples) == 2 and len(hold.examples) == 0
    )
    # a and b must land in the SAME split (same scaffold)
    split_of = {e.mol_id: "hold" for e in hold.examples}
    split_of.update({e.mol_id: "pre" for e in pre.examples})
    assert split_of["a"] == split_of["b"]


def test_scaffold_hash_holdout_raises_when_a_side_is_empty():
    examples = [_ex(f"m{i}", "CCO", key=2) for i in range(3)]  # all key%3 != 0
    ds = MinimalQuantumDataset(examples=examples)
    try:
        scaffold_hash_holdout(ds, k=3)
    except ValueError as exc:
        assert "holdout" in str(exc).lower() or "empty" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError when holdout side is empty")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_validation_holdout.py -v`
Expected: FAIL with `ImportError: cannot import name 'scaffold_hash_holdout'`.

- [ ] **Step 3: Implement `scaffold_hash_holdout`**

In `qchem_gnn/validation.py`, add `from .eval import scaffold_key_from_smiles` to the imports, and insert this function immediately after `split_holdout` (after line 40):

```python
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
```

- [ ] **Step 4: Wire it into `run_validation`**

In `qchem_gnn/validation.py`, replace the holdout block in `run_validation` (lines 371-374):

```python
    holdout_cfg = cfg["holdout"]
    pretrain_ds, holdout = split_holdout(
        dataset, fraction=holdout_cfg["fraction"], seed=holdout_cfg["seed"]
    )
```

with:

```python
    holdout_cfg = cfg["holdout"]
    if "k" in holdout_cfg:
        pretrain_ds, holdout = scaffold_hash_holdout(dataset, k=holdout_cfg["k"])
    else:
        pretrain_ds, holdout = split_holdout(
            dataset, fraction=holdout_cfg["fraction"], seed=holdout_cfg["seed"]
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_validation_holdout.py tests/test_validation_end_to_end.py -v`
Expected: PASS (new holdout tests; existing end-to-end test still uses the `fraction` path).

- [ ] **Step 6: Commit**

```bash
git add qchem_gnn/validation.py tests/test_validation_holdout.py
git commit -m "feat(scale): scaffold-disjoint hash holdout wired into run_validation"
```

---

### Task 5: `preprocess` CLI command

**Files:**
- Modify: `qchem_gnn/cli.py` (add subparser in `build_parser`; add `run_preprocess`; dispatch in `main` at lines 811-831)
- Test: `tests/test_cli_preprocess.py` (new file)

**Interfaces:**
- Consumes: `preprocess_shard` (Task 3).
- Produces: CLI `qchem_gnn preprocess --dataset-root R --subset-ids "0,1,2" --cache-dir C [--overwrite]`; `run_preprocess(args) -> int` returning `0` on success and writing one `shard_{id:03d}.pt` per requested id.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_preprocess.py`:

```python
import qchem_gnn.cli as cli


def test_preprocess_command_writes_one_cache_per_shard(tmp_path, monkeypatch):
    written = []

    def _fake_preprocess_shard(dataset_root, subset_id, cache_dir, *, overwrite=False):
        path = tmp_path / "cache" / f"shard_{subset_id:03d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        written.append(subset_id)
        return path

    monkeypatch.setattr(cli, "preprocess_shard", _fake_preprocess_shard)

    rc = cli.main([
        "preprocess",
        "--dataset-root", str(tmp_path / "root"),
        "--subset-ids", "0,2,5",
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert rc == 0
    assert written == [0, 2, 5]
    assert (tmp_path / "cache" / "shard_000.pt").exists()
    assert (tmp_path / "cache" / "shard_005.pt").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_cli_preprocess.py -v`
Expected: FAIL — `argparse` errors on the unknown `preprocess` command (`SystemExit`), or `AttributeError: module 'qchem_gnn.cli' has no attribute 'preprocess_shard'`.

- [ ] **Step 3: Add the import and parser**

In `qchem_gnn/cli.py`, add to the imports:
```python
from .shard_cache import preprocess_shard
```

In `build_parser`, after the `adapt_cmd` block (before `return parser` on line 141), add:
```python
    preprocess = subparsers.add_parser(
        "preprocess", help="Extract shards into compact scaffold-keyed caches"
    )
    preprocess.add_argument("--dataset-root", required=True, help="Dataset root with subsets/, geometries/, results/")
    preprocess.add_argument("--subset-ids", required=True, help="Comma-separated subset ids, e.g. 0,1,2")
    preprocess.add_argument("--cache-dir", required=True, help="Output directory for compact shard caches")
    preprocess.add_argument("--overwrite", action="store_true", default=False, help="Re-extract even if a valid cache exists")
```

- [ ] **Step 4: Add `run_preprocess` and dispatch**

Add the handler (next to the other `run_*` functions):
```python
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
```

In `main` (lines 811-831), add a dispatch branch alongside the others (e.g., after the `contrastive-pretrain` branch):
```python
    if args.command == "preprocess":
        return run_preprocess(args)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_cli_preprocess.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add qchem_gnn/cli.py tests/test_cli_preprocess.py
git commit -m "feat(scale): preprocess CLI command for compact shard caches"
```

---

### Task 6: Validation loads from compact cache + scaling config

**Files:**
- Modify: `qchem_gnn/validation.py` (`_load_dataset`, lines 355-364)
- Create: `configs/validate_scaled.yaml`
- Test: `tests/test_validation_cache_loading.py` (new file)

**Interfaces:**
- Consumes: `load_compact_shards` (Task 3), `scaffold_hash_holdout` (Task 4).
- Produces: `_load_dataset(pretrain_cfg)` returns a dataset from compact caches when `pretrain_cfg` contains `cache_dir`, else falls back to the existing raw `load_quantum_zinc_subset_range` path.

- [ ] **Step 1: Write the failing test**

Create `tests/test_validation_cache_loading.py`:

```python
import torch

from qchem_gnn.shard_cache import SHARD_CACHE_VERSION
from qchem_gnn.validation import _load_dataset
from qchem_gnn.minimal import MinimalQuantumExample


def _ex(mol_id, smiles):
    return MinimalQuantumExample(
        mol_id=mol_id,
        smiles=smiles,
        graph=type("G", (), {})(),
        charge=0,
        conformer_count=0,
        node_target=torch.zeros(1, 1),
        edge_target=torch.zeros(0, 1),
        graph_target=torch.zeros(2),
        scaffold_key=1,
    )


def test_load_dataset_reads_compact_cache_when_cache_dir_present(tmp_path):
    for sid, mols in ((0, ["CCO"]), (1, ["CCN", "CCC"])):
        torch.save(
            {
                "version": SHARD_CACHE_VERSION,
                "examples": [_ex(f"s{sid}_{i}", s) for i, s in enumerate(mols)],
                "skipped_mol_ids": (),
            },
            tmp_path / f"shard_{sid:03d}.pt",
        )
    cfg = {"cache_dir": str(tmp_path), "subset_ids": [0, 1]}
    ds = _load_dataset(cfg)
    assert len(ds) == 3
    assert [e.mol_id for e in ds.examples] == ["s0_0", "s1_0", "s1_1"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_validation_cache_loading.py -v`
Expected: FAIL — `_load_dataset` calls `load_quantum_zinc_subset_range` and raises `KeyError: 'dataset_root'`.

- [ ] **Step 3: Implement the cache branch**

In `qchem_gnn/validation.py`, replace `_load_dataset` (lines 355-364):

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
```

with:

```python
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
```

- [ ] **Step 4: Create the scaling config**

Create `configs/validate_scaled.yaml`:

```yaml
# Full-scale validation: trains the proven backbone from compact shard caches.
# Build caches first:
#   python -m qchem_gnn preprocess --dataset-root zinc-250k \
#     --subset-ids 0,1,2,...,49 --cache-dir zinc-250k/compact_cache
# Then sweep scale with scripts/scaling_sweep.sh, or run a single point here.

pretrain:
  cache_dir: zinc-250k/compact_cache
  subset_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # one scale point; sweep overrides this
  hidden_dim: 64
  message_passing_steps: 3
  hidden_dim_3d: 64
  message_passing_steps_3d: 3
  epochs: 200
  learning_rate: 0.001
  batch_size: 16

arms:
  quantum_scaffold:
    teacher_weight: 1.0
    conformer_pool_mode: energy
    energy_temperature: 298.15
    use_scaffold_negmask: true

comparisons: []

seeds: [0, 1, 2]
holdout:
  k: 10                # scaffold-disjoint: ~1/10 of scaffolds held out

probes:
  - method: mlp_head
  - method: finetune

adapt:
  dataset: { csv: data/delaney-processed.csv, smiles_col: auto, targets: auto }
  task: regression
  adapter: { hidden_dims: [128, 64], dropout: 0.1 }
  training: { epochs: 300, lr: 1.0e-3, batch_size: 128, patience: 40, seed: 42 }
  split: { test_frac: 0.2, val_frac: 0.25, seed: 42, stratify: true }

outputs:
  dir: runs/validate_scaled
  report: runs/validate_scaled/report
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_validation_cache_loading.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add qchem_gnn/validation.py configs/validate_scaled.yaml tests/test_validation_cache_loading.py
git commit -m "feat(scale): validation loads compact caches; add scaled config"
```

---

### Task 7: Scaling sweep script + MAE-vs-scale aggregator

**Files:**
- Create: `scripts/scaling_sweep.sh`
- Create: `scripts/aggregate_scaling.py`
- Test: `tests/test_aggregate_scaling.py` (new file)

**Interfaces:**
- Consumes: validation `report.json` files. The structure (verified against a real run) is `{"aggregate": {"extrinsic": {method: {arm: {"mae_mean": float, "mae_std": float, "n": int, "r2_mean": float, "r2_std": float}}}}}` — `extrinsic` is a nested dict keyed first by method, then by arm.
- Produces: `aggregate_scaling(report_paths: dict[int, str]) -> list[dict]` rows `{"scale": int, "method": str, "arm": str, "mae_mean": float}`, and a `scripts/scaling_sweep.sh` that preprocesses, runs validation at several scales, and prints the table.

- [ ] **Step 1: Confirm the report shape**

The aggregator depends on the exact `extrinsic` nesting. Verify it:

Run: `python -c "import json; d=json.load(open('runs/validate/report.json')); ext=d['aggregate']['extrinsic']; m=next(iter(ext)); a=next(iter(ext[m])); print(m, a, sorted(ext[m][a]))"`
Expected: prints a method, an arm, and a field list including `mae_mean` — i.e., `extrinsic[method][arm]["mae_mean"]`. The test in Step 2 pins this nested contract.

- [ ] **Step 2: Write the failing test**

Create `tests/test_aggregate_scaling.py`:

```python
import json

from scripts.aggregate_scaling import aggregate_scaling


def _write_report(path, extrinsic):
    path.write_text(json.dumps({"aggregate": {"extrinsic": extrinsic}}))


def test_aggregate_scaling_collects_mae_by_scale(tmp_path):
    r10 = tmp_path / "s10.json"
    r50 = tmp_path / "s50.json"
    # extrinsic is nested: method -> arm -> stats
    _write_report(r10, {"mlp_head": {"quantum_scaffold": {"mae_mean": 0.90}}})
    _write_report(r50, {"mlp_head": {"quantum_scaffold": {"mae_mean": 0.80}}})

    rows = aggregate_scaling({10: str(r10), 50: str(r50)})
    by_scale = {r["scale"]: r["mae_mean"] for r in rows if r["method"] == "mlp_head"}
    assert by_scale == {10: 0.90, 50: 0.80}
    assert all({"scale", "method", "arm", "mae_mean"} <= set(r) for r in rows)
    assert rows[0]["arm"] == "quantum_scaffold"
```

- [ ] **Step 3: Implement the aggregator**

Create `scripts/aggregate_scaling.py`:

```python
"""Tabulate downstream MAE vs. pretraining data scale from validation reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def aggregate_scaling(report_paths: dict[int, str]) -> list[dict]:
    """Read per-scale validation reports into MAE-vs-scale rows.

    ``report_paths`` maps a scale (number of shards) to that run's report.json.
    ``extrinsic`` is nested as ``extrinsic[method][arm]["mae_mean"]``.
    """
    rows: list[dict] = []
    for scale in sorted(report_paths):
        payload = json.loads(Path(report_paths[scale]).read_text())
        extrinsic = payload["aggregate"]["extrinsic"]
        for method, arms in extrinsic.items():
            for arm, stats in arms.items():
                rows.append(
                    {
                        "scale": scale,
                        "method": method,
                        "arm": arm,
                        "mae_mean": float(stats["mae_mean"]),
                    }
                )
    return rows


def _print_table(rows: list[dict]) -> None:
    print(f"{'scale':>6} {'method':>10} {'arm':>18} {'mae_mean':>10}")
    for r in sorted(rows, key=lambda x: (x["method"], x["scale"])):
        print(f"{r['scale']:>6} {r['method']:>10} {r['arm']:>18} {r['mae_mean']:>10.4f}")


if __name__ == "__main__":
    # Usage: aggregate_scaling.py 10=runs/s10/report.json 50=runs/s50/report.json
    paths = {}
    for arg in sys.argv[1:]:
        scale_str, _, path = arg.partition("=")
        paths[int(scale_str)] = path
    _print_table(aggregate_scaling(paths))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_aggregate_scaling.py -v`
Expected: PASS.

- [ ] **Step 5: Create the sweep script**

Create `scripts/scaling_sweep.sh`:

```bash
#!/usr/bin/env bash
# Measure downstream MAE vs. pretraining data scale.
#
# Usage:
#   bash scripts/scaling_sweep.sh <dataset_root> <cache_dir> [scales]
#
# Arguments:
#   dataset_root  ZINC root with subsets/, geometries/, results/
#   cache_dir     Where compact shard caches live / will be written
#   scales        Space-separated shard counts (default: "1 10 50")
#
# Example:
#   bash scripts/scaling_sweep.sh zinc-250k zinc-250k/compact_cache "1 10 50"
set -euo pipefail

DATASET_ROOT="${1:?Usage: scaling_sweep.sh <dataset_root> <cache_dir> [scales]}"
CACHE_DIR="${2:?Usage: scaling_sweep.sh <dataset_root> <cache_dir> [scales]}"
SCALES="${3:-1 10 50}"
BASE_CONFIG="configs/validate_scaled.yaml"

# Preprocess enough shards for the largest scale (skip-if-exists makes this cheap).
MAX_SCALE=$(echo "$SCALES" | tr ' ' '\n' | sort -n | tail -1)
MAX_IDX=$((MAX_SCALE - 1))
IDS=$(seq -s, 0 "$MAX_IDX")
python -m qchem_gnn preprocess --dataset-root "$DATASET_ROOT" --subset-ids "$IDS" --cache-dir "$CACHE_DIR"

REPORT_ARGS=()
for SCALE in $SCALES; do
    LAST_IDX=$((SCALE - 1))
    IDS=$(seq -s, 0 "$LAST_IDX")
    OUT_DIR="runs/scaling_s${SCALE}"
    # Override subset_ids and output dir for this scale via a temp config.
    python - "$BASE_CONFIG" "$CACHE_DIR" "$IDS" "$OUT_DIR" <<'PY'
import sys, yaml
base, cache_dir, ids, out_dir = sys.argv[1:5]
cfg = yaml.safe_load(open(base))
cfg["pretrain"]["cache_dir"] = cache_dir
cfg["pretrain"]["subset_ids"] = [int(x) for x in ids.split(",")]
cfg["outputs"]["dir"] = out_dir
cfg["outputs"]["report"] = f"{out_dir}/report"
yaml.safe_dump(cfg, open(f"/tmp/scaling_cfg_{out_dir.replace('/', '_')}.yaml", "w"))
PY
    python -m qchem_gnn.validation --config "/tmp/scaling_cfg_${OUT_DIR//\//_}.yaml"
    REPORT_ARGS+=("${SCALE}=${OUT_DIR}/report.json")
done

echo "=== MAE vs scale ==="
python scripts/aggregate_scaling.py "${REPORT_ARGS[@]}"
```

- [ ] **Step 6: Make scripts executable and verify the aggregator runs**

Run:
```bash
chmod +x scripts/scaling_sweep.sh
python -c "from scripts.aggregate_scaling import aggregate_scaling; print('import ok')"
```
Expected: `import ok`.

- [ ] **Step 7: Commit**

```bash
git add scripts/scaling_sweep.sh scripts/aggregate_scaling.py tests/test_aggregate_scaling.py
git commit -m "feat(scale): scaling sweep script and MAE-vs-scale aggregator"
```

---

## Final verification

After all tasks:

- [ ] Run the full suite: `python -m pytest tests -q` — expected: all green.
- [ ] Smoke-test preprocessing on one real shard:
  `python -m qchem_gnn preprocess --dataset-root zinc-250k --subset-ids 0 --cache-dir zinc-250k/compact_cache`
  — expected: writes `zinc-250k/compact_cache/shard_000.pt` (~90 MB), and re-running prints the same path without re-extracting.
- [ ] Update `README.md` "Production training and inference scripts" section with the `preprocess` command and `scripts/scaling_sweep.sh` (fold into the existing section; commit separately).

## Notes on scope

- Phase 2 (shard-block streaming loader, shuffle buffer, MoCo memory-queue negatives) is **out of scope** per the design doc; do not build it. The preload path is correct until the dataset grows past ~1 M molecules.
- Do not modify `losses.py`, the model, the teacher heads, the checkpoint format, or the adapt subsystem.
