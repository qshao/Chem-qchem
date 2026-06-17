# Scaffold-Aware Negative Masking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add scaffold-aware negative masking to InfoNCE: molecules sharing a Murcko scaffold are excluded from each other's negative set, directly fixing the false-negative problem in contrastive pretraining on a small ZINC shard.

**Architecture:** A new pure function `build_scaffold_negative_mask` in `eval.py` computes an `[N, N]` bool mask once before training from Murcko scaffold identities. `info_nce_contrastive_loss` gains an optional `negative_mask` parameter that sets masked off-diagonal logits to `-inf`. The trainer slices the pre-computed mask per batch. Config/CLI plumb `use_scaffold_negmask: bool = False` through the existing contrastive plumbing. A `quantum_scaffold` arm is added to the experiment YAML, differing from `quantum` in exactly `use_scaffold_negmask: true`.

**Tech Stack:** Python 3.13+, PyTorch, RDKit (already a dependency — `_infer_scaffolds` and `MurckoScaffold` already used in `eval.py`), PyYAML, pytest.

## Global Constraints

- `build_scaffold_negative_mask(examples) -> torch.Tensor`: returns `[N, N]` CPU bool tensor; `True` at `[i, j]` iff `i ≠ j` and `scaffold[i] == scaffold[j]`. Diagonal always `False`.
- `info_nce_contrastive_loss` default `negative_mask=None` is byte-for-byte identical to current behaviour; no existing caller is affected.
- Masked logits: only off-diagonal entries in `negative_mask` that are `True` are set to `-inf`; the diagonal (positive pair) is never masked regardless of mask value.
- `use_scaffold_negmask` default is `False`; existing config files without the key continue to work.
- Matched ablation: `quantum_scaffold` arm vs `quantum` arm differs in exactly `use_scaffold_negmask: true`. Projection heads, teacher, temperature, batch size, seeds, conformer pooling are identical.
- Trainer calls `build_scaffold_negative_mask(examples)` once before the epoch loop (not per batch). The result is sliced per batch with `full_mask[global_idx][:, global_idx].to(device)`.
- The experiment reuses cached `baseline`, `quantum`, `quantum_vicreg` backbones; only three `quantum_scaffold_s*.pt` backbones are new.
- The full existing test suite (~170 tests) must pass; no change to the inference contract or checkpoint format.

---

### Task 1: `build_scaffold_negative_mask` in `eval.py`

**Files:**
- Modify: `qchem_gnn/eval.py` (insert after `_infer_scaffolds` which ends at line 172, before `_dataset_targets` at line 175)
- Test: `tests/test_scaffold_mask.py` (new file)

**Interfaces:**
- Consumes: `_infer_scaffolds(molecule_ids, smiles)` already in `eval.py` (line 164); `torch`.
- Produces: `build_scaffold_negative_mask(examples) -> torch.Tensor` — callable from `contrastive_pretrain.py` in Task 3. `examples` is any list of objects with a `.smiles: str` attribute (compatible with `MinimalQuantumExample`).

- [ ] **Step 1: Create the test file**

Create `tests/test_scaffold_mask.py`:

```python
from types import SimpleNamespace

import torch

from qchem_gnn.eval import build_scaffold_negative_mask


def _ex(smi):
    return SimpleNamespace(smiles=smi)


def test_shared_scaffold_entries_are_true():
    # Toluene and aniline both reduce to benzene under Murcko; ethanol is unique.
    examples = [_ex("Cc1ccccc1"), _ex("Nc1ccccc1"), _ex("CCO")]
    mask = build_scaffold_negative_mask(examples)
    assert mask.shape == (3, 3)
    assert mask[0, 1].item() is True
    assert mask[1, 0].item() is True


def test_unique_scaffolds_all_false():
    examples = [_ex("CO"), _ex("CCO"), _ex("CCN")]
    mask = build_scaffold_negative_mask(examples)
    assert not mask.any()


def test_diagonal_always_false():
    examples = [_ex("Cc1ccccc1"), _ex("Nc1ccccc1")]
    mask = build_scaffold_negative_mask(examples)
    assert not mask[0, 0].item()
    assert not mask[1, 1].item()


def test_cross_entries_with_unique_scaffold_are_false():
    examples = [_ex("Cc1ccccc1"), _ex("Nc1ccccc1"), _ex("CCO")]
    mask = build_scaffold_negative_mask(examples)
    assert not mask[0, 2].item()
    assert not mask[2, 0].item()
    assert not mask[1, 2].item()
    assert not mask[2, 1].item()


def test_returns_bool_cpu_tensor():
    examples = [_ex("CO"), _ex("CCO")]
    mask = build_scaffold_negative_mask(examples)
    assert mask.dtype == torch.bool
    assert not mask.is_cuda
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_scaffold_mask.py -q
```

Expected: `ImportError: cannot import name 'build_scaffold_negative_mask'`

- [ ] **Step 3: Implement `build_scaffold_negative_mask`**

In `qchem_gnn/eval.py`, insert after line 172 (after `_infer_scaffolds`), before line 175 (`_dataset_targets`):

```python
def build_scaffold_negative_mask(examples) -> torch.Tensor:
    n = len(examples)
    ids = [str(i) for i in range(n)]
    smiles = [ex.smiles for ex in examples]
    scaffolds = _infer_scaffolds(ids, smiles)
    groups: dict[str, list[int]] = {}
    for i, mol_id in enumerate(ids):
        s = scaffolds[mol_id]
        groups.setdefault(s, []).append(i)
    mask = torch.zeros(n, n, dtype=torch.bool)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        for i in indices:
            for j in indices:
                if i != j:
                    mask[i, j] = True
    return mask
```

Also add `import torch` at the top of `eval.py` if not already present. Check the existing imports first — `torch` is already imported via `from torch import nn`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_scaffold_mask.py -q
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/eval.py tests/test_scaffold_mask.py
git commit -m "feat(contrastive): build_scaffold_negative_mask from Murcko scaffold identities"
```

---

### Task 2: `info_nce_contrastive_loss` with `negative_mask`

**Files:**
- Modify: `qchem_gnn/losses.py` (lines 74–90, the `info_nce_contrastive_loss` function)
- Test: `tests/test_contrastive_loss.py` (append)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `info_nce_contrastive_loss(z_a, z_b, temperature=0.1, negative_mask=None)` — the new signature used by Task 3. Existing callers passing only `z_a, z_b, temperature` are unaffected.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_contrastive_loss.py`:

```python
def test_infonce_negative_mask_none_is_default_behaviour():
    torch.manual_seed(0)
    z = torch.randn(4, 8)
    assert torch.allclose(
        info_nce_contrastive_loss(z, z.clone()),
        info_nce_contrastive_loss(z, z.clone(), negative_mask=None),
    )


def test_infonce_all_false_mask_unchanged():
    torch.manual_seed(1)
    z = torch.randn(4, 8)
    mask = torch.zeros(4, 4, dtype=torch.bool)
    assert torch.allclose(
        info_nce_contrastive_loss(z, z.clone()),
        info_nce_contrastive_loss(z, z.clone(), negative_mask=mask),
    )


def test_infonce_full_off_diagonal_mask_zero_loss():
    # With all negatives masked only the positive remains;
    # log(softmax([x, -inf, -inf, ...])[0]) = 0.
    torch.manual_seed(2)
    z = torch.randn(4, 8)
    mask = ~torch.eye(4, dtype=torch.bool)
    loss = info_nce_contrastive_loss(z, z.clone(), negative_mask=mask)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_infonce_partial_mask_lowers_loss():
    # Fewer negatives -> smaller denominator -> higher positive probability -> lower loss.
    torch.manual_seed(3)
    z = torch.randn(8, 4)
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[0, 1] = True
    mask[1, 0] = True
    loss_unmasked = info_nce_contrastive_loss(z, z.clone())
    loss_masked = info_nce_contrastive_loss(z, z.clone(), negative_mask=mask)
    assert float(loss_masked) <= float(loss_unmasked) + 1e-5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_contrastive_loss.py -q -k "negative_mask"
```

Expected: `TypeError: info_nce_contrastive_loss() got an unexpected keyword argument 'negative_mask'`

- [ ] **Step 3: Add `negative_mask` to `info_nce_contrastive_loss`**

In `qchem_gnn/losses.py`, replace the entire `info_nce_contrastive_loss` function (lines 74–90):

```python
def info_nce_contrastive_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.1,
    negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if z_a.shape != z_b.shape:
        raise ValueError("z_a and z_b must have the same shape")
    if z_a.shape[0] < 2:
        raise ValueError("contrastive loss needs at least 2 examples for in-batch negatives")

    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = (z_a @ z_b.t()) / temperature
    if negative_mask is not None:
        eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(negative_mask & ~eye, float("-inf"))
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_ab + loss_ba)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_contrastive_loss.py -q
```

Expected: all tests pass (existing InfoNCE + VICReg tests + 4 new mask tests)

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/losses.py tests/test_contrastive_loss.py
git commit -m "feat(contrastive): optional negative_mask parameter in info_nce_contrastive_loss"
```

---

### Task 3: Trainer plumbing

**Files:**
- Modify: `qchem_gnn/contrastive_pretrain.py` (import at line 13; new import line; signature at lines 99–100; pre-loop mask init after line 134; batch slice after line 157; InfoNCE call at lines 203–206)
- Test: `tests/test_contrastive_pretrain.py` (append)

**Interfaces:**
- Consumes: `build_scaffold_negative_mask` (Task 1); `info_nce_contrastive_loss(..., negative_mask=...)` (Task 2).
- Produces: `contrastive_pretrain_on_dataset(..., use_scaffold_negmask: bool = False, ...)` — the new signature used by config/CLI (Task 4) and by the harness via `kwargs.update(arm_overrides)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_contrastive_pretrain.py` (`import math`, `contrastive_pretrain_on_dataset`, and `make_tiny_quantum_dataset` are already imported at the top of that file — do not re-add them):

```python
def test_contrastive_pretrain_scaffold_mask_runs(tmp_path):
    # The tiny fixture uses CO/CCO/CCN/CCC — all unique scaffolds,
    # so the mask is all-False. This is a regression test that the
    # code path runs without error and produces a finite loss.
    dataset = make_tiny_quantum_dataset(tmp_path)
    result = contrastive_pretrain_on_dataset(
        dataset,
        hidden_dim=16,
        hidden_dim_3d=16,
        epochs=2,
        batch_size=4,
        teacher_weight=1.0,
        conformer_pool_mode="energy",
        use_scaffold_negmask=True,
        seed=0,
    )
    assert result.loss_history
    assert math.isfinite(result.loss_history[-1])


def test_contrastive_pretrain_scaffold_mask_default_false(tmp_path):
    # Omitting use_scaffold_negmask produces the same result as use_scaffold_negmask=False.
    dataset = make_tiny_quantum_dataset(tmp_path)
    result = contrastive_pretrain_on_dataset(
        dataset, hidden_dim=16, hidden_dim_3d=16, epochs=2, batch_size=4, seed=0,
    )
    assert result.loss_history
    assert math.isfinite(result.loss_history[-1])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_contrastive_pretrain.py -q -k "scaffold"
```

Expected: `TypeError: contrastive_pretrain_on_dataset() got an unexpected keyword argument 'use_scaffold_negmask'`

- [ ] **Step 3: Add the eval import**

In `qchem_gnn/contrastive_pretrain.py`, after the existing imports block (which ends around line 17), add a new import line:

```python
from .eval import build_scaffold_negative_mask
```

Add it after line 16 (`from .splits import scaffold_or_random_split`) so it reads:

```python
from .splits import scaffold_or_random_split
from .eval import build_scaffold_negative_mask
```

- [ ] **Step 4: Add `use_scaffold_negmask` to the signature**

In `contrastive_pretrain_on_dataset`, replace lines 99–100:

```python
    vicreg_cov_weight: float = 1.0,
    seed: int = 0,
```

with:

```python
    vicreg_cov_weight: float = 1.0,
    use_scaffold_negmask: bool = False,
    seed: int = 0,
```

- [ ] **Step 5: Compute the full mask before the epoch loop**

Replace line 134–135:

```python
    num_examples = len(examples)

    for _ in range(epochs):
```

with:

```python
    num_examples = len(examples)

    full_mask: torch.Tensor | None = None
    if use_scaffold_negmask:
        full_mask = build_scaffold_negative_mask(examples)

    for _ in range(epochs):
```

- [ ] **Step 6: Slice the batch mask and pass it to InfoNCE**

In the batch loop, replace lines 157–158:

```python
                coords_index = [pos for pos, _ in usable]
                usable_examples = [ex for _, ex in usable]
```

with:

```python
                coords_index = [pos for pos, _ in usable]
                usable_examples = [ex for _, ex in usable]
                batch_mask: torch.Tensor | None = None
                if full_mask is not None:
                    global_idx = [batch_indices[p] for p in coords_index]
                    batch_mask = full_mask[global_idx][:, global_idx].to(device)
```

Then replace lines 203–206 (the InfoNCE branch):

```python
                    elif contrastive_loss == "infonce":
                        contrastive = info_nce_contrastive_loss(
                            view_2d, view_3d, temperature=temperature
                        )
```

with:

```python
                    elif contrastive_loss == "infonce":
                        contrastive = info_nce_contrastive_loss(
                            view_2d, view_3d, temperature=temperature,
                            negative_mask=batch_mask,
                        )
```

- [ ] **Step 7: Run all pretrain tests**

```bash
python -m pytest tests/test_contrastive_pretrain.py -q
```

Expected: all pass (existing tests + 2 new scaffold tests)

- [ ] **Step 8: Commit**

```bash
git add qchem_gnn/contrastive_pretrain.py tests/test_contrastive_pretrain.py
git commit -m "feat(contrastive): use_scaffold_negmask plumbing in trainer"
```

---

### Task 4: Config and CLI plumbing

**Files:**
- Modify: `qchem_gnn/config.py` (`VALID_SECTION_KEYS["contrastive"]` lines 33–50; `DEFAULT_CONFIG["contrastive"]` lines 80–97; `_validate_config` after line 248; `config_to_namespace` lines 410–416)
- Modify: `qchem_gnn/cli.py` (argparse at line 98; `_config_from_args` tuple at lines 196–200; `run_contrastive_pretrain` call at lines 481–486)
- Test: `tests/test_config.py` (append)

**Interfaces:**
- Consumes: `use_scaffold_negmask` param from Task 3.
- Produces: `contrastive` config section accepts `use_scaffold_negmask` (bool, default `False`); `config_to_namespace` exposes it as `ns.use_scaffold_negmask`; `cli.py` `--use-scaffold-negmask` store-true flag. In the validation harness, `use_scaffold_negmask: true` in an arm's override flows to the trainer via the existing `kwargs.update(arm_overrides)` in `_pretrain_kwargs` — no harness wiring needed.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py` (`pytest`, `ConfigError`, `config_to_namespace`, and `resolve_config` are already imported at the top of that file — do not re-add them):

```python
def _scaffold_base(extra_contrastive=None):
    return {
        "command": "contrastive-pretrain",
        "dataset": {"dataset_root": "zinc-250k", "subset_ids": [44]},
        "contrastive": extra_contrastive or {},
        "outputs": {"checkpoint": "out.pt"},
    }


def test_config_default_scaffold_negmask_is_false():
    cfg = resolve_config(_scaffold_base())
    assert cfg["contrastive"]["use_scaffold_negmask"] is False


def test_config_accepts_scaffold_negmask_true():
    cfg = resolve_config(_scaffold_base({"use_scaffold_negmask": True}))
    assert cfg["contrastive"]["use_scaffold_negmask"] is True


def test_config_rejects_string_scaffold_negmask():
    with pytest.raises(ConfigError):
        resolve_config(_scaffold_base({"use_scaffold_negmask": "yes"}))


def test_namespace_includes_scaffold_negmask():
    cfg = resolve_config(_scaffold_base({"use_scaffold_negmask": True}))
    ns = config_to_namespace(cfg)
    assert ns.use_scaffold_negmask is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_config.py -q -k "scaffold"
```

Expected: `KeyError` on `use_scaffold_negmask` (key not in config yet)

- [ ] **Step 3: Add the key to `VALID_SECTION_KEYS` and defaults**

In `config.py`, in `VALID_SECTION_KEYS["contrastive"]` (lines 33–50), add `"use_scaffold_negmask"` after `"vicreg_cov_weight"`:

```python
    "contrastive": {
        "batch_size",
        "supervised_weight",
        "contrastive_weight",
        "temperature",
        "teacher_weight",
        "energy_temperature",
        "hidden_dim_3d",
        "num_rbf",
        "cutoff",
        "message_passing_steps_3d",
        "conformer_pool_mode",
        "contrastive_loss",
        "vicreg_sim_weight",
        "vicreg_var_weight",
        "vicreg_cov_weight",
        "use_scaffold_negmask",
        "seed",
    },
```

In `DEFAULT_CONFIG["contrastive"]` (lines 80–97), add `"use_scaffold_negmask": False` after `"vicreg_cov_weight": 1.0`:

```python
    "contrastive": {
        "batch_size": 8,
        "supervised_weight": 1.0,
        "contrastive_weight": 1.0,
        "temperature": 0.1,
        "teacher_weight": 1.0,
        "energy_temperature": 298.15,
        "hidden_dim_3d": 32,
        "num_rbf": 16,
        "cutoff": 5.0,
        "message_passing_steps_3d": 2,
        "conformer_pool_mode": "mean",
        "contrastive_loss": "infonce",
        "vicreg_sim_weight": 25.0,
        "vicreg_var_weight": 25.0,
        "vicreg_cov_weight": 1.0,
        "use_scaffold_negmask": False,
        "seed": 0,
    },
```

- [ ] **Step 4: Add validation**

In `_validate_config`, after the VICReg weight checks (after line 248):

```python
    if not isinstance(contrastive["use_scaffold_negmask"], bool):
        raise ConfigError("contrastive.use_scaffold_negmask must be a boolean")
```

- [ ] **Step 5: Add to `config_to_namespace`**

In `config_to_namespace`, in the `values.update({...})` block for `command == "contrastive-pretrain"` (lines 410–416), add `"use_scaffold_negmask"` after `"vicreg_cov_weight"`:

```python
                    "vicreg_cov_weight": _coerce_float(contrastive["vicreg_cov_weight"], "contrastive.vicreg_cov_weight"),
                    "use_scaffold_negmask": bool(contrastive["use_scaffold_negmask"]),
                    "seed": _coerce_int(contrastive["seed"], "contrastive.seed"),
```

- [ ] **Step 6: Add the CLI argument**

In `qchem_gnn/cli.py`, after line 98 (`contrastive.add_argument("--vicreg-cov-weight", ...)`), add:

```python
    contrastive.add_argument("--use-scaffold-negmask", action="store_true",
                              default=argparse.SUPPRESS,
                              help="Mask scaffold-similar negatives in InfoNCE")
```

(`default=argparse.SUPPRESS` means the key is absent from `args` when the flag is not passed, so it does not override the YAML config value.)

- [ ] **Step 7: Map the arg in `_config_from_args`**

In `_config_from_args`, in the contrastive tuple (ending at lines 199–200), add before `("seed", "seed")`:

```python
                ("vicreg_cov_weight", "vicreg_cov_weight"),
                ("use_scaffold_negmask", "use_scaffold_negmask"),
                ("seed", "seed"),
```

- [ ] **Step 8: Pass to the trainer in `run_contrastive_pretrain`**

In `run_contrastive_pretrain`, the call currently ends at lines 484–486:

```python
        vicreg_cov_weight=args.vicreg_cov_weight,
        seed=args.seed,
    )
```

Replace with:

```python
        vicreg_cov_weight=args.vicreg_cov_weight,
        use_scaffold_negmask=args.use_scaffold_negmask,
        seed=args.seed,
    )
```

- [ ] **Step 9: Run the config and CLI tests**

```bash
python -m pytest tests/test_config.py tests/test_cli.py -q
```

Expected: all pass (4 new scaffold tests + all existing)

- [ ] **Step 10: Commit**

```bash
git add qchem_gnn/config.py qchem_gnn/cli.py tests/test_config.py
git commit -m "feat(contrastive): plumb use_scaffold_negmask through config and CLI"
```

---

### Task 5: Add `quantum_scaffold` arm and extend the integration test

**Files:**
- Modify: `configs/validate_quantum_teacher.yaml` (arms block + comparisons block)
- Modify: `tests/test_validation_end_to_end.py` (extend `test_run_validation_writes_report`)

**Interfaces:**
- Consumes: `use_scaffold_negmask` arm override (Task 4) flowing through `_pretrain_kwargs`'s `kwargs.update(arm_overrides)` in `qchem_gnn/validation.py` — no harness change needed.
- Produces: four-arm experiment config (`baseline`, `quantum`, `quantum_vicreg`, `quantum_scaffold`) with three comparisons; integration test asserts the fourth backbone and three verdict blocks.

- [ ] **Step 1: Extend the end-to-end test**

In `tests/test_validation_end_to_end.py`, replace the `"arms"` and `"comparisons"` block in the cfg dict:

```python
        "arms": {"baseline": {"teacher_weight": 0.0, "conformer_pool_mode": "mean"},
                 "quantum": {"teacher_weight": 1.0, "conformer_pool_mode": "energy"},
                 "quantum_vicreg": {"teacher_weight": 1.0, "conformer_pool_mode": "energy",
                                    "contrastive_loss": "vicreg"}},
        "comparisons": [
            {"name": "teacher_vs_baseline", "reference": "baseline", "treatment": "quantum"},
            {"name": "vicreg_vs_infonce", "reference": "quantum", "treatment": "quantum_vicreg"},
        ],
```

with:

```python
        "arms": {"baseline": {"teacher_weight": 0.0, "conformer_pool_mode": "mean"},
                 "quantum": {"teacher_weight": 1.0, "conformer_pool_mode": "energy"},
                 "quantum_vicreg": {"teacher_weight": 1.0, "conformer_pool_mode": "energy",
                                    "contrastive_loss": "vicreg"},
                 "quantum_scaffold": {"teacher_weight": 1.0, "conformer_pool_mode": "energy",
                                      "use_scaffold_negmask": True}},
        "comparisons": [
            {"name": "teacher_vs_baseline", "reference": "baseline", "treatment": "quantum"},
            {"name": "vicreg_vs_infonce", "reference": "quantum", "treatment": "quantum_vicreg"},
            {"name": "scaffold_vs_infonce", "reference": "quantum", "treatment": "quantum_scaffold"},
        ],
```

Then replace the tail assertions (after `assert "vicreg_vs_infonce" in report_text`):

```python
    assert "vicreg_vs_infonce" in report_text
    assert (out_dir / "quantum_scaffold_s0.pt").exists()
    assert len(aggregate["verdicts"]) == 3
    assert {v["name"] for v in aggregate["verdicts"]} == {
        "teacher_vs_baseline", "vicreg_vs_infonce", "scaffold_vs_infonce"
    }
    assert "scaffold_vs_infonce" in report_text
```

- [ ] **Step 2: Run the end-to-end test**

```bash
python -m pytest tests/test_validation_end_to_end.py -q
```

Expected: PASS. This drives `run_validation` with the inline four-arm cfg and exercises the scaffold mask path end-to-end (tiny fixture molecules CO/CCO/CCN/CCC all have unique scaffolds, so the mask is all-False — a valid regression). If it FAILS, the failure pinpoints a real break in the scaffold arm path; fix before continuing.

- [ ] **Step 3: Update the experiment YAML**

In `configs/validate_quantum_teacher.yaml`, replace the `arms:` and `comparisons:` blocks:

```yaml
arms:
  baseline:         { teacher_weight: 0.0, conformer_pool_mode: mean }
  quantum:          { teacher_weight: 1.0, conformer_pool_mode: energy, energy_temperature: 298.15 }
  quantum_vicreg:   { teacher_weight: 1.0, conformer_pool_mode: energy, energy_temperature: 298.15,
                      contrastive_loss: vicreg }
  quantum_scaffold: { teacher_weight: 1.0, conformer_pool_mode: energy, energy_temperature: 298.15,
                      use_scaffold_negmask: true }

comparisons:
  - { name: teacher_vs_baseline,  reference: baseline,  treatment: quantum }
  - { name: vicreg_vs_infonce,    reference: quantum,    treatment: quantum_vicreg }
  - { name: scaffold_vs_infonce,  reference: quantum,    treatment: quantum_scaffold }
```

- [ ] **Step 4: Verify the YAML parses correctly**

```bash
python -c "
from qchem_gnn.validation import load_validation_config
c = load_validation_config('configs/validate_quantum_teacher.yaml')
print(list(c['arms']))
print([x['name'] for x in c['comparisons']])
"
```

Expected output:
```
['baseline', 'quantum', 'quantum_vicreg', 'quantum_scaffold']
['teacher_vs_baseline', 'vicreg_vs_infonce', 'scaffold_vs_infonce']
```

- [ ] **Step 5: Run the full test suite**

```bash
python -m pytest tests -q
```

Expected: all pass (~175+ tests including all new scaffold tests from Tasks 1–5)

- [ ] **Step 6: Commit**

```bash
git add configs/validate_quantum_teacher.yaml tests/test_validation_end_to_end.py
git commit -m "feat(validation): add quantum_scaffold arm and scaffold_vs_infonce comparison"
```

---

## Notes for the implementer

- The real experiment run is `python -m qchem_gnn.validation --config configs/validate_quantum_teacher.yaml`. It reuses the nine cached backbones (`baseline_s{0,1,2}.pt`, `quantum_s{0,1,2}.pt`, `quantum_vicreg_s{0,1,2}.pt`) and trains only the three `quantum_scaffold_s*.pt` backbones. Do NOT run this as part of the plan — the synthetic end-to-end test in Task 5 covers the code path.
- `build_scaffold_negative_mask` lives in `eval.py` because that file already owns `_infer_scaffolds` and all RDKit chemistry utilities.
- The `device = next(model.parameters()).device` line in the trainer (line 134) runs before the epoch loop — the device is known at mask-init time, but the mask itself is created on CPU and moved per-batch to avoid keeping a large GPU tensor pinned for the whole run.
- `default=argparse.SUPPRESS` on `--use-scaffold-negmask` ensures the CLI flag does not override YAML `use_scaffold_negmask: true` with `False` when the flag is absent. Without `SUPPRESS`, `store_true` defaults to `False`, which would always overwrite the config.
- The harness does not need to be changed: `_pretrain_kwargs` already does `kwargs.update(arm_overrides)`, so `use_scaffold_negmask: true` in an arm override dict flows directly to `contrastive_pretrain_on_dataset`.
