# VICReg Contrastive Objective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a negative-free VICReg contrastive objective as a drop-in alternative to InfoNCE, plumb it through config/CLI, generalize the validation harness to N arms with pairwise verdicts, and add a `quantum_vicreg` arm to the existing study so the harness measures whether VICReg beats InfoNCE.

**Architecture:** A new pure `vicreg_loss` in `losses.py` mirrors the existing `info_nce_contrastive_loss` shape contract. `contrastive_pretrain_on_dataset` gains a `contrastive_loss` switch that branches at the single contrastive call site after the projection heads — everything else (pairing, teacher, supervised loss, dims) is identical between arms. The validation harness changes its hardcoded two-arm/single-verdict logic to config-driven N arms and a list of pairwise comparisons, backward-compatible when `comparisons` is omitted.

**Tech Stack:** Python 3.13+, PyTorch, PyYAML, pytest.

## Global Constraints

- VICReg loss: `L = sim_weight*inv + var_weight*var + cov_weight*cov`; canonical defaults `sim_weight=25.0`, `var_weight=25.0`, `cov_weight=1.0`, `gamma=1.0`, `eps=1e-4`.
- `vicreg_loss(z_a, z_b, ...)` shape contract matches `info_nce_contrastive_loss`: inputs are `[batch, D]`, must be equal shape, and `batch < 2` raises `ValueError`.
- Matched ablation: the InfoNCE arm and the VICReg arm differ in EXACTLY the contrastive loss function. Projection heads stay `hidden_dim → hidden_dim` (no wide expander). The 2D↔3D pairing, teacher, supervised loss, dims, epochs, learning rate are identical.
- InfoNCE remains the default everywhere (`contrastive_loss: str = "infonce"`); every existing caller is unaffected.
- `contrastive_loss` valid values are exactly `{"infonce", "vicreg"}`; invalid values are rejected at config validation, not at training time.
- The VICReg weights are non-negative floats validated with `_ensure_non_negative_float`.
- Verdict heuristic is unchanged per comparison: on frozen `mlp_head` MAE, *helps* iff `mean_reference − mean_treatment > sqrt(std_ref² + std_treat²)`; `n < 2` successful seeds in either arm → `insufficient seeds`; `n == 0` → `n/a`.
- The harness is backward-compatible: when `comparisons` is absent, it defaults to a single `{name: teacher_vs_baseline, reference: baseline, treatment: quantum}` verdict.
- The experiment reuses cached `baseline`/`quantum` backbones in `runs/validate/`; only the three `quantum_vicreg_s*.pt` backbones are new.
- The full existing test suite (~155 tests) must pass; no change to the inference contract, checkpoint format, or model logic beyond the single contrastive call site.

---

### Task 1: `vicreg_loss` pure function

**Files:**
- Modify: `qchem_gnn/losses.py` (add `_off_diagonal` + `vicreg_loss` after `info_nce_contrastive_loss`, ending at line 91)
- Test: `tests/test_contrastive_loss.py`

**Interfaces:**
- Consumes: nothing from other tasks; `torch`, `torch.nn.functional as F` (already imported in `losses.py`).
- Produces: `vicreg_loss(z_a: torch.Tensor, z_b: torch.Tensor, *, sim_weight: float = 25.0, var_weight: float = 25.0, cov_weight: float = 1.0, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor`. Raises `ValueError` on shape mismatch and on `batch < 2`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_contrastive_loss.py`:

```python
import pytest
import torch

from qchem_gnn.losses import vicreg_loss


def test_vicreg_identical_views_zero_invariance():
    # With only the invariance term active, identical views give ~0 loss.
    z = torch.randn(8, 4)
    loss = vicreg_loss(z, z.clone(), sim_weight=1.0, var_weight=0.0, cov_weight=0.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_vicreg_penalizes_collapse():
    # A collapsed (constant) embedding incurs a large variance penalty;
    # a high-variance embedding incurs ~none.
    collapsed = torch.ones(8, 4)
    varied = torch.randn(8, 4) * 5.0
    loss_collapsed = vicreg_loss(collapsed, collapsed, sim_weight=0.0, var_weight=1.0, cov_weight=0.0)
    loss_varied = vicreg_loss(varied, varied, sim_weight=0.0, var_weight=1.0, cov_weight=0.0)
    assert float(loss_collapsed) > float(loss_varied)
    assert float(loss_collapsed) > 1.0


def test_vicreg_covariance_penalizes_correlated_dims():
    # Two perfectly correlated dims -> non-zero covariance term;
    # two orthogonal dims -> ~zero covariance term.
    correlated = torch.tensor([[1.0, 2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, -2.0]])
    decorrelated = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    loss_corr = vicreg_loss(correlated, correlated, sim_weight=0.0, var_weight=0.0, cov_weight=1.0)
    loss_dec = vicreg_loss(decorrelated, decorrelated, sim_weight=0.0, var_weight=0.0, cov_weight=1.0)
    assert float(loss_corr) > float(loss_dec)
    assert float(loss_dec) == pytest.approx(0.0, abs=1e-6)


def test_vicreg_requires_two_examples():
    with pytest.raises(ValueError):
        vicreg_loss(torch.randn(1, 4), torch.randn(1, 4))


def test_vicreg_shape_mismatch_raises():
    with pytest.raises(ValueError):
        vicreg_loss(torch.randn(8, 4), torch.randn(8, 5))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_contrastive_loss.py -q -k vicreg`
Expected: FAIL with `ImportError: cannot import name 'vicreg_loss'`

- [ ] **Step 3: Implement `_off_diagonal` and `vicreg_loss`**

Append to `qchem_gnn/losses.py` (after `info_nce_contrastive_loss`, which currently ends at line 91):

```python
def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    n = matrix.shape[0]
    return matrix.flatten()[:-1].reshape(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    *,
    sim_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    if z_a.shape != z_b.shape:
        raise ValueError("z_a and z_b must have the same shape")
    if z_a.shape[0] < 2:
        raise ValueError("vicreg loss needs at least 2 examples for variance/covariance statistics")

    batch, dim = z_a.shape

    # Invariance: pull the two views together.
    inv = F.mse_loss(z_a, z_b)

    # Variance: hinge keeps each dimension's batch std above gamma (anti-collapse).
    std_a = torch.sqrt(z_a.var(dim=0) + eps)
    std_b = torch.sqrt(z_b.var(dim=0) + eps)
    var = torch.relu(gamma - std_a).mean() + torch.relu(gamma - std_b).mean()

    # Covariance: decorrelate dimensions via the off-diagonal of the empirical covariance.
    za_c = z_a - z_a.mean(dim=0)
    zb_c = z_b - z_b.mean(dim=0)
    cov_a = (za_c.t() @ za_c) / (batch - 1)
    cov_b = (zb_c.t() @ zb_c) / (batch - 1)
    cov = _off_diagonal(cov_a).pow(2).sum() / dim + _off_diagonal(cov_b).pow(2).sum() / dim

    return sim_weight * inv + var_weight * var + cov_weight * cov
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_contrastive_loss.py -q -k vicreg`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/losses.py tests/test_contrastive_loss.py
git commit -m "feat(contrastive): add negative-free VICReg loss"
```

---

### Task 2: Trainer switch (`contrastive_loss`)

**Files:**
- Modify: `qchem_gnn/contrastive_pretrain.py` (import at line 13; signature at lines 76-95; contrastive call site at lines 186-189)
- Test: `tests/test_contrastive_pretrain.py`

**Interfaces:**
- Consumes: `vicreg_loss` (Task 1).
- Produces: `contrastive_pretrain_on_dataset(..., contrastive_loss: str = "infonce", vicreg_sim_weight: float = 25.0, vicreg_var_weight: float = 25.0, vicreg_cov_weight: float = 1.0, ...)`. When `contrastive_loss == "vicreg"`, the contrastive term is `vicreg_loss(view_2d, view_3d, ...)`; otherwise it is the existing InfoNCE term. The two projected views `view_2d = proj_2d(molecule_2d)` and `view_3d = proj_3d(molecule_3d)` are unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_contrastive_pretrain.py`:

```python
import math

from qchem_gnn.contrastive_pretrain import contrastive_pretrain_on_dataset
from tests._validation_fixtures import make_tiny_quantum_dataset


def test_contrastive_pretrain_vicreg_runs(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    result = contrastive_pretrain_on_dataset(
        dataset,
        hidden_dim=16,
        hidden_dim_3d=16,
        epochs=2,
        batch_size=4,
        teacher_weight=1.0,
        conformer_pool_mode="energy",
        contrastive_loss="vicreg",
        seed=0,
    )
    assert result.loss_history
    assert math.isfinite(result.loss_history[-1])


def test_contrastive_pretrain_infonce_still_default(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    result = contrastive_pretrain_on_dataset(
        dataset, hidden_dim=16, hidden_dim_3d=16, epochs=2, batch_size=4, seed=0,
    )
    assert result.loss_history
    assert math.isfinite(result.loss_history[-1])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_contrastive_pretrain.py -q -k vicreg`
Expected: FAIL with `TypeError: contrastive_pretrain_on_dataset() got an unexpected keyword argument 'contrastive_loss'`

- [ ] **Step 3: Add the import**

In `qchem_gnn/contrastive_pretrain.py`, change the import at line 13 from:

```python
from .losses import compute_multitask_loss, info_nce_contrastive_loss
```

to:

```python
from .losses import compute_multitask_loss, info_nce_contrastive_loss, vicreg_loss
```

- [ ] **Step 4: Add the parameters to the signature**

In `contrastive_pretrain_on_dataset`, the signature currently has these lines (90-93):

```python
    teacher_weight: float = 1.0,
    energy_temperature: float = 298.15,
    conformer_pool_mode: str = "mean",
    seed: int = 0,
```

Replace them with:

```python
    teacher_weight: float = 1.0,
    energy_temperature: float = 298.15,
    conformer_pool_mode: str = "mean",
    contrastive_loss: str = "infonce",
    vicreg_sim_weight: float = 25.0,
    vicreg_var_weight: float = 25.0,
    vicreg_cov_weight: float = 1.0,
    seed: int = 0,
```

- [ ] **Step 5: Branch at the contrastive call site**

In the same function, the contrastive block currently reads (lines 186-189):

```python
                    molecule_2d = model_output.mol_embedding[coords_index]
                    contrastive = info_nce_contrastive_loss(
                        proj_2d(molecule_2d), proj_3d(molecule_3d), temperature=temperature
                    )
```

Replace it with:

```python
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_contrastive_pretrain.py -q`
Expected: PASS (existing tests + the 2 new ones)

- [ ] **Step 7: Commit**

```bash
git add qchem_gnn/contrastive_pretrain.py tests/test_contrastive_pretrain.py
git commit -m "feat(contrastive): contrastive_loss switch selecting InfoNCE or VICReg"
```

---

### Task 3: Config and CLI plumbing

**Files:**
- Modify: `qchem_gnn/config.py` (`VALID_SECTION_KEYS["contrastive"]` lines 33-46; `DEFAULT_CONFIG["contrastive"]` lines 76-89; `_validate_config` near lines 226-229; `config_to_namespace` contrastive update near lines 379-394)
- Modify: `qchem_gnn/cli.py` (argparse near line 93; `_config_from_args` tuple lines 180-193; `run_contrastive_pretrain` call lines 456-474)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `contrastive_pretrain_on_dataset` VICReg kwargs (Task 2).
- Produces: config section `contrastive` accepts `contrastive_loss` (`"infonce"`|`"vicreg"`, default `"infonce"`) and `vicreg_sim_weight`/`vicreg_var_weight`/`vicreg_cov_weight` (non-negative floats, defaults 25.0/25.0/1.0). `config_to_namespace` exposes them as `ns.contrastive_loss`, `ns.vicreg_sim_weight`, `ns.vicreg_var_weight`, `ns.vicreg_cov_weight` for the `contrastive-pretrain` command.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
import pytest

from qchem_gnn.config import ConfigError, config_to_namespace, resolve_config


def _vicreg_base(extra_contrastive=None):
    return {
        "command": "contrastive-pretrain",
        "dataset": {"dataset_root": "zinc-250k", "subset_ids": [44]},
        "contrastive": extra_contrastive or {},
        "outputs": {"checkpoint": "out.pt"},
    }


def test_config_default_contrastive_loss_is_infonce():
    cfg = resolve_config(_vicreg_base())
    assert cfg["contrastive"]["contrastive_loss"] == "infonce"
    assert cfg["contrastive"]["vicreg_sim_weight"] == 25.0
    assert cfg["contrastive"]["vicreg_var_weight"] == 25.0
    assert cfg["contrastive"]["vicreg_cov_weight"] == 1.0


def test_config_accepts_vicreg_contrastive_loss():
    cfg = resolve_config(_vicreg_base({"contrastive_loss": "vicreg", "vicreg_cov_weight": 2.0}))
    assert cfg["contrastive"]["contrastive_loss"] == "vicreg"
    assert cfg["contrastive"]["vicreg_cov_weight"] == 2.0


def test_config_rejects_invalid_contrastive_loss():
    with pytest.raises(ConfigError):
        resolve_config(_vicreg_base({"contrastive_loss": "barlow"}))


def test_config_rejects_negative_vicreg_weight():
    with pytest.raises(ConfigError):
        resolve_config(_vicreg_base({"vicreg_sim_weight": -1.0}))


def test_namespace_includes_vicreg_fields():
    cfg = resolve_config(_vicreg_base({"contrastive_loss": "vicreg"}))
    ns = config_to_namespace(cfg)
    assert ns.contrastive_loss == "vicreg"
    assert ns.vicreg_sim_weight == 25.0
    assert ns.vicreg_cov_weight == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_config.py -q -k "vicreg or contrastive_loss"`
Expected: FAIL — the `default`/`accepts`/`namespace` tests fail with `KeyError`/`AttributeError` (the keys don't exist on the resolved config yet), and the `rejects` tests fail because no `ConfigError` is raised. Note: passing an unknown key like `contrastive_loss` in the input currently raises `ConfigError` "contrastive section has unknown keys", which is why the keys must be added to `VALID_SECTION_KEYS` in Step 3.

- [ ] **Step 3: Add the keys to `VALID_SECTION_KEYS` and defaults**

In `qchem_gnn/config.py`, the `"contrastive"` set in `VALID_SECTION_KEYS` (lines 33-46) ends with `"conformer_pool_mode",` and `"seed",`. Add the four keys so the set reads:

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
        "seed",
    },
```

Then in `DEFAULT_CONFIG["contrastive"]` (lines 76-89), which ends with `"conformer_pool_mode": "mean",` and `"seed": 0,`, add the defaults so it reads:

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
        "seed": 0,
    },
```

- [ ] **Step 4: Add validation**

In `_validate_config`, immediately after the `conformer_pool_mode` check (lines 228-229):

```python
    if contrastive["conformer_pool_mode"] not in {"mean", "weighted", "energy"}:
        raise ConfigError("contrastive.conformer_pool_mode must be one of mean, weighted, energy")
```

add:

```python
    if contrastive["contrastive_loss"] not in {"infonce", "vicreg"}:
        raise ConfigError("contrastive.contrastive_loss must be one of infonce, vicreg")
    contrastive["vicreg_sim_weight"] = _ensure_non_negative_float(
        contrastive["vicreg_sim_weight"], "contrastive.vicreg_sim_weight"
    )
    contrastive["vicreg_var_weight"] = _ensure_non_negative_float(
        contrastive["vicreg_var_weight"], "contrastive.vicreg_var_weight"
    )
    contrastive["vicreg_cov_weight"] = _ensure_non_negative_float(
        contrastive["vicreg_cov_weight"], "contrastive.vicreg_cov_weight"
    )
```

- [ ] **Step 5: Add to `config_to_namespace`**

In `config_to_namespace`, the `values.update({...})` block for `command == "contrastive-pretrain"` (lines 379-394) currently ends with:

```python
                    "conformer_pool_mode": contrastive["conformer_pool_mode"],
                    "seed": _coerce_int(contrastive["seed"], "contrastive.seed"),
                }
            )
```

Insert the four fields before `"seed"`:

```python
                    "conformer_pool_mode": contrastive["conformer_pool_mode"],
                    "contrastive_loss": contrastive["contrastive_loss"],
                    "vicreg_sim_weight": _coerce_float(contrastive["vicreg_sim_weight"], "contrastive.vicreg_sim_weight"),
                    "vicreg_var_weight": _coerce_float(contrastive["vicreg_var_weight"], "contrastive.vicreg_var_weight"),
                    "vicreg_cov_weight": _coerce_float(contrastive["vicreg_cov_weight"], "contrastive.vicreg_cov_weight"),
                    "seed": _coerce_int(contrastive["seed"], "contrastive.seed"),
                }
            )
```

- [ ] **Step 6: Add the CLI arguments**

In `qchem_gnn/cli.py`, after the `--conformer-pool-mode` argument (line 93):

```python
    contrastive.add_argument("--conformer-pool-mode", help="Conformer pooling mode: mean, weighted, or energy")
```

add:

```python
    contrastive.add_argument("--contrastive-loss", help="Contrastive objective: infonce or vicreg")
    contrastive.add_argument("--vicreg-sim-weight", type=float, help="VICReg invariance weight")
    contrastive.add_argument("--vicreg-var-weight", type=float, help="VICReg variance weight")
    contrastive.add_argument("--vicreg-cov-weight", type=float, help="VICReg covariance weight")
```

- [ ] **Step 7: Map the CLI args into the config dict**

In `_config_from_args`, the contrastive tuple (lines 180-193) currently ends:

```python
                ("conformer_pool_mode", "conformer_pool_mode"),
                ("seed", "seed"),
            ):
```

Insert the four mappings before `("seed", "seed")`:

```python
                ("conformer_pool_mode", "conformer_pool_mode"),
                ("contrastive_loss", "contrastive_loss"),
                ("vicreg_sim_weight", "vicreg_sim_weight"),
                ("vicreg_var_weight", "vicreg_var_weight"),
                ("vicreg_cov_weight", "vicreg_cov_weight"),
                ("seed", "seed"),
            ):
```

- [ ] **Step 8: Pass the args into the trainer**

In `run_contrastive_pretrain`, the `contrastive_pretrain_on_dataset(...)` call (lines 456-474) currently has:

```python
        conformer_pool_mode=args.conformer_pool_mode,
        seed=args.seed,
    )
```

Replace with:

```python
        conformer_pool_mode=args.conformer_pool_mode,
        contrastive_loss=args.contrastive_loss,
        vicreg_sim_weight=args.vicreg_sim_weight,
        vicreg_var_weight=args.vicreg_var_weight,
        vicreg_cov_weight=args.vicreg_cov_weight,
        seed=args.seed,
    )
```

- [ ] **Step 9: Run the config tests and the CLI suite**

Run: `python -m pytest tests/test_config.py tests/test_cli.py -q`
Expected: PASS (the 5 new config tests + existing config/CLI tests). The namespace always carries the four fields because `config_to_namespace` reads them from the resolved (default-filled) config.

- [ ] **Step 10: Commit**

```bash
git add qchem_gnn/config.py qchem_gnn/cli.py tests/test_config.py
git commit -m "feat(contrastive): plumb contrastive_loss and VICReg weights through config and CLI"
```

---

### Task 4: Generalize the harness to N arms with pairwise verdicts

**Files:**
- Modify: `qchem_gnn/validation.py` (`_verdict` lines 100-116; `aggregate_results` lines 119-155; `render_report` lines 162-195; `run_validation` aggregate call lines 352-362)
- Test: `tests/test_validation_aggregate.py` (update existing assertions + add new tests), `tests/test_validation_end_to_end.py` (update one assertion)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `aggregate_results(extrinsic_rows, intrinsic_rows, arms=None, comparisons=None) -> dict` returning `{"arms": [...], "extrinsic": {method: {arm: {...}}}, "verdicts": [verdict, ...], "intrinsic": {arm: {prop: {...}}}}`. When `arms` is `None`, arms are derived from the rows (first-appearance order, falling back to `ARMS`). When `comparisons` is `None`, it defaults to `[{"name": "teacher_vs_baseline", "reference": "baseline", "treatment": "quantum"}]`.
  - Each verdict: `{"name": str, "reference": str, "treatment": str, "method": "mlp_head", "metric": "mae", "delta": float|None, "combined_std": float|None, "result": "helps"|"within noise"|"insufficient seeds"|"n/a"}`.
  - `render_report(aggregate) -> str` prints one verdict block per entry in `aggregate["verdicts"]` and uses `aggregate["arms"]` for table ordering.

- [ ] **Step 1: Write the new failing tests**

Append to `tests/test_validation_aggregate.py`:

```python
def test_multiple_comparisons_produce_multiple_verdicts():
    rows = [
        _ex("baseline", 0, "mlp_head", 1.30, 0.30),
        _ex("baseline", 1, "mlp_head", 1.32, 0.29),
        _ex("baseline", 2, "mlp_head", 1.28, 0.31),
        _ex("quantum", 0, "mlp_head", 1.00, 0.52),
        _ex("quantum", 1, "mlp_head", 1.01, 0.51),
        _ex("quantum", 2, "mlp_head", 0.99, 0.53),
        _ex("quantum_vicreg", 0, "mlp_head", 0.80, 0.62),
        _ex("quantum_vicreg", 1, "mlp_head", 0.81, 0.61),
        _ex("quantum_vicreg", 2, "mlp_head", 0.79, 0.63),
    ]
    comparisons = [
        {"name": "teacher_vs_baseline", "reference": "baseline", "treatment": "quantum"},
        {"name": "vicreg_vs_infonce", "reference": "quantum", "treatment": "quantum_vicreg"},
    ]
    agg = aggregate_results(rows, [], arms=["baseline", "quantum", "quantum_vicreg"], comparisons=comparisons)
    names = [v["name"] for v in agg["verdicts"]]
    assert names == ["teacher_vs_baseline", "vicreg_vs_infonce"]
    assert agg["verdicts"][0]["result"] == "helps"
    assert agg["verdicts"][1]["result"] == "helps"
    assert agg["verdicts"][1]["delta"] == pytest.approx(0.20, abs=1e-6)


def test_comparison_with_missing_arm_is_na():
    rows = [_ex("baseline", 0, "mlp_head", 1.2, 0.4), _ex("baseline", 1, "mlp_head", 1.2, 0.4)]
    comparisons = [{"name": "vicreg_vs_infonce", "reference": "quantum", "treatment": "quantum_vicreg"}]
    agg = aggregate_results(rows, [], arms=["baseline"], comparisons=comparisons)
    assert agg["verdicts"][0]["result"] == "n/a"


def test_default_comparison_is_single_teacher_vs_baseline():
    rows = [
        _ex("baseline", 0, "mlp_head", 1.20, 0.40),
        _ex("baseline", 1, "mlp_head", 1.22, 0.39),
        _ex("quantum", 0, "mlp_head", 1.00, 0.52),
        _ex("quantum", 1, "mlp_head", 1.01, 0.51),
    ]
    agg = aggregate_results(rows, [])
    assert len(agg["verdicts"]) == 1
    assert agg["verdicts"][0]["name"] == "teacher_vs_baseline"
```

`import pytest` is already at the top of this test file (used by existing tests).

- [ ] **Step 2: Update the existing aggregate assertions to the `verdicts` shape**

In `tests/test_validation_aggregate.py`, the existing tests read `agg["verdict"]`. Change each:

- In `test_mean_std_and_helps_verdict`: replace

```python
    assert agg["verdict"]["result"] == "helps"
    assert agg["verdict"]["delta"] > 0
```

with

```python
    assert agg["verdicts"][0]["result"] == "helps"
    assert agg["verdicts"][0]["delta"] > 0
```

- In `test_within_noise_verdict`: replace `assert agg["verdict"]["result"] == "within noise"` with `assert agg["verdicts"][0]["result"] == "within noise"`.
- In `test_insufficient_seeds_verdict`: replace `assert agg["verdict"]["result"] == "insufficient seeds"` with `assert agg["verdicts"][0]["result"] == "insufficient seeds"`.
- In `test_failed_rows_excluded_and_na_when_empty`: replace `assert agg["verdict"]["result"] == "n/a"` with `assert agg["verdicts"][0]["result"] == "n/a"`.

- [ ] **Step 3: Update the end-to-end assertion**

In `tests/test_validation_end_to_end.py`, replace:

```python
    assert "verdict" in aggregate
```

with:

```python
    assert "verdicts" in aggregate
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `python -m pytest tests/test_validation_aggregate.py -q`
Expected: FAIL — the new tests reference `agg["verdicts"]` which does not exist yet (current code returns `verdict`), and the updated assertions also reference `verdicts`.

- [ ] **Step 5: Rewrite `_verdict` for named pairwise comparisons**

In `qchem_gnn/validation.py`, replace `_verdict` (lines 100-116) with:

```python
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
```

- [ ] **Step 6: Rewrite `aggregate_results` for N arms and comparisons**

Replace `aggregate_results` (lines 119-155) with:

```python
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
        comparisons = _DEFAULT_COMPARISONS

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
```

- [ ] **Step 7: Rewrite `render_report` for multiple verdicts and config-driven arms**

Replace `render_report` (lines 162-195) with:

```python
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
```

- [ ] **Step 8: Pass arms and comparisons from `run_validation`**

In `run_validation`, the arm loop (lines 352-362) iterates `ARMS`. Replace:

```python
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
```

with:

```python
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
```

- [ ] **Step 9: Run the validation suite to verify it passes**

Run: `python -m pytest tests/test_validation_aggregate.py tests/test_validation_end_to_end.py tests/test_validation_cell.py -q`
Expected: PASS (updated + new aggregate tests, the end-to-end test, and the unchanged cell test).

- [ ] **Step 10: Commit**

```bash
git add qchem_gnn/validation.py tests/test_validation_aggregate.py tests/test_validation_end_to_end.py
git commit -m "feat(validation): N-arm harness with per-comparison pairwise verdicts"
```

---

### Task 5: Add the `quantum_vicreg` arm to the experiment and extend the integration test

**Files:**
- Modify: `configs/validate_quantum_teacher.yaml` (arms block + new comparisons block)
- Test: `tests/test_validation_end_to_end.py` (extend the existing end-to-end test)

**Interfaces:**
- Consumes: `run_validation` with `comparisons` (Task 4); the trainer's `contrastive_loss` override (Task 2) flowing through `_pretrain_kwargs`'s `kwargs.update(arm_overrides)`.
- Produces: a three-arm experiment config (`baseline`, `quantum`, `quantum_vicreg`) with a two-entry `comparisons` block; the end-to-end test asserts the third backbone and both verdicts.

- [ ] **Step 1: Extend the end-to-end test**

In `tests/test_validation_end_to_end.py`, the `cfg` dict in `test_run_validation_writes_report` has an `arms` block with `baseline` and `quantum`. Add the third arm and a `comparisons` key. Replace:

```python
        "arms": {"baseline": {"teacher_weight": 0.0, "conformer_pool_mode": "mean"},
                 "quantum": {"teacher_weight": 1.0, "conformer_pool_mode": "energy"}},
        "seeds": [0],
```

with:

```python
        "arms": {"baseline": {"teacher_weight": 0.0, "conformer_pool_mode": "mean"},
                 "quantum": {"teacher_weight": 1.0, "conformer_pool_mode": "energy"},
                 "quantum_vicreg": {"teacher_weight": 1.0, "conformer_pool_mode": "energy",
                                    "contrastive_loss": "vicreg"}},
        "comparisons": [
            {"name": "teacher_vs_baseline", "reference": "baseline", "treatment": "quantum"},
            {"name": "vicreg_vs_infonce", "reference": "quantum", "treatment": "quantum_vicreg"},
        ],
        "seeds": [0],
```

Then, after the existing assertions at the end of that test (which check `report.json`/`report.md` exist and both backbones), add:

```python
    assert (out_dir / "quantum_vicreg_s0.pt").exists()
    assert len(aggregate["verdicts"]) == 2
    assert {v["name"] for v in aggregate["verdicts"]} == {"teacher_vs_baseline", "vicreg_vs_infonce"}
    report_text = (out_dir / "report.md").read_text()
    assert "vicreg_vs_infonce" in report_text
```

- [ ] **Step 2: Run the test to confirm the third-arm path works end-to-end**

Run: `python -m pytest tests/test_validation_end_to_end.py -q`
Expected: PASS. This test drives `run_validation` with an inline three-arm cfg (not the YAML), so it exercises the code from Tasks 2 and 4 directly. It is a coverage extension, not a red-first step — if it FAILS, the failure pinpoints a real break in the VICReg arm's training or the multi-comparison aggregation, which must be fixed before proceeding. The YAML experiment artifact is added and verified separately in Steps 3-5.

- [ ] **Step 3: Add the arm and comparisons to the experiment config**

In `configs/validate_quantum_teacher.yaml`, replace the `arms:` block:

```yaml
arms:
  baseline: { teacher_weight: 0.0, conformer_pool_mode: mean }
  quantum:  { teacher_weight: 1.0, conformer_pool_mode: energy, energy_temperature: 298.15 }
```

with:

```yaml
arms:
  baseline:       { teacher_weight: 0.0, conformer_pool_mode: mean }
  quantum:        { teacher_weight: 1.0, conformer_pool_mode: energy, energy_temperature: 298.15 }
  quantum_vicreg: { teacher_weight: 1.0, conformer_pool_mode: energy, energy_temperature: 298.15,
                    contrastive_loss: vicreg }

comparisons:
  - { name: teacher_vs_baseline, reference: baseline, treatment: quantum }
  - { name: vicreg_vs_infonce,   reference: quantum,   treatment: quantum_vicreg }
```

- [ ] **Step 4: Run the end-to-end test to verify it passes**

Run: `python -m pytest tests/test_validation_end_to_end.py -q`
Expected: PASS (1 passed) — three backbones train on synthetic data, two verdicts render.

- [ ] **Step 5: Verify the experiment config parses and the run path is intact**

Run: `python -c "from qchem_gnn.validation import load_validation_config; c = load_validation_config('configs/validate_quantum_teacher.yaml'); print(list(c['arms'])); print([x['name'] for x in c['comparisons']])"`
Expected: prints `['baseline', 'quantum', 'quantum_vicreg']` then `['teacher_vs_baseline', 'vicreg_vs_infonce']`.

- [ ] **Step 6: Run the full suite for no regressions**

Run: `python -m pytest tests -q`
Expected: PASS (previous ~155 + the new VICReg/config/aggregate tests from Tasks 1-5).

- [ ] **Step 7: Commit**

```bash
git add configs/validate_quantum_teacher.yaml tests/test_validation_end_to_end.py
git commit -m "feat(validation): add quantum_vicreg arm and vicreg_vs_infonce comparison"
```

---

## Notes for the implementer

- The real experiment run is `python -m qchem_gnn.validation --config configs/validate_quantum_teacher.yaml` against the existing `runs/validate/` cache: it reuses the six cached `baseline_s*.pt`/`quantum_s*.pt` backbones (skip-if-exists) and trains only the three `quantum_vicreg_s*.pt` backbones plus their probes. Do NOT run this as part of the plan — the synthetic end-to-end test in Task 5 covers the code path.
- VICReg keeps the existing `hidden_dim → hidden_dim` projection heads on purpose. Do not widen them; a wide expander would break the matched ablation and is explicitly out of scope.
- The trainer does not validate the `contrastive_loss` string (anything other than `"vicreg"` falls through to InfoNCE) — string validation lives in `config.py`, by design.
- The harness change is shape-breaking (`verdict` → `verdicts`); Task 4 updates every existing reader (`test_validation_aggregate.py`, `test_validation_end_to_end.py`) so the suite is green at the end of Task 4. Backward compatibility is preserved at the *config* level: omitting `comparisons` yields the single legacy verdict.
- `_pretrain_kwargs` already does `kwargs.update(arm_overrides)`, so an arm declaring `contrastive_loss: vicreg` reaches the trainer with no harness wiring — that is why Task 4 needs no change to `_pretrain_kwargs`.
