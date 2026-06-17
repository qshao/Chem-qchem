# VICReg Contrastive Objective — Design

**Date:** 2026-06-17
**Status:** Approved design, pending implementation plan

## Goal

The current contrastive pretraining aligns a molecule's 2D graph embedding to
its Boltzmann-pooled 3D conformer embedding with a symmetric **InfoNCE** loss
using in-batch negatives. At `batch_size: 16` that is only 15 negatives per
anchor, and every other molecule in the batch — including scaffold-similar
ones that should be close — is treated as a negative. Both facts cap the
quality of the learned representation.

This design adds a **negative-free VICReg objective** as a drop-in alternative
to InfoNCE and measures whether it improves the 2D backbone, using the matched
ablation harness already built (`qchem_gnn/validation.py`). VICReg sidesteps
both structural weaknesses at once: with no negatives there are no false
negatives, and no dependence on batch size for negative count. Its variance
hinge is an explicit anti-collapse guarantee, which is the safest property to
rely on at this batch size.

## Scope and decisions

Settled during brainstorming:

- **Objective:** VICReg (not Barlow Twins). The variance hinge is a hard
  anti-collapse term rather than a statistical side-effect, and VICReg needs
  no batch normalization — both matter at `batch_size: 16`.
- **Positive pairing unchanged:** the two views remain the projected 2D
  molecule embedding and the projected Boltzmann-pooled 3D embedding. We swap
  the *loss*, not the views.
- **Matched ablation:** the InfoNCE arm and the VICReg arm differ in **exactly**
  the contrastive loss function. Projection heads, pairing, teacher, supervised
  loss, dims, epochs, learning rate — all identical.
- **Projection heads stay `hidden_dim → hidden_dim`.** No VICReg-style wide
  expander, precisely to keep the ablation matched. Wider expanders are a
  follow-up tunable, not part of this isolation.
- **Experiment shape:** add one new arm (`quantum_vicreg`) to the existing
  two-arm study and reuse the cached `baseline` and `quantum` backbones; only
  three new backbones train.

## Architecture

Four focused changes; everything else reused.

1. **`qchem_gnn/losses.py`** — a new pure function `vicreg_loss`, sibling to the
   existing `info_nce_contrastive_loss`, with the same shape contract.
2. **`qchem_gnn/contrastive_pretrain.py`** — a `contrastive_loss` switch
   selecting InfoNCE or VICReg at the single call site; nothing else moves.
3. **`qchem_gnn/config.py` + `qchem_gnn/cli.py`** — plumb `contrastive_loss`
   and the three VICReg weights through config defaults, validation, and CLI.
4. **`qchem_gnn/validation.py`** — generalize the harness from two hardcoded
   arms to N config-driven arms, and from one hardcoded verdict to a list of
   pairwise comparisons. Backward-compatible: absent `comparisons`, behavior is
   identical to today.

Reused unchanged: the 3D encoder, the teacher heads, Boltzmann pooling, the
adapt subsystem, the cell runner's caching/isolation, and the report renderer's
structure (it gains a verdict block per comparison).

### Component 1 — `vicreg_loss`

```python
def vicreg_loss(z_a, z_b, *, sim_weight=25.0, var_weight=25.0,
                cov_weight=1.0, gamma=1.0, eps=1e-4) -> torch.Tensor
```

`z_a`, `z_b` are `[batch, D]` projected views of the same molecules.

- **Invariance:** `inv = mse(z_a, z_b)` — pulls the two views together.
- **Variance:** `std = sqrt(var(dim=0) + eps)` per dimension for each view;
  `var = relu(gamma - std_a).mean() + relu(gamma - std_b).mean()`. The hinge
  keeps every embedding dimension's batch std above `gamma`, preventing
  collapse to a constant.
- **Covariance:** for each view, the off-diagonal of the `D×D` empirical
  covariance, squared, summed, divided by `D`; decorrelates dimensions.
- `L = sim_weight*inv + var_weight*var + cov_weight*cov`.

Contract matches `info_nce_contrastive_loss`: same input shapes, and `batch < 2`
raises `ValueError` (the variance/covariance statistics are undefined for a
single sample). Defaults are the canonical VICReg values (25/25/1, `gamma=1`).

### Component 2 — Trainer switch

`contrastive_pretrain_on_dataset` gains `contrastive_loss: str = "infonce"` plus
`vicreg_sim_weight`, `vicreg_var_weight`, `vicreg_cov_weight` (defaulting to
25/25/1). The contrastive block is unchanged up to the projections; only the
final call branches:

```python
view_2d = proj_2d(molecule_2d)
view_3d = proj_3d(molecule_3d)
if contrastive_loss == "vicreg":
    contrastive = vicreg_loss(view_2d, view_3d,
                              sim_weight=vicreg_sim_weight,
                              var_weight=vicreg_var_weight,
                              cov_weight=vicreg_cov_weight)
else:
    contrastive = info_nce_contrastive_loss(view_2d, view_3d, temperature=temperature)
```

The `contrastive_weight` multiplier, the `usable >= 2` guard, the teacher term,
and the supervised term are all untouched. InfoNCE remains the default, so every
existing caller is unaffected.

### Component 3 — Config and CLI plumbing

`config.py`: add to the `contrastive` section's `VALID_SECTION_KEYS`,
`DEFAULT_CONFIG`, and validation:

- `contrastive_loss` — string, must be `"infonce"` or `"vicreg"`; default
  `"infonce"`.
- `vicreg_sim_weight`, `vicreg_var_weight`, `vicreg_cov_weight` — non-negative
  floats; defaults 25.0, 25.0, 1.0.

These join the `config_to_namespace` values dict. `cli.py` adds the matching
generate-config handler entries and passes the four values into
`run_contrastive_pretrain`.

In the validation harness these reach the trainer through the existing
per-arm override mechanism — `_pretrain_kwargs` already does
`kwargs.update(arm_overrides)`, so an arm declaring `contrastive_loss: vicreg`
lands in the trainer with no new wiring.

### Component 4 — Harness generalization (N arms, pairwise verdicts)

Today `validation.py` hardcodes `ARMS = ("baseline", "quantum")` and a single
baseline-vs-quantum verdict. Two changes make it general:

- **Arms are config-driven.** `run_validation` iterates `cfg["arms"]` (insertion
  order); `aggregate_results` and `render_report` take the arm list rather than
  the module constant.
- **`verdict` (singular) becomes `verdicts` (a list).** The config gains an
  optional `comparisons` block:

  ```yaml
  comparisons:
    - { name: teacher_vs_baseline, reference: baseline, treatment: quantum }
    - { name: vicreg_vs_infonce,   reference: quantum,   treatment: quantum_vicreg }
  ```

  Each comparison reuses the **same heuristic** unchanged — on frozen
  `mlp_head` MAE, *helps* iff `mean_reference − mean_treatment >
  sqrt(std_ref² + std_treat²)` — parameterized by the two arm names. The same
  small-n rules apply per comparison: `n < 2` successful seeds in either arm →
  `insufficient seeds`; `n == 0` → `n/a`. The report prints one verdict block
  per comparison.

**Backward compatibility:** if `comparisons` is absent, it defaults to
`[{name: teacher_vs_baseline, reference: baseline, treatment: quantum}]`. The
existing config, the five aggregation tests, and the report layout all behave
exactly as before. Cached `baseline`/`quantum` probe JSONs stay valid — the
aggregation reads only arm names and metric values from rows.

## Data flow

```
configs/validate_quantum_teacher.yaml  (3 arms, 2 comparisons)
   │  run_validation
   ▼
load shard 44 → split holdout once (seed 1234)
   │  for arm in {baseline, quantum, quantum_vicreg}: for seed in {0,1,2}:
   ▼
  run_one_cell(arm, overrides, seed, ...)
   │   baseline_s*.pt, quantum_s*.pt  ── cached, skip-if-exists (reused)
   │   quantum_vicreg_s*.pt           ── NEW: trained with contrastive_loss=vicreg
   ▼
  extrinsic rows (mlp_head, finetune, engine MAE/R²) + intrinsic rows
   ▼
  aggregate_results(rows, arms, comparisons)
   │   verdict[teacher_vs_baseline]  = baseline vs quantum
   │   verdict[vicreg_vs_infonce]    = quantum vs quantum_vicreg
   ▼
  runs/validate/report.{json,md}   (two verdict blocks)
```

## Error handling

- `vicreg_loss` raises `ValueError` on `batch < 2` and on mismatched shapes,
  matching `info_nce_contrastive_loss`.
- Invalid `contrastive_loss` values are rejected at config validation, not at
  training time.
- A comparison naming an arm absent from the results yields a `n/a` verdict
  rather than raising — a missing arm should not abort the whole report.
- All existing per-cell isolation, the non-finite backbone guard, and
  skip-if-exists caching are inherited unchanged.

## Testing

TDD, reusing `tests/_validation_fixtures.py` and the synthetic HDF5 fixtures.

- **Unit — `vicreg_loss`:** identical views → invariance term ≈ 0; a collapsed
  (constant) embedding → large variance penalty vs a varied one; correlated
  dimensions → non-zero covariance term; `batch < 2` raises.
- **Unit — trainer switch:** `contrastive_pretrain_on_dataset(...,
  contrastive_loss="vicreg")` runs end-to-end on the tiny fixture and returns a
  finite loss history; the default InfoNCE path is byte-for-byte unchanged
  (regression).
- **Unit — multi-comparison aggregation:** three-arm rows produce two verdicts;
  `vicreg_vs_infonce` fires `helps`/`within noise` correctly; omitting
  `comparisons` yields exactly the single legacy verdict.
- **Unit — config:** `contrastive_loss: vicreg` parses and validates; an invalid
  value raises; defaults preserve `infonce` and 25/25/1.
- **Integration:** the end-to-end test gains a third arm and a comparison,
  asserts a `quantum_vicreg_s0.pt` backbone exists and two verdict blocks render.
- **Regression:** the full ~155-test suite passes; the InfoNCE-only path and the
  two-arm report are untouched when `comparisons` is omitted.

## Risks

1. **VICReg weights are dataset-sensitive.** The canonical 25/25/1 may not be
   optimal for ~325 molecules at `hidden_dim 64`. Mitigation: the weights are
   configurable; if the first VICReg arm underperforms, weight tuning is the
   first follow-up — but it is out of scope for this iteration, which answers
   the binary "does VICReg beat InfoNCE off the shelf."
2. **Covariance estimate is noisy at small batch.** With `batch_size 16` and the
   `usable >= 2` filter, the `D×D` covariance is crudely estimated. This is an
   inherent VICReg property; the variance hinge still guarantees no collapse.
3. **n = 3 seeds remains a heuristic, not a significance test.** Unchanged from
   the existing harness; the report keeps the disclaimer line.

## Out of scope

- Wider VICReg expander projection heads (kept matched at `hidden_dim` here).
- Conformer multi-view positives, Barlow Twins, MoCo queues, scaffold-aware
  negative masking — the other brainstormed directions, each gated on this
  result.
- Tuning the VICReg weights beyond their canonical defaults.
- Any change to the inference contract, the checkpoint format, or model logic
  beyond the single contrastive call site.
