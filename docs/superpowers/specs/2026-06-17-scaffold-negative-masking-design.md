# Scaffold-Aware Negative Masking — Design

**Date:** 2026-06-17
**Status:** Approved design, pending implementation plan

## Goal

The InfoNCE contrastive loss treats every non-paired molecule in a batch as a
negative. With ~276 pretrain molecules drawn from a single ZINC shard, many
share Murcko scaffolds — they have the same ring system and core structure —
so they should be geometrically close in embedding space, not pushed apart.
These false negatives cap the quality of the learned 2D backbone.

This design adds **scaffold-aware negative masking**: scaffold-similar pairs are
excluded from the InfoNCE denominator at train time. The mask is computed once
from Murcko scaffold identities before training begins and sliced per batch with
no per-step chemistry computation. The change is measured using the existing
matched-ablation harness: a new `quantum_scaffold` arm differs from `quantum` in
exactly `use_scaffold_negmask: true`.

## Scope and decisions

- **Similarity criterion: Murcko scaffold exact match.** Two molecules sharing
  the same Murcko scaffold SMILES are excluded from each other's negative set.
  Binary, no threshold hyperparameter, scientifically motivated. Tanimoto
  threshold is a follow-up if scaffold match alone proves insufficient.
- **Masking is one-sided exclusion, not reweighting.** Masked logits are set to
  `-inf` (removed from the softmax denominator), not down-weighted. This is the
  standard approach in supervised contrastive literature.
- **Diagonal (the positive pair) is never masked** regardless of scaffold.
- **Experiment shape:** add one new arm (`quantum_scaffold`) to the existing
  three-arm study; only three new backbones train. `baseline`, `quantum`, and
  `quantum_vicreg` backbones are reused from cache.
- **Matched ablation:** `quantum_scaffold` vs `quantum` differs in exactly
  `use_scaffold_negmask`. Projection heads, teacher, temperature, batch size,
  seeds, everything else identical.

## Architecture

Four focused changes.

1. **`qchem_gnn/eval.py`** — new public function
   `build_scaffold_negative_mask(examples)` returning an `[N, N]` bool tensor.
2. **`qchem_gnn/losses.py`** — `info_nce_contrastive_loss` gains an optional
   `negative_mask` parameter; masked off-diagonal logits set to `-inf`.
3. **`qchem_gnn/contrastive_pretrain.py`** — `contrastive_pretrain_on_dataset`
   gains `use_scaffold_negmask: bool = False`; computes the full mask once
   before the training loop, slices per batch.
4. **`qchem_gnn/config.py` + `qchem_gnn/cli.py`** — `use_scaffold_negmask`
   bool plumbed through config defaults, validation, namespace, and CLI.

Reused unchanged: the 3D encoder, the teacher heads, Boltzmann pooling, the
adapt subsystem, the cell runner, the report renderer, and the VICReg arm.

### Component 1 — `build_scaffold_negative_mask`

```python
def build_scaffold_negative_mask(examples) -> torch.Tensor:
```

Input: list of `MinimalQuantumExample` (the pretrain set, length N).

- Calls `_infer_scaffolds([str(i) for i in range(N)], [ex.smiles for ex in examples])`,
  which maps each string index to its Murcko scaffold SMILES (falling back to
  raw SMILES if RDKit cannot parse the molecule — these get a unique scaffold
  key and are never masked). Both `_infer_scaffolds` and
  `build_scaffold_negative_mask` live in `eval.py`.
- Groups indices by scaffold SMILES.
- Fills a `[N, N]` bool tensor: `True` at `[i, j]` iff `i ≠ j` and
  `scaffold[i] == scaffold[j]`. Diagonal is always `False`.
- Returns a CPU bool tensor; it is moved to the correct device at slice time.

Edge case: if a molecule's entire batch consists of scaffold mates, every
off-diagonal entry in its row is `True`. The cross-entropy loss for that row is
trivially ≈ 0 (only the positive remains). The trainer emits a
`warnings.warn` once per occurrence and continues.

### Component 2 — `info_nce_contrastive_loss` with `negative_mask`

```python
def info_nce_contrastive_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.1,
    negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
```

`negative_mask` is a `[batch, batch]` bool tensor. `True` at `[i, j]` means
pair `(i, j)` should not contribute as a negative.

```python
if negative_mask is not None:
    eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(negative_mask & ~eye, float('-inf'))
```

Both `loss_ab` and `loss_ba` use the same masked `logits`. Default `None`
preserves existing byte-for-byte behaviour.

### Component 3 — Trainer plumbing

`contrastive_pretrain_on_dataset` gains `use_scaffold_negmask: bool = False`.

Before the epoch loop:

```python
full_mask = None
if use_scaffold_negmask:
    full_mask = build_scaffold_negative_mask(examples)
```

Inside the batch loop, after `coords_index` is known:

```python
batch_mask = None
if full_mask is not None:
    global_idx = [batch_indices[p] for p in coords_index]
    batch_mask = full_mask[global_idx][:, global_idx].to(device)
```

Then pass `negative_mask=batch_mask` to `info_nce_contrastive_loss` (and
`negative_mask=None` to `vicreg_loss`, which doesn't use it).

### Component 4 — Config and CLI plumbing

`config.py`: add `use_scaffold_negmask` to `VALID_SECTION_KEYS["contrastive"]`,
`DEFAULT_CONFIG["contrastive"]` (default `False`), and `_validate_config`
(must be `bool`; raises `ConfigError` for non-bool values).
`config_to_namespace` exposes it as `ns.use_scaffold_negmask`.

`cli.py`: `--use-scaffold-negmask` store-true flag in the contrastive argument
group; mapped through `_config_from_args`; passed to `run_contrastive_pretrain`.

In the validation harness, `use_scaffold_negmask: true` in an arm's override
dict reaches the trainer through the existing `kwargs.update(arm_overrides)`
mechanism in `_pretrain_kwargs` — no harness wiring needed.

## Experiment structure

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

Only `quantum_scaffold_s{0,1,2}.pt` are new; all other backbones are cached.
The report renders four extrinsic/intrinsic rows per method and three verdict
blocks.

## Data flow

```
pretrain_ds.examples  (N ~ 276 molecules after holdout)
   │  build_scaffold_negative_mask(examples)  [once before epoch loop]
   ▼
full_mask: [N, N] bool tensor  (True where same scaffold, i≠j)
   │  per batch: global_idx = [batch_indices[p] for p in coords_index]
   ▼
batch_mask: [usable, usable] bool tensor  (sliced + moved to device)
   │  info_nce_contrastive_loss(view_2d, view_3d, temperature, negative_mask=batch_mask)
   ▼
masked logits: scaffold-similar off-diagonal entries set to -inf
   ▼
symmetric cross-entropy loss
```

## Error handling

- **Unparseable SMILES:** `_infer_scaffolds` falls back to raw SMILES as the
  scaffold key — molecule is never masked, no crash.
- **All negatives masked in a row:** `warnings.warn` once per occurrence;
  training continues with trivial loss for that molecule that batch.
- **Invalid `use_scaffold_negmask` config value:** `ConfigError` at load time
  (not at training time).
- All existing per-cell isolation, non-finite backbone guard, and skip-if-exists
  caching inherited unchanged.

## Testing

TDD, reusing `tests/_validation_fixtures.py`.

- **Unit — `build_scaffold_negative_mask`:** toluene (`Cc1ccccc1`) and aniline
  (`Nc1ccccc1`) both reduce to benzene under Murcko → `True` at their
  cross-entries; ethanol (`CCO`) has a unique scaffold → `False` with both.
  Diagonal always `False`.
- **Unit — masked InfoNCE:** identical views with a full off-diagonal mask
  (batch=4, all pairs masked) → loss is 0.0 (only positive remains); partial
  mask raises no error; `negative_mask=None` is byte-for-byte identical to the
  existing unmasked path.
- **Unit — trainer switch:** `contrastive_pretrain_on_dataset(...,
  use_scaffold_negmask=True)` runs end-to-end on the tiny fixture (all different
  scaffolds → mask all-`False`) and returns a finite loss history. The
  `use_scaffold_negmask=False` default path is a regression.
- **Unit — config:** `use_scaffold_negmask: true` parses and validates; a
  string value raises `ConfigError`; default is `False`.
- **Integration:** end-to-end test gains a `quantum_scaffold` arm and
  `scaffold_vs_infonce` comparison; asserts `quantum_scaffold_s0.pt` exists and
  three verdict blocks render.
- **Regression:** full ~170-test suite passes; the unmasked InfoNCE path and
  existing arms are untouched.

## Risks

1. **Scaffold coverage at batch=16.** If fewer than 2 molecules per batch share
   a scaffold (common at batch=16 with 276 molecules), masking has no effect
   on most batches. The improvement signal may be small but should be
   directionally positive.
2. **All-masked rows.** Rare but possible at small batch with scaffold-heavy
   data. Mitigated by the warning and by the fact that the positive is always
   preserved (loss ≈ 0 is not a crash).
3. **n = 3 seeds remains a heuristic.** Unchanged from the harness.

## Out of scope

- Tanimoto threshold masking (follow-up if scaffold match shows no signal).
- Multi-conformer positives, MoCo queue — other brainstormed directions.
- Any change to the VICReg arm, the inference contract, or checkpoint format.
