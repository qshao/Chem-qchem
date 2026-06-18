# Pretraining Checkpoint / Resume Design

**Date:** 2026-06-18

**Status:** Approved (pending spec review)

## Goal

Add opt-in checkpoint/resume to the contrastive pretraining loop so that long
scaled runs (many shards × many epochs × multiple seeds) survive crashes,
preemption, or OOM and continue from the last saved epoch instead of restarting
from epoch 0.

## Motivation

The preprocess/train split is already correct: `preprocess_shard` is the only
path that runs the heavy extractor, and training reads compact caches read-only
via `load_compact_shards` (it never regenerates a shard). The gap is purely in
the pretraining *loop*:

- `contrastive_pretrain_on_dataset` always builds fresh models + optimizer and
  loops `for _ in range(epochs)` from scratch.
- The only existing recovery is coarse: `run_one_cell` skips a seed entirely
  when its final backbone **and** intrinsic JSON already exist (cell-skip). A
  seed that dies at epoch 150 of 200 restarts at 0.

The function already returns `optimizer_state_dict`, `epoch`, and `global_step`
— the scaffolding for resume exists but was never wired up.

## Scope

In scope:
- Periodic checkpointing of full pretraining state during the epoch loop.
- Faithful (bit-exact) resume from the latest checkpoint.
- Config-level opt-in and cadence control.
- Strict fingerprint validation that refuses to resume into an inconsistent run.

Out of scope (YAGNI):
- Multi-checkpoint history / rotation. A single rolling file per arm/seed.
- Checkpointing the downstream adapter probes (they are fast).
- Changing the existing cell-skip logic (it already handles completed seeds).
- Warm-start-from-arbitrary-backbone as a separate feature.

## Design

### Where the logic lives

Entirely in `contrastive_pretrain_on_dataset` (`qchem_gnn/contrastive_pretrain.py`)
plus thin wiring in `run_one_cell` (`qchem_gnn/validation.py`). No new module.

### New parameters on `contrastive_pretrain_on_dataset`

```python
checkpoint_path: Path | None = None   # rolling checkpoint file; None disables
checkpoint_every: int = 10            # write every N epochs
resume: bool = False                  # attempt to load checkpoint_path at start
```

All three default to today's behavior (no checkpointing, no resume), so existing
callers are unaffected.

### Checkpoint payload

A single rolling file per arm/seed, written **atomically** (`torch.save` to a
`.tmp` sibling, then `os.replace` onto the final path) so a crash mid-write
cannot corrupt it.

Contents:

| Key | Purpose |
|---|---|
| `version` | format integer (`PRETRAIN_CHECKPOINT_VERSION = 1`) |
| `epoch` | number of completed epochs |
| `model_state_dict` | `MolecularQuantumGNN` weights |
| `encoder3d_state_dict` | `Conformer3DEncoder` weights |
| `teacher_state_dict` | `QuantumTeacherHeads` weights |
| `proj_2d_state_dict` | 2D projection head weights |
| `proj_3d_state_dict` | 3D projection head weights |
| `optimizer_state_dict` | Adam moments over all of the above |
| `rng_state` | `torch.get_rng_state()` captured *after* the completed epochs |
| `loss_history` | per-epoch total loss so far |
| `contrastive_loss_history` | per-epoch contrastive loss so far |
| `config_fingerprint` | structural + hyperparameter fields (see below) |

`target_normalization` is recomputed deterministically from the dataset on every
call, so it is not stored.

### Control flow

```
build models + optimizer (as today)

if resume and checkpoint_path exists:
    payload = load(checkpoint_path)         # raises on corrupt file
    validate_fingerprint(payload, current)  # raises CheckpointMismatchError on mismatch
    load_state_dict for all 5 modules + optimizer
    torch.set_rng_state(payload.rng_state)  # AFTER construction
    loss_history, contrastive_loss_history = payload histories
    start_epoch = payload.epoch
elif resume and checkpoint_path missing:
    warn("no checkpoint to resume; starting fresh")
    torch.manual_seed(seed); start_epoch = 0
else:
    torch.manual_seed(seed); start_epoch = 0

if start_epoch >= epochs:
    # already complete (e.g. re-run after finish) — idempotent
    skip loop

for epoch in range(start_epoch, epochs):
    ... train one epoch ...
    append to loss histories
    if checkpoint_path and ((epoch + 1) % checkpoint_every == 0 or epoch + 1 == epochs):
        write_checkpoint(epoch + 1)   # atomic

final embedding pass + return result (as today)
```

The RNG ordering matters: `set_rng_state` runs strictly **after** model
construction, so the init-time RNG draws are overwritten by `load_state_dict`
and the stream is reset to its post-epoch position. This makes a resumed run
bit-exactly equal to an uninterrupted run.

### Fingerprint validation

On resume, compare the stored `config_fingerprint` against the current call's
parameters. Mismatch on any field raises `CheckpointMismatchError` listing the
differing fields.

Fingerprint fields:

- `hidden_dim`, `num_message_passing_steps`
- `hidden_dim_3d`, `num_rbf`, `cutoff`, `num_message_passing_steps_3d`
- `batch_size`, `learning_rate`
- `supervised_weight`, `contrastive_weight`, `teacher_weight`
- `temperature`, `energy_temperature`, `conformer_pool_mode`
- `contrastive_loss`, `vicreg_sim_weight`, `vicreg_var_weight`, `vicreg_cov_weight`
- `use_scaffold_negmask`
- `seed`
- `node_targets` (model output width)
- `num_examples` (catches a changed shard set)

`epochs` is **deliberately excluded** — raising it is the allowed "extend
training" case (resume from epoch 200 to continue to 300).

### Wiring in `run_one_cell`

- Checkpoint path: `out_dir / f"{arm}_s{seed}.ckpt.pt"` — distinct from the
  final backbone `out_dir / f"{arm}_s{seed}.pt"`.
- Reads `resume` and `checkpoint_every` from the `pretrain:` config block and
  passes them through.
- The existing **cell-skip is unchanged**: a fully-done seed (backbone **and**
  intrinsic present, no `--overwrite`) still short-circuits before any model is
  built. Resume only engages when the final backbone does not yet exist but a
  `.ckpt.pt` does.
- **`--overwrite` forces `resume=False`** and clobbers any partial checkpoint —
  a clean restart from epoch 0.
- After a cell completes (final backbone **and** intrinsic both written), the
  now-redundant `.ckpt.pt` is deleted. If only the backbone was written
  (intrinsic eval threw), the checkpoint is left in place so a re-run can resume.
- `CheckpointMismatchError` is **not** caught by the per-cell `try/except`
  (which otherwise records a "failed cell" and continues). A config mismatch
  affects every cell, so it propagates and stops the whole run loudly.

### Config surface

In the `pretrain:` block of the validation YAML (both optional; defaults
preserve current behavior):

```yaml
pretrain:
  resume: true          # default false
  checkpoint_every: 10  # default 10
```

## Error Handling

| Situation | Behavior |
|---|---|
| `resume=true`, checkpoint missing | Warn, start fresh (epoch 0). Not an error. |
| `resume=true`, checkpoint corrupt/unreadable | Raise, with a message suggesting `--overwrite` to restart. |
| `resume=true`, fingerprint mismatch | Raise `CheckpointMismatchError` naming the differing fields. Propagates past per-cell isolation. |
| `start_epoch >= epochs` | Skip the loop, return a valid result (idempotent re-run). |
| Crash mid-checkpoint-write | Atomic temp+rename leaves the previous valid checkpoint intact. |

## Testing

New file `tests/test_pretrain_checkpoint.py`, using a tiny 2-molecule dataset
and short (2–4 epoch) runs:

1. **Roundtrip** — pretrain 4 epochs with `checkpoint_every=2`; assert the
   `.ckpt.pt` exists and reports `epoch == 4`.
2. **Resume continuity (key correctness test)** — run 2 epochs → checkpoint,
   then resume with `epochs=4`; assert the final embeddings are numerically
   close to a single uninterrupted 4-epoch run. Proves RNG + optimizer
   restoration is faithful.
3. **Fingerprint mismatch raises** — write a checkpoint at `hidden_dim=16`,
   resume at `hidden_dim=32` → `CheckpointMismatchError` naming `hidden_dim`.
4. **Epochs change allowed** — checkpoint at epoch 2, resume with `epochs=4` →
   continues to 4 with no error.
5. **Resume past completion** — checkpoint with `epoch == epochs`, resume → no
   training runs, returns a valid result immediately.
6. **Overwrite ignores checkpoint** (validation level) — `overwrite=True` starts
   fresh and clobbers an existing `.ckpt.pt`.

## Files Touched

- `qchem_gnn/contrastive_pretrain.py` — new params, checkpoint I/O, fingerprint
  validation, `CheckpointMismatchError`, `PRETRAIN_CHECKPOINT_VERSION`.
- `qchem_gnn/validation.py` — wire checkpoint path + config keys into
  `run_one_cell`; let `CheckpointMismatchError` propagate; cleanup logic.
- `configs/validate_scaled.yaml` — document `resume` / `checkpoint_every`.
- `tests/test_pretrain_checkpoint.py` — new tests.
- `docs/tutorials/scaled_pretraining.md` — document resume usage.
