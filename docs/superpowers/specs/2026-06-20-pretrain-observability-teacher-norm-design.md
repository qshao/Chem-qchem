# Pretraining Observability + Teacher-Loss Normalization — Design

**Date:** 2026-06-20
**Status:** Approved (pending spec review)

## Goal

Make contrastive-pretraining loss interpretable and debuggable, and fix the
teacher loss term that currently dominates the total loss by ~6 orders of
magnitude because it trains on raw, un-normalized quantum targets.

## Background

`contrastive_pretrain_on_dataset` (qchem_gnn/contrastive_pretrain.py) combines
three loss terms:

- **supervised** — `_supervised_loss_for_batch`, which normalizes node/edge/graph
  targets via `normalize_targets(...)` using stats from
  `compute_target_normalization(dataset)`.
- **contrastive** — InfoNCE or VICReg on projected 2D/3D embeddings.
- **teacher** — `teacher_loss(...)` (qchem_gnn/teacher_heads.py:41), plain
  `F.mse_loss` on **raw** per-conformer quantum targets (energy in Hartree,
  polarizability, chelpg, wbi).

Because the teacher targets are un-normalized, squared errors on raw energies
(~hundreds–thousands of Hartree) produce per-batch losses ~1e17, swamping the
other two terms and making the printed loss uninterpretable. A 1000-step demo
showed train_loss alternating between ~7e17 and ~7e11 every other log line
while the (stable, normalized) val proxy barely moved.

Today the only output is a single `print()` of the summed loss per `log_every`
window — no per-term breakdown, no persisted log. The dominance bug is therefore
invisible.

## Decisions (locked during brainstorming)

1. **Scope:** tight bundle — per-term logging + structured metrics file +
   teacher-target normalization fix. NaN guard, gradient clipping, LR scheduler,
   and CI are explicitly out of scope.
2. **Teacher normalization:** reuse the existing supervised stats from
   `compute_target_normalization(dataset)`. The teacher and supervised targets
   are the same physical properties, so a separate per-conformer stats path is
   unnecessary.
3. **Eval units:** de-normalize teacher predictions inside `evaluate_teacher`
   so reported intrinsic MAE stays in physical units and comparable to past
   reports.

## Components

### 1. Teacher-loss normalization

**File:** `qchem_gnn/contrastive_pretrain.py`

In the inner `_batch_forward`, after `assemble_conformer_targets(usable_examples)`
produces `node_t, edge_t, graph_t`, normalize them with the already-computed
`normalization` stats via `normalize_targets(node_t, edge_t, graph_t, normalization)`
before calling `teacher_loss(...)`. The teacher head now trains in normalized
space, on the same scale as the supervised term.

No change to `teacher_loss` itself — it stays a pure 3-term MSE; it simply
receives normalized targets and (since the head is unconstrained) learns to
emit normalized values.

### 2. Intrinsic-eval de-normalization

**File:** `qchem_gnn/validation.py`

`evaluate_teacher(teacher, encoder3d, holdout_examples)` gains a required
`normalization` parameter:

```python
def evaluate_teacher(teacher, encoder3d, holdout_examples, normalization) -> dict:
```

After the forward pass produces `node_pred, edge_pred, graph_pred` (in
normalized space), invert the stats before computing `_pearson_mae`:

- `node_pred = node_pred * normalization["node_std"] + normalization["node_mean"]`
- `edge_pred = edge_pred * normalization["edge_std"] + normalization["edge_mean"]`
- `graph_pred = graph_pred * normalization["graph_std"].squeeze(0) + normalization["graph_mean"].squeeze(0)`

(matching the broadcasting in `normalize_targets`).

`run_one_cell` passes `result.target_normalization` into the
`evaluate_teacher(...)` call.

Pearson r is scale-invariant and therefore unchanged; MAE returns to physical
units.

### 3. Per-term loss reporting

**File:** `qchem_gnn/contrastive_pretrain.py`

`_batch_forward` currently returns `(total, contrastive)`. Change it to return a
dict of detached float-able scalars:

```python
{"total": total, "supervised": supervised, "contrastive": contrastive, "teacher": teacher_term}
```

`total` remains the autograd tensor used for `.backward()`; the other entries are
used only for reporting (the loop calls `float(...)` on them). The training loop
accumulates each of the four terms over the `log_every` window and averages.

### 4. Structured metrics file

**File:** `qchem_gnn/contrastive_pretrain.py`

`contrastive_pretrain_on_dataset` gains `metrics_path: Path | None = None`. When
set, at each `log_every` boundary (and at the final step) it **appends one JSON
line** and flushes:

```json
{"step": 50, "total_steps": 1000, "train_loss": 1.83, "train_supervised": 0.42,
 "train_contrastive": 0.91, "train_teacher": 0.50, "val_loss": 1.79,
 "val_supervised": 0.40, "val_contrastive": 0.95, "val_teacher": 0.44,
 "steps_per_sec": 12.4, "wall_seconds": 4.03}
```

- `train_*` are the window-averaged terms. `val_*` are computed over the full
  `val_dataset` (when provided) at the same boundary, reusing `_batch_forward`
  under `eval()`/`no_grad()`; omitted (keys absent) when no `val_dataset`.
- `steps_per_sec` and `wall_seconds` are derived from a `time.perf_counter()`
  reading captured at training start and at each log boundary.
- Append mode (`"a"`), one `json.dumps(...)` + `"\n"` per line, flushed after
  each write so the file survives a crash mid-run.
- The parent directory is created if absent.

**Wiring:**

- `run_one_cell` (validation.py) auto-sets
  `metrics_path = out_dir / f"{arm_name}_s{seed}.metrics.jsonl"` — on by default
  in the harness.
- CLI `run_contrastive_pretrain` (cli.py) derives it from `--output`:
  `Path(args.output).with_suffix(".metrics.jsonl")`.

On resume, the file is **appended to**, not truncated (consistent with the
rolling-checkpoint resume model). A fresh run with `overwrite` in the harness
deletes the stale metrics file alongside the stale checkpoint.

### 5. Console line

The human-readable `print(...)` in the training loop gains the breakdown:

```
step     50/1000 | loss 1.83 (sup 0.42 / con 0.91 / tea 0.50) | val 1.79
```

The `| val X.XX` portion is omitted when `val_dataset` is None (matching current
behavior).

## Data flow

```
dataset ──► compute_target_normalization ──► normalization stats
                                                  │
                          ┌───────────────────────┼───────────────────────┐
                          ▼                        ▼                       ▼
                 supervised target          teacher target          evaluate_teacher
                 normalization              normalization            (de-normalize
                 (existing)                 (NEW, sec 1)              predictions, sec 2)
                          │                        │
                          └──────────┬─────────────┘
                                     ▼
                        _batch_forward returns
                        {total, supervised, contrastive, teacher}  (sec 3)
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                                              ▼
      console line (sec 5)                          metrics.jsonl (sec 4)
```

## Error handling

- `metrics_path` write failures should not crash training: wrap the append in a
  narrow `try/except OSError` that warns once and continues (training progress
  matters more than the log). The console line still prints.
- `evaluate_teacher` already runs inside `run_one_cell`'s `try/except` that marks
  the intrinsic row `failed`; the new de-normalization adds no new failure mode
  (pure tensor arithmetic on tensors already in scope).

## Testing

**File:** `tests/test_pretrain_observability.py` (new), plus edits to existing
tests that consumed the old `_batch_forward` return shape.

1. **Loss-magnitude regression** — train the tiny fixture a few steps and assert
   the final total loss is within a sane band (e.g. `< 1e4`). This is the test
   that would have caught the original bug.
2. **Per-term breakdown** — assert through the observable contract (the
   metrics.jsonl line and the result object), not the inner closure: a run with
   `teacher_weight=0, contrastive_weight=0` produces `train_teacher ≈ 0` and
   `train_contrastive ≈ 0`, and `train_loss ≈ train_supervised`; a run with all
   three weights on produces a `train_loss` consistent with the weighted sum of
   the three reported terms (within the configured weights, to a tolerance).
3. **Eval de-normalization** — `evaluate_teacher` with the dataset's real stats
   returns MAE in physical units; verify that de-normalizing a known normalized
   prediction recovers the expected raw value, and that Pearson r is unchanged
   relative to the pre-fix code path on identical inputs.
4. **metrics.jsonl** — with `log_every=2, total_steps=4`, the file has exactly
   the expected number of lines; each parses as JSON and carries the documented
   keys; `val_*` keys present iff `val_dataset` given.
5. **Resume append** — a resumed run appends to (does not truncate) an existing
   metrics file.
6. Update any existing tests that call `evaluate_teacher` directly to pass the
   new `normalization` argument.

## Files touched

- `qchem_gnn/contrastive_pretrain.py` — teacher-target normalization, per-term
  return, metrics-file writer, console line, `metrics_path` param.
- `qchem_gnn/validation.py` — `evaluate_teacher` signature + de-normalization,
  `run_one_cell` wiring (metrics_path, normalization, overwrite cleanup).
- `qchem_gnn/cli.py` — derive `metrics_path` from `--output` in
  `run_contrastive_pretrain`.
- `tests/test_pretrain_observability.py` — new test module.
- Existing tests referencing `evaluate_teacher` or the `_batch_forward` return
  shape — updated.

## Out of scope

NaN/Inf mid-training guard, gradient clipping, LR warmup/scheduler, TensorBoard,
and CI workflows. Recorded for a future iteration.
