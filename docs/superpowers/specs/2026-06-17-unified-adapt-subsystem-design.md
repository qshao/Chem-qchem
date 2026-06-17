# Unified Adaptation Subsystem — Design

**Date:** 2026-06-17
**Status:** Approved (design)
**Topic:** Make downstream adaptation easy to adapt, robust, and easy to sweep hyperparameters

## Problem

The project has two parallel ways of running things that do not talk to each other:

1. A mature **YAML config + CLI** system (`qchem_gnn/config.py`, `qchem_gnn/cli.py`) with
   validation, deep-merge overrides, and named commands — used for `train`, `pretrain`,
   `contrastive-pretrain`, `export-embeddings`, `eval`, `downstream`.
2. Three **standalone `argparse` scripts** for downstream adaptation
   (`scripts/train_mlp_head.py`, `scripts/train_finetune.py`,
   `scripts/train_engine_adapter.py`) plus `scripts/predict_property.py`. Each training
   script independently duplicates `_detect_columns` and `_stratified_split`, exposes
   hyperparameters only as CLI flags, has no connection to the YAML system, and provides
   no built-in way to sweep hyperparameters or compare runs.

This makes the project hard to adapt to new applications (every new property dataset means
new bespoke flags), fragile (duplicated split/column logic drifts), and tedious for
hyperparameter exploration (sweeps are done by hand, as the epoch-sweep comparison table
was).

### Non-goal / boundary with the existing `downstream` command

The existing `downstream` command (`run_fine_tuning`, `run_linear_probe`,
`run_morgan_baseline`, `run_sample_efficiency` in `qchem_gnn/eval.py`) is **not** redundant
with this work and is left untouched. Its purpose is *research/diagnostic evaluation of a
checkpoint's learned representations on the checkpoint's own native quantum dataset*
(full-batch, transient, prints metrics). The new `adapt` command's purpose is *building a
deployable, saved adapter from an arbitrary external CSV* (minibatched, early stopping,
save/load for inference). The design documents this boundary; it does not merge them.

## Goals

- **Easy to adapt:** adapting to a new property/dataset/application is writing one YAML
  config file, not new code.
- **Robust:** a single source of truth for data loading, column detection, and splitting;
  config validation with clear errors; deterministic seeds; graceful handling of
  unparseable SMILES; a test suite.
- **Easy hyperparameter sweeps:** a `sweep:` block expands a grid, runs each cell, and
  writes/prints a comparison table automatically.

## Decisions (from brainstorming)

| Question | Decision |
|----------|----------|
| Primary run interface | Unified YAML + CLI: a new `adapt` command |
| Hyperparameter sweeps | Built-in `sweep:` block (grid expansion + comparison report) |
| Task scope | Multi-target, regression **and** classification |
| Fate of the 3 `train_*.py` scripts | Replace with example YAML configs (one generic `qchem adapt` path) |
| Implementation scope | One spec; phased implementation plan |

## Architecture

Refactor the two flat modules (`qchem_gnn/adapters.py`, `qchem_gnn/engine_adapter.py`) into
a focused package and wire it into the existing config/CLI system.

```
qchem_gnn/adapt/
  __init__.py      # public API re-exports
  data.py          # CSV load, column auto-detect, multi-target parse, stratified split,
                   #   per-target label normalization  — SINGLE SOURCE OF TRUTH
  config.py        # AdaptConfig dataclass + validation helpers used by the YAML layer
  methods/
    __init__.py
    base.py        # AdaptMethod protocol + shared TrainResult / LoadedAdapter dataclasses
    mlp_head.py    # frozen backbone + MLP head
    finetune.py    # backbone + head joint (dual LR, grad clip)
    engine.py      # ENGINE side structure (multi-exit, early-exit / ensemble inference)
  registry.py      # method name -> method module/class
  runner.py        # orchestrate ONE run: backbone -> data -> split -> train -> eval -> save
  sweep.py         # expand grid (dotted keys), run runner per cell, collect comparison table
  metrics.py       # regression (MAE/RMSE/R2) + classification (AUC/acc/F1), per target
```

Back-compat shims: `qchem_gnn/adapters.py` and `qchem_gnn/engine_adapter.py` remain as thin
modules that re-export from `qchem_gnn.adapt.*` so existing imports
(`from qchem_gnn.engine_adapter import ...`) keep working.

### Method interface (key abstraction)

Every method implements one small protocol so `runner` and `sweep` are method-agnostic and
never branch on method type:

```python
class AdaptMethod(Protocol):
    name: str

    def train(self, backbone, data: AdaptData, cfg: MethodConfig) -> TrainResult: ...
    def save(self, path: Path, result: TrainResult, meta: dict) -> None: ...

    @staticmethod
    def load(path: Path) -> LoadedAdapter: ...

    @staticmethod
    def predict(loaded: LoadedAdapter, smiles: list[str], **kw) -> tuple[np.ndarray, list[int]]: ...
```

Adding a 4th method later (e.g. LoRA) = write one file in `methods/` + one line in
`registry.py`. Saved adapters carry an `adapter_type` key; `predict_property.py` reads it and
dispatches to the right method's `load`/`predict`.

### Data flow (single run)

```
load_backbone(cfg.backbone)
        │
load_dataset(cfg.dataset)  ──>  AdaptData{smiles, targets[N,T], valid_idx, target_names}
        │                         (column auto-detect, drop unparseable SMILES)
stratified_split(targets, cfg.split)  ──>  train/val/test indices
        │                         (stratify on first target's quantiles for regression,
        │                          on class labels for classification)
method.train(backbone, data, cfg)  ──>  TrainResult{adapter, label_stats, log, test_metrics}
        │
method.save(cfg.outputs.adapter, result, meta)
        │
write report (per-target test metrics) -> cfg.outputs.report
```

### Sweep flow

```
expand cfg.sweep.grid (dotted keys -> Cartesian product of override dicts)
for each override:
    merged = deep_merge(base_cfg, override)
    result = runner.run(merged)            # writes its own adapter if outputs.adapter templated
    collect (override, test_metrics)
write cfg.sweep.report (CSV) + print comparison table
```

`outputs.adapter` may contain a template (e.g. `runs/sweep_{idx}.pt`) so each cell saves
distinctly; if absent, sweep cells are evaluated but not persisted.

## Config schema

New top-level command `adapt`. New sections; validated in `qchem_gnn/config.py` alongside
existing ones.

```yaml
command: adapt
method: finetune                 # mlp_head | finetune | engine
backbone: runs/example_contrastive.pt
task: regression                 # regression | classification

dataset:
  csv: data/delaney-processed.csv
  smiles_col: auto               # auto-detect if "auto"/omitted
  targets: [measured log solubility in mols per litre]   # one or more columns; "auto" allowed

adapter:                         # method-specific knobs (validated per method)
  hidden_dims: [128, 64]
  dropout: 0.1

training:
  epochs: 200
  head_lr: 1.0e-3
  backbone_lr: 5.0e-5            # finetune only
  batch_size: 64
  patience: 30
  grad_clip: 1.0                 # finetune only

split:
  test_frac: 0.2
  val_frac: 0.25
  seed: 42
  stratify: true

outputs:
  adapter: runs/finetune_sol.pt
  report: runs/finetune_metrics.json

sweep:                           # OPTIONAL — presence triggers sweep mode
  grid:
    training.epochs: [10, 50, 100, 200, 400]
    adapter.dropout: [0.0, 0.1]
  report: runs/sweep_results.csv
```

Validation rules (added to `config.py`):
- `method` ∈ {mlp_head, finetune, engine}; `task` ∈ {regression, classification}.
- `backbone` required, must be a path string.
- `dataset.targets` non-empty list of strings or `"auto"`.
- method-specific keys validated against an allowlist per method (e.g. `backbone_lr`/`grad_clip`
  only valid for `finetune`); unknown keys raise `ConfigError`.
- `split` fractions in `(0,1)`; `seed` integer.
- If `sweep` present: `grid` is a non-empty mapping of dotted-key → non-empty list; every
  dotted key must resolve to a valid config path.

## Multi-target & task type

- **Heads** emit `len(targets)` units.
- **Regression:** loss = MSE; per-target z-score normalization fit on the train split; metrics
  MAE/RMSE/R² reported per target plus a macro average.
- **Classification:** loss = BCEWithLogits (binary/multilabel) or CrossEntropy (multiclass);
  no label normalization; metrics AUC/accuracy/F1 per target. Predictions return
  probabilities.
- **ENGINE** generalizes its per-exit head from `Linear(h, 1)` to `Linear(h, T)`; the summed
  multi-exit loss and early-exit/ensemble inference operate over the target dimension.

## CLI

```bash
qchem adapt configs/adapt_finetune_solubility.yaml        # single run (no sweep:)
qchem adapt configs/adapt_engine_epoch_sweep.yaml         # sweep (has sweep:)
```

`cli.py` adds an `adapt` subparser (config path + the same dotted overrides other commands
accept). On dispatch it loads/resolves the config; if `sweep:` is present it calls
`sweep.run`, otherwise `runner.run`.

Inference (`scripts/predict_property.py`) is updated to import from `qchem_gnn.adapt` and to
dispatch purely on the saved `adapter_type`, supporting multi-target output columns.

## What changes for existing code

- `qchem_gnn/adapters.py`, `qchem_gnn/engine_adapter.py` → logic moves into `qchem_gnn/adapt/`;
  the old module paths become thin re-export shims (no breakage for current imports/tests).
- `scripts/train_mlp_head.py`, `scripts/train_finetune.py`, `scripts/train_engine_adapter.py`
  → **removed**, replaced by example configs in `configs/`:
  - `configs/adapt_mlp_head_solubility.yaml`
  - `configs/adapt_finetune_solubility.yaml`
  - `configs/adapt_engine_solubility.yaml`
  - `configs/adapt_engine_epoch_sweep.yaml` (reproduces the by-hand epoch sweep)
- `scripts/predict_solubility.py` → kept (ENGINE-specific convenience) but re-pointed at the
  package; `scripts/predict_property.py` becomes the recommended general entry point.
- `qchem_gnn/eval.py` `downstream` path → **unchanged**.
- Tutorial `docs/tutorials/engine_solubility_tutorial.md` → commands updated to `qchem adapt`.

## Robustness measures

- Single source of truth for column detection + split in `adapt/data.py` (removes current
  triplicate drift risk).
- Config validation with explicit `ConfigError` messages for every field.
- Deterministic: all randomness seeded from `split.seed` and a training seed; splits
  reproducible.
- Unparseable SMILES skipped with a reported count; never crash a run.
- Tests:
  - unit: `data.py` column detection (incl. Delaney's lowercase `smiles` + long target name),
    stratified split proportions and disjointness, label normalization round-trip,
    `registry` lookup, `config` validation (valid + each invalid case), sweep grid expansion.
  - integration: end-to-end smoke test per method on a tiny synthetic CSV (≤32 rows) →
    train a couple of epochs, save, reload, predict; assert shapes and finite metrics.
  - regression: a multi-target run and a classification run on synthetic data.

## Implementation phases (for the plan)

1. **Package + single-run regression** — `adapt/` package, method protocol, port the three
   existing methods (single-target regression), `runner`, config schema + validation, CLI
   `adapt` command, back-compat shims, example single-run configs, port `predict_property.py`.
   Tests for data/split/registry/config + per-method smoke tests. Reproduce existing Delaney
   numbers via configs.
2. **Sweep** — `sweep.py`, `sweep:` config + validation, comparison report (CSV + printed
   table), `adapt_engine_epoch_sweep.yaml`. Test grid expansion and a tiny end-to-end sweep.
3. **Multi-target + classification** — generalize heads/losses/metrics, ENGINE multi-output
   exits, `task`/multi-`targets` config, classification metrics. Multi-target and
   classification regression tests. Update tutorial.

## Success criteria

- A new property dataset is adapted by writing one YAML config; no new Python required.
- `qchem adapt cfg.yaml` reproduces the previously hand-run MLP-head / fine-tune / ENGINE
  Delaney results within noise.
- A single sweep config reproduces the epoch-sweep comparison table automatically.
- Column-detection and split logic exist in exactly one place.
- All new code covered by unit + integration tests; existing tests still pass.
