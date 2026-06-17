# Quantum-Teacher Validation Harness — Design

**Date:** 2026-06-17
**Status:** Approved design, pending implementation plan

## Goal

Answer one question with evidence: **does the per-conformer quantum teacher
(added in the previous revision) actually improve the 2D backbone?** The
machinery was built on the assumption that it does; nothing has measured it.
This design specifies a reproducible, one-command harness that answers the
question two ways:

- **Extrinsic** — does a quantum-trained backbone transfer better to the ESOL
  solubility benchmark than a matched baseline?
- **Intrinsic** — does the teacher actually learn to predict held-out
  conformer quantum labels (chelpg, energy, isotropic polarizability)?

The intrinsic diagnostic exists to disambiguate a null extrinsic result:
"the teacher never learned" vs. "the teacher learned but the signal didn't
transfer."

## Experimental design (the science)

A **matched ablation** is the only comparison that isolates the teacher's
contribution. The shipped `example_contrastive.pt` differs from the quantum
config in five ways at once (shard, molecule count, proxy-vs-real student
target, pooling mode, teacher), so comparing against it proves nothing
specific. Instead, two arms differ in **exactly** the treatment:

| | Arm A (baseline) | Arm B (quantum) |
|---|---|---|
| `use_results` | true | true |
| student target | chelpg `[N,1]` | chelpg `[N,1]` |
| `teacher_weight` | 0 | 1 |
| `conformer_pool_mode` | mean | energy |

Everything else — shard (subset 44), molecules, model dims, epochs, learning
rate, contrastive settings — is identical. Both arms see the same real
conformer geometry and the same real chelpg student target; only the teacher
supervision and the Boltzmann pooling change.

- **Data:** all 325 complete molecules in `zinc-250k/results/results_044.h5`
  (not the 64-molecule slice the original quantum config used). Pretraining a
  transferable backbone from a few dozen molecules is too fragile to trust.
- **Seeds:** 3 pretraining seeds per arm → 6 backbones. A single number on
  ~325 molecules can flip with the seed; mean ± std over 3 seeds distinguishes
  a real effect from seed luck.
- **Probe:** each backbone is adapted to ESOL with the existing adapt
  subsystem, fixed adapt seed 42:
  - **`mlp_head` (frozen backbone) — decisive.** Freezing the backbone makes
    any difference purely a function of the learned representations. This is
    the right instrument for "is the backbone better."
  - **`finetune` — headline, not decisive.** It unfreezes the backbone and can
    fine-tune away pretraining differences; a useful practical number but a
    blurry scientific one.
  - ~12 benchmark runs total (6 backbones × 2 methods).

**Verdict rule:** the quantum teacher *helps* iff, on frozen `mlp_head` MAE,
`mean_A − mean_B > sqrt(std_A² + std_B²)` (the effect exceeds combined seed
noise). This is an honest heuristic for n = 3, not a formal significance test;
3 seeds cannot support a t-test and the design does not pretend otherwise.

## Architecture

**One new module + one config; everything else reused.**

- **`configs/validate_quantum_teacher.yaml`** — the experiment as a single
  declarative artifact: shared pretraining settings, the two arm overrides,
  the seed list, the holdout fraction, the two probes with their ESOL adapt
  settings, and output paths.
- **`qchem_gnn/validation.py`** — orchestration and reporting only. It owns no
  training or model logic; it calls pretraining and adapt as black boxes and
  adds three things: the holdout split, the teacher eval, and the aggregation.
- **One small change to existing code:** `ContrastivePretrainingResult` gains
  two in-memory fields, `teacher` and `encoder3d` (default `None`), so the
  harness can run the intrinsic eval. These are never serialized; the
  checkpoint and inference contracts are untouched.

**Reused unchanged:** `contrastive_pretrain_on_dataset`, the entire `adapt/`
subsystem (`runner`, `methods`, `metrics`), the synthetic HDF5 test fixtures,
and the ESOL benchmark configs.

### Component 1 — Experiment config (`validate_quantum_teacher.yaml`)

Declares (illustrative values; the plan fixes exact numbers):

```yaml
command: validate-quantum-teacher

pretrain:
  dataset_root: zinc-250k
  subset_ids: [44]
  limit_per_shard: 400        # >= 325 so all complete molecules load
  use_results: true
  hidden_dim: 64
  message_passing_steps: 3
  epochs: 100
  learning_rate: 0.005
  # shared contrastive settings (batch_size, temperature, 3D dims, etc.)

arms:
  baseline: { teacher_weight: 0.0, conformer_pool_mode: mean }
  quantum:  { teacher_weight: 1.0, conformer_pool_mode: energy, energy_temperature: 298.15 }

seeds: [0, 1, 2]

holdout:
  fraction: 0.15
  seed: 1234                  # fixed; independent of pretraining seed

probes:
  - method: mlp_head          # frozen backbone — decisive
  - method: finetune          # headline
adapt:                        # shared ESOL settings for every probe
  dataset: { csv: data/delaney-processed.csv, smiles_col: auto, targets: auto }
  task: regression
  adapter: { hidden_dims: [128, 64], dropout: 0.1 }
  training: { epochs: 300, lr: 1.0e-3, batch_size: 128, patience: 40, seed: 42 }
  split: { test_frac: 0.2, val_frac: 0.25, seed: 42, stratify: true }

outputs:
  dir: runs/validate
  report: runs/validate/report   # writes report.json and report.md
overwrite: false                  # true bypasses skip-if-exists caching
```

### Component 2 — Holdout split

A single deterministic split of the 325-molecule dataset into a pretrain set
and a holdout set, computed **once** with `holdout.seed` and reused across all
six runs. Properties: deterministic for a fixed seed, disjoint, honors the
fraction. Reusing the same holdout for every arm/seed makes the intrinsic
comparison apples-to-apples and guarantees no holdout molecule leaks into any
arm's training.

### Component 3 — Intrinsic teacher diagnostic (`evaluate_teacher`)

`evaluate_teacher(teacher, encoder3d, holdout_examples) -> dict`:

- Builds a `ConformerEncoderBatch` from the holdout molecules' conformers,
  runs `encoder3d.forward_with_nodes(...)`, applies the teacher heads, and
  compares predictions to the per-conformer DFT labels assembled by
  `assemble_conformer_targets`.
- Returns Pearson r and MAE for each of: chelpg (node), energy and isotropic
  polarizability (graph). wbi (edge) is optional and may be reported if cheap.
- Skips any holdout example missing a target; raises only if the **entire**
  holdout is unusable (a config error worth stopping for).

For Arm A (`teacher_weight = 0`) the teacher heads receive no gradient, so
their predictions are at chance — this is the intended untrained reference.
Arm B is the real test. The diagnostic is primarily an Arm-B question with
Arm A as the chance baseline.

### Component 4 — Orchestration (`run_validation`)

1. Load the shard once; split into pretrain + holdout (Component 2).
2. For each `arm × seed` (6 cells):
   a. If the backbone checkpoint exists and not `overwrite`, load it; else
      train via `contrastive_pretrain_on_dataset(pretrain_set, **arm, seed)`
      and save `runs/validate/{arm}_s{seed}.pt`.
   b. Non-finite guard: if final loss or any saved weight is non-finite, mark
      the cell `failed` and skip its probes.
   c. Intrinsic eval on the holdout (Component 3).
   d. For each probe method: if its metrics JSON exists and not `overwrite`,
      load it; else run the adapt runner on ESOL and save it.
   e. Each backbone-train and each probe runs in its own try/except; a failure
      is recorded as a `failed` row with the exception message and the harness
      continues.
3. Aggregate → report (Component 5).

**Skip-if-exists caching** makes a long run restartable: a killed run resumes
from the first incomplete cell. `overwrite: true` forces a clean recompute.

### Component 5 — Aggregation and report

Collects every `(arm, seed, method, MAE, R², status)` extrinsic row and every
`(arm, seed, property, r, MAE)` intrinsic row, then emits `report.json` (all
raw rows, for re-analysis) and `report.md` with three sections:

1. **Extrinsic (the verdict).** Per method, Arm A vs Arm B as **mean ± std
   over successful seeds**, the delta, the per-row `n=`, and a verdict column.
   The top-line verdict keys off frozen `mlp_head` MAE via the rule above.
2. **Intrinsic (did the teacher learn?).** Per property, Arm B teacher r/MAE
   against Arm A's chance reference.
3. **Raw rows.** Everything, for reproducibility.

mean ± std is computed over whichever seeds succeeded. With `n < 2` successful
seeds in either arm, std is undefined and the verdict prints
`insufficient seeds` rather than a false ✅/～. An arm×method with zero
successes reads `n/a` and its verdict is suppressed.

## Data flow

```
results_044.h5 (325 complete molecules)
   │  load_quantum_zinc_dataset(subset 44, use_results=true, limit>=325)
   ▼
[325 MinimalQuantumExamples]
   │  fixed split (holdout.seed)
   ├──────────────► holdout (~15%) ── never seen in pretraining
   ▼
pretrain set (~85%)
   │   for arm in {baseline, quantum}: for seed in {0,1,2}:
   ▼
  contrastive_pretrain_on_dataset(pretrain_set, **arm, seed)
   │   → result.model (2D student)         → result.teacher, result.encoder3d
   ▼ save runs/validate/{arm}_s{seed}.pt
  ┌─ EXTRINSIC ─────────────────────┐    ┌─ INTRINSIC ───────────────────────┐
  │ adapt runner on ESOL (seed 42): │    │ evaluate_teacher(teacher,encoder3d,│
  │   mlp_head (frozen) | finetune  │    │   holdout) -> per-property r, MAE  │
  │ -> MAE, R² per method           │    └────────────────────────────────────┘
  └─────────────────────────────────┘
   └──────────────► aggregate ◄───────────────────┘
                       ▼
            runs/validate/report.{json,md}
```

## Error handling

- **Skip-if-exists caching** for backbones and probe metrics; `overwrite`
  bypasses it.
- **Per-cell isolation:** each backbone-train and each probe is wrapped in
  try/except; failures become `failed` rows and the run continues.
- **Non-finite guard:** a backbone with non-finite final loss or weights is
  marked failed and its probes skipped, so no garbage backbone is adapted.
- **Holdout integrity:** `evaluate_teacher` skips examples missing a target
  and raises only if the entire holdout is unusable.
- **Aggregation with missing cells:** mean ± std over successful seeds, `n=`
  reported; zero successes → `n/a`; `n < 2` → `insufficient seeds` verdict.

## Testing

TDD, reusing the synthetic HDF5 fixtures (`tests/_quantum_fixtures.py`).

- **Unit — split:** deterministic for a fixed seed, disjoint, fraction
  honored; same seed → identical holdout.
- **Unit — `evaluate_teacher`:** finite r/MAE per property on a tiny set; a
  teacher fit to its targets scores higher r than a randomly-initialized one
  (the metric measures learning, not noise).
- **Unit — aggregation:** hand-built rows produce correct mean ± std; the
  verdict fires ✅ only when `Δ > sqrt(stdA²+stdB²)`, `～` within noise,
  `insufficient seeds` when `n < 2`; failed/missing cells handled (`n=`,
  `n/a`).
- **Unit — caching:** a second `run_validation` with artifacts present does no
  retraining (the pretrain function is not called for cached cells);
  `overwrite` forces recompute.
- **Unit — result fields:** `contrastive_pretrain_on_dataset` returns
  non-`None` `teacher` and `encoder3d`; checkpoint serialization is unchanged
  (the new fields are not written to disk).
- **Integration — tiny end-to-end:** synthetic HDF5, 1 seed, 2 epochs,
  `mlp_head` only → `report.json` + `report.md` with all three sections,
  finite numbers, a verdict string.
- **Regression:** the full existing suite still passes; proxy path and
  inference contract untouched.

## Risks

1. **Transfer signal may be genuinely small.** ~325 pretraining molecules is
   little to transfer to 1128 ESOL molecules. The intrinsic diagnostic is the
   mitigation: it separates "teacher didn't learn" from "didn't transfer."
2. **Wall-clock.** 6 pretrains + ~12 adapt runs (300 epochs each on ESOL) is
   the heavy part. Mitigated by sequential execution with skip-if-exists
   caching; no parallel/distributed execution in scope.
3. **n = 3 is small.** The std-based verdict is a stated heuristic, not a
   significance test; the report says so plainly.

## Out of scope

- Per-conformer contrastive views, multi-shard scaling, and additional quantum
  targets (iao_pops/Fukui) — these are bets that depend on the answer this
  harness produces.
- Parallel/distributed execution and formal statistical significance testing.
- Changing the inference contract or any training/model logic; the harness
  only orchestrates, evaluates the teacher, and reports.
