# Per-Conformer Quantum Supervision — Design

**Date:** 2026-06-17
**Status:** Approved design, pending implementation plan

## Goal

Make the contrastive pretraining actually exploit the dataset's defining
feature: each molecule has multiple low-energy conformers, each with its own
DFT-computed quantum labels. Today those labels are mean-collapsed and (on the
default path) never read at all. This design supervises a geometry-aware
teacher on **per-conformer** quantum properties and distills a
**Boltzmann-ensemble** prediction into the 2D student, so SMILES-only
inference still works.

## Motivation — what's broken today

Validated against the code and a real DFT shard (`results_044.h5`):

1. **Per-conformer labels are averaged away.** `quantum_data._aggregate_targets`
   (lines 132–154) does `.mean(dim=0)` over conformers, destroying the
   conformation-dependent variation that is the entire point of computing
   quantum properties on multiple geometries.
2. **The default path reads no quantum data.** `minimal_contrastive_pretrain.yaml`
   sets `use_results: false`, so `_build_proxy_targets` fabricates targets from
   graph structure. The shipped `example_contrastive.pt` never saw DFT data.
3. **The real-data path additionally crashes.** `_aggregate_targets` stacks
   `chelpg` `(57,)` with `iao_pops` `(173,)` — a shape mismatch that throws and
   silently falls back to proxy targets.
4. **The 3D/quantum encoder gets no quantum supervision.** It is trained only by
   InfoNCE to match the 2D embedding (`contrastive_pretrain.py:140`), inverting
   the intended "geometry+quantum teaches 2D" direction.
5. **Conformer energies are never loaded** (`quantum_data.py:247` hardcodes
   `None`), so energy-weighted pooling can't fire — mean pooling weights an
   unlikely high-energy conformer equally with the ground state.

## Validated data schema (`results_044.h5`)

Smallest real shard: 2.4 GB, **325 molecules** with complete per-conformer DFT
(small because DFT finished 325 of the shard's ~998). Each molecule group has
`conf_0..conf_N` (N varies: 5, 3, 1 observed).

| dataset/attr | shape | use |
|---|---|---|
| `chelpg` | `(num_atoms,)` | **node target** (per-atom charge) |
| `wbi` | `(num_atoms, num_atoms)` | **edge target** (gather endpoints) |
| `coords` | `(num_atoms, 3)` | fed to 3D encoder |
| `energy` (attr) | scalar (Hartree) | **graph target** + Boltzmann weight |
| `polarizability` (attr) | `(3, 3)` | **graph target** as isotropic = trace/3 |
| `iao_pops` | `(173,)` per-orbital | **dropped** (not per-atom; needs basis map) |
| `fukui_p`, `dm` | per-AO / large | unused |

## Final target set (all per-conformer)

- **Node:** chelpg → `[N, 1]`
- **Edge:** wbi (gathered from matrix) → `[E, 1]`
- **Graph:** `[energy, isotropic_polarizability]` → `[2]`

`iao_pops` dropped for now (per-orbital length, no stored basis→atom mapping).
Revisit later via PySCF if orbital populations are wanted.

## Architecture

**Reused unchanged:** `MolecularQuantumGNN` (2D student; node/edge/graph heads +
`mol_embedding`), `Conformer3DEncoder` (teacher backbone),
`ConformerEncoderBatch`, `info_nce_contrastive_loss`, `compute_multitask_loss`,
all adapters and the entire inference path.

**Changed:**
1. Data layer keeps per-conformer tensors and loads energies.
2. Teacher gains prediction heads (today it only emits an embedding).
3. Training loop adds a teacher regression term and switches contrastive
   pooling to energy-weighted.

The 2D inference path is untouched; the payoff is a student whose weights are
shaped by real per-conformer quantum physics.

### Component 1 — Data layer (`quantum_data.py`)

- Replace `_aggregate_targets` with a per-conformer loader producing:
  - `conformer_node_targets` `[C, N, 1]` (chelpg)
  - `conformer_edge_targets` `[C, E, 1]` (wbi endpoints)
  - `conformer_graph_targets` `[C, 2]` (energy, iso-polarizability)
  - `conformer_energies` `[C]` (from `conf.attrs["energy"]`)
- Isotropic polarizability = `trace(alpha_3x3) / 3` (rotation-invariant).
- New helper `boltzmann_weights(energies, T=298.15)`:
  `softmax(-(E - E.min()) / kT)`, kT in Hartree (≈ 9.4e-4 at 298 K). Missing or
  degenerate energies → uniform weights, no crash.
- Molecule-level `node/edge/graph_target` (what the 2D student regresses)
  becomes the **Boltzmann-weighted average** over conformers, not the plain
  mean.
- Filter loaded molecules to mol_ids present in the HDF5 shard (the 325 real
  ones); skip the rest rather than proxy-filling.
- `MinimalQuantumExample` gains the four per-conformer fields above.

### Component 2 — Teacher heads (`teacher_heads.py`)

Three lightweight heads on the 3D encoder's per-atom features:
- node head: per-atom feature → chelpg `[·, 1]`
- edge head: concatenated endpoint features → wbi `[·, 1]`
- graph head: energy-conformer-pooled feature → `[energy, iso_polarizability]`

Teacher loss = per-conformer MSE on all three, summed over every conformer in
the batch, against the per-conformer DFT labels. This is the change that gives
the 3D teacher genuine quantum supervision.

### Component 3 — Training loop (`contrastive_pretrain.py`)

```
total = w_sup       * student_ensemble_MSE      # student vs Boltzmann-avg labels
      + w_teacher   * teacher_perconformer_MSE  # NEW
      + w_contrast  * energy_weighted_InfoNCE    # pooling mode -> "energy"
```

New config keys under `contrastive`: `teacher_weight`, `energy_temperature`,
and `conformer_pool_mode: energy`. A real-data config sets `use_results: true`
against `results_044.h5`.

### Component 4 — Inference

Unchanged. 2D model + adapters. The student now carries quantum-informed
representations.

## Data flow

```
DFT HDF5 (results_044) ─► per-conformer labels + energies
   │                          │
   │                          ├─► Boltzmann-avg ─► 2D student target (MSE)
   │                          └─► per-conformer  ─► teacher target   (MSE)
conformer coords ─► 3D encoder ─► per-atom feats ─► teacher heads (per-conf preds)
                                     │
                                     └─ energy-weighted pool ─► 3D mol emb
2D graph ─► student ─► node/edge/graph preds + 2D mol emb
                                     │
        energy-weighted InfoNCE(2D mol emb, 3D mol emb)
```

## Testing

- **Unit:** `boltzmann_weights` (known energies → known weights; degenerate →
  uniform; missing → uniform). Per-conformer loader returns correct `[C, …]`
  shapes; ensemble average matches a hand Boltzmann calc. Isotropic
  polarizability = trace/3. Teacher head output shapes. Teacher loss drops on a
  tiny overfit batch.
- **Integration:** one full pretrain step with `use_results: true` on a small
  slice of `results_044.h5` (e.g. first 8 molecules) — all three loss terms
  finite and nonzero; checkpoint saves/loads.
- **Regression:** existing 120 tests still pass; proxy-target path
  (`use_results: false`) still works for users without DFT shards.

## Risks

1. Conformer count varies (1–5) — loader and batching must handle ragged C.
2. Energy units are Hartree; verify kT scaling produces sane weights on a real
   conformer set (energies within a molecule differ by ~1e-2–1e-1 Hartree).
3. Reading from a 2.4 GB HDF5 in tests — slice to a handful of molecules; never
   load the whole shard into memory.

## Out of scope

- iao_pops / fukui / density-matrix targets.
- Changing the inference contract (stays 2D-only).
- Multi-shard or full-dataset training (this design targets the one small real
  shard; scaling is a later step).
