# Cross-Modal 2D↔3D Contrastive Pretraining — Design

**Date:** 2026-06-17
**Status:** Approved (design); pending implementation plan
**Component area:** `qchem_gnn/` representation learning

## Motivation

The project trains a 2D molecular GNN with supervised, quantum-informed
multi-task regression (per-atom CHELPG/IAO, per-edge WBI, per-graph
energy/polarizability). The repository also stores a large 3D/quantum view of
every molecule — conformer geometries (~1.4 GB) and per-conformer DFT outputs
(~1.7 TB) — but **the encoder never consumes a single 3D coordinate**. The 3D
data enters only as conformer-*averaged* regression targets; the geometry
itself is discarded.

This is the canonical setting for **cross-modal contrastive distillation**: an
expensive teacher view (3D/quantum) is available at pretraining time, but we
want a cheap student (2D graph) at inference. We distill the 3D/quantum
information into the 2D encoder via a contrastive objective, then deploy the 2D
encoder alone.

We deliberately choose cross-modal (2D↔3D) contrastive over augmentation-based
single-view contrastive (GraphCL/MolCLR): random node/edge-drop augmentations
are chemically dubious and yield weak positive pairs, whereas the 3D/quantum
view is a genuine, physics-grounded second view.

## Goals

- Distill the existing 3D/quantum view into the 2D encoder so that the exported
  2D embeddings transfer better downstream.
- Keep **inference 2D-only**: the embedding-export pipeline and
  `encode_graph_embeddings` interface are unchanged.
- Make the benefit **empirically falsifiable** via an ablation on the existing
  downstream harness.

## Non-Goals

- No 3D encoder at inference time (rules out deploying Approach B / denoising).
- No equivariant network machinery — we rely on distance-based invariant
  features only.
- No memory-bank / momentum-encoder negatives (in-batch negatives only for the
  first version; YAGNI).
- No new downstream tasks — reuse the existing linear probe / fine-tune /
  sample-efficiency evaluation.

## Architecture Overview

```
                 ┌──────────────────────────┐
 SMILES / 2D ───▶│  2D MPNN encoder (exists) │──▶ mol_embedding ─┬─▶ quantum heads (exists)
                 └──────────────────────────┘                    │      └─ supervised MSE (anchor)
                                                                 │
                                                  proj_2d ──▶ z_2d ─┐
                                                                    │  symmetric InfoNCE
                 ┌──────────────────────────┐   proj_3d ──▶ z_3d ─┘  (training only)
 conformer  ───▶│  3D distance-RBF encoder  │──▶ per-conf emb ──▶ energy-weighted pool ─▶ 3D mol emb
 coords+E        │      (NEW, training-only) │
                 └──────────────────────────┘

 total_loss = w_sup · multitask_quantum_regression  +  w_con · cross_modal_contrastive
```

At inference, only the boxed 2D encoder runs; `proj_2d`, `proj_3d`, and the 3D
encoder are dropped.

## Components

### 1. 3D teacher encoder — `qchem_gnn/encoder3d.py` (new)

- **Input per conformer:** atomic numbers + 3D coordinates, plus the molecule's
  existing bond connectivity (`edge_index`).
- **Edge featurization:** expand each bonded pair's 3D interatomic distance with
  a Gaussian radial basis (RBF), e.g. `num_rbf` centers over `[0, cutoff]`.
  Distance is rotation/translation invariant, so the encoder is invariant by
  construction without equivariant layers.
- **Backbone:** reuse the residual message-passing block style from
  `model.py` (same `hidden_dim`, SiLU MLPs, LayerNorm residual) so the teacher
  is architecturally parallel to the student. Atom embedding may be shared in
  spirit but is a separate module (the teacher is training-only).
- **Output:** a per-conformer 3D embedding (mean-pool over atoms, then a small
  head — mirroring the 2D `molecular_embedding_head`).
- **Unit contract:** "Given conformer coords + connectivity + energies for a
  batch of molecules, return one molecule-level 3D embedding per molecule."

### 2. Conformer ensemble pooling — `qchem_gnn/conformer.py` (extend)

- Today `ConformerBatch` carries `conformer_index` + `conformer_energy` only.
  Extend it to also carry **per-conformer atom coordinates** (and the mapping
  from conformer atoms to their parent molecule graph).
- Collapse per-conformer 3D embeddings to one molecule-level embedding using the
  **existing `pool_conformer_embeddings`** in energy-weighted (Boltzmann) mode.
  `mean` mode remains available as a fallback / ablation.
- Validation invariants in `__post_init__` extended to cover the coordinate
  tensor shapes.

### 3. Contrastive loss — `qchem_gnn/losses.py` (extend)

- New function: symmetric InfoNCE / NT-Xent over a batch of paired
  `(z_2d, z_3d)` molecule embeddings.
- Steps: project each modality through a small MLP head, L2-normalize, compute
  similarity matrix scaled by temperature `τ`, take cross-entropy against the
  matched-pair diagonal in both directions, average:
  `L = 0.5·(CE(2D→3D) + CE(3D→2D))`.
- **In-batch negatives**: every other molecule in the minibatch.
- Projection heads live in the training module and are discarded at inference.
- The dormant `consistency` slot in `compute_multitask_loss` is superseded by
  this dedicated contrastive term; the multitask loss continues to own the
  supervised regression terms.

### 4. Joint minibatched pretraining loop — `qchem_gnn/contrastive_pretrain.py` (new)

- The existing `pretrain.py` is **full-batch on ~16 molecules/shard**;
  contrastive needs negatives, so this path requires real minibatching.
- New module (keeps `pretrain.py` focused) implementing a minibatched loop:
  - batch size ≈ 64–256 (configurable),
  - per step: run 2D encoder (quantum heads + supervised MSE) **and** 3D teacher
    encoder (+ pooling), compute `w_sup·supervised + w_con·contrastive`,
    backprop through both encoders,
  - projection heads + 3D encoder parameters included in the optimizer.
- Returns a result object analogous to `PretrainingResult` (model, loss history,
  exported 2D embeddings, normalization stats, optimizer/scheduler state, epoch,
  step) so checkpointing/resume stays consistent with existing patterns.

### 5. Data pipeline — `qchem_gnn/quantum_data.py` (extend)

- The loader currently aggregates conformer descriptors into averaged targets
  and discards coordinates. Extend it to **surface per-conformer coordinates and
  per-conformer energies** alongside the existing targets.
- Coordinates come from the geometry pickles (`geometries/coords_*.pkl`);
  energies come from the per-conformer DFT result groups (already read for
  targets). When 3D data is missing for a molecule, it is excluded from the
  contrastive term for that batch (supervised term still applies) rather than
  failing the run.

### 6. Config + CLI — `qchem_gnn/config.py`, `configs/`, `qchem_gnn/cli.py`

- New config fields: `contrastive_weight`, `supervised_weight`, `temperature`,
  3D encoder dims (`hidden_dim_3d`, `num_rbf`, `cutoff`,
  `num_message_passing_steps_3d`), `conformer_pool_mode`, `batch_size`.
- New `configs/minimal_contrastive_pretrain.yaml` mirroring the minimal,
  fast-running style of the existing example configs (single small shard).
- New CLI subcommand `contrastive-pretrain` following the existing YAML-driven
  command pattern, with CLI overrides.

## Success Criteria

An ablation built on the **existing** downstream harness (`eval.py`):

1. Pretrain **supervised-only** (current path) → export 2D embeddings.
2. Pretrain **supervised + contrastive** (new path) → export 2D embeddings.
3. Run **linear probe** and **sample-efficiency** on a held-out target
   (logP/qed/SAS aux labels are already present; or a held-out quantum target).

The contrastive variant is considered a win if it improves probe R²/MAE or the
sample-efficiency curve over the supervised-only baseline on the same held-out
target and split. A null/negative result is a valid outcome and should be
reported as such.

## Testing

Following existing `tests/` conventions:

- `test_encoder3d`: forward shape; **rotation/translation invariance** of the 3D
  embedding; handles single- and multi-conformer molecules.
- `test_contrastive_loss`: output shape (scalar); symmetry; gradient flows;
  perfect-pair sanity (aligned embeddings → low loss, shuffled → high loss).
- `test_conformer`: coordinates carried through `ConformerBatch`; energy-weighted
  pooling reduces conformer embeddings correctly; validation invariants.
- `test_contrastive_pretrain`: one joint training step runs end-to-end on a tiny
  batch and the loss decreases over a few steps; checkpoint round-trips.
- `test_config` / `test_cli`: new fields parse and the new subcommand dispatches.

## Phasing (for the implementation plan)

1. **Phase 1 — model + loss (pure, no data):** `encoder3d.py`, contrastive loss
   in `losses.py`, conformer coord plumbing in `conformer.py`, with their unit
   tests. Fully testable on synthetic tensors.
2. **Phase 2 — data + loop:** extend `quantum_data.py` to surface coords/energies;
   `contrastive_pretrain.py` minibatched joint loop; config + CLI + YAML.
3. **Phase 3 — ablation:** wire the supervised-only vs +contrastive comparison
   into the existing downstream eval and report the result.

## Risks / Open Questions

- **Minibatch refactor is the main cost.** The full-batch `pretrain.py` loop
  cannot supply negatives; the new path must own minibatching.
- **Batch size vs negatives.** Small minibatches weaken in-batch negatives; the
  minimal config will be small/fast for CI, with a larger config for real runs.
- **Missing 3D data per molecule** must degrade gracefully (drop from the
  contrastive term, keep supervised).
- **Conformer count imbalance** across molecules — energy-weighted pooling must
  handle variable conformer counts (already supported by the pooling utility).
