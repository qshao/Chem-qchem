# Scaled Pretraining Pipeline — Design

**Date:** 2026-06-18
**Status:** Approved design, pending implementation plan

## Goal

Train the proven contrastive backbone (InfoNCE + quantum teacher + scaffold-aware
negative masking) on the full available ZINC dataset (~250K molecules across ~250
shards) instead of the single shard (~276 pretrain molecules) used to date, and
measure whether the ~900× data scale-up produces robust downstream accuracy gains.

This serves two goals simultaneously: **real accuracy gains** (data scale is the
single largest untried lever — every prior objective tweak landed "within noise"
because the signal was buried under seed variance on a tiny dataset) and
**productionization** (a documented, resumable preprocessing + loading pipeline
that scales from 1 to all shards without touching the training loop).

## Scope and decisions

- **Phase 1 (this spec): preload.** The full compact cache for 250K molecules is
  ~22 GB; the target machine has ~114 GB free RAM. Phase 1 preloads selected
  compact shards into one in-memory dataset and trains the existing backbone
  unchanged. This is the fastest, zero-trainer-rewrite path to the scaling answer,
  with perfect global shuffle (best in-batch negative diversity) and zero disk I/O
  after the one-time load.
- **Phase 2 (separate spec, deferred): shard-block streaming.** Constant-RAM,
  unbounded-scale streaming loader (WebDataset-style: shuffle shard order per
  epoch, one shard resident at a time, with a shuffle buffer). Build only when the
  dataset grows past ~1 M molecules (compact cache approaches the RAM ceiling).
  MoCo-style memory-queue negatives also slot into Phase 2.
- **Bounded-LRU random access is rejected.** Its only advantage — perfect shuffle
  without preload — is moot when preload fits in RAM; with >1000 shards and global
  random access it re-reads a full shard per example (catastrophic I/O).
- **Compaction by dropping the density matrix.** Each conformer's `dm` (436×436
  float64 ≈ 1.5 MB) is ~99% of HDF5 storage and is never used in training. The
  existing extraction in `quantum_data.py` already ignores `dm`/`fukui_p`; the
  preprocessing step persists what extraction already produces. Result: ~6.8 GB
  per shard → ~90 MB per shard; ~1.7 TB → ~22 GB.
- **Scaffold mask rebuilt for scale.** A global `[N, N]` mask over 250K molecules
  is ~60 GB and infeasible. Replace it with a precomputed **globally stable**
  `scaffold_key` per example — a deterministic hash of the Murcko scaffold SMILES
  (`int.from_bytes(hashlib.blake2b(scaffold_smiles.encode(), digest_size=8).digest())`),
  **not** Python's per-process-salted `hash()` and **not** a per-shard appearance
  index (either would map the same scaffold to different keys across shards). The
  mask is built per batch from `scaffold_key[i] == scaffold_key[j]` (O(batch²),
  trivial). The global-mask path and the trainer's `full_mask` / `global_idx`
  plumbing are removed.
- **Scaffold-disjoint holdout.** Replace the in-memory shuffle-split with a
  deterministic hash-of-scaffold holdout (`scaffold_key % K == 0`, reusing the
  same globally stable `scaffold_key`): scaffold-disjoint (no leakage into teacher
  or downstream eval), no global materialization required, identical bucketing for
  a scaffold no matter which shard it appears in.
- **Backbone unchanged.** Model, projection heads, teacher heads, InfoNCE,
  Boltzmann pooling, VICReg arm, adapt subsystem, and the inference/checkpoint
  contract are all reused byte-for-byte. Only data feeding and scaffold-mask
  construction change.

## Architecture

Two cleanly separated subsystems plus minimal trainer and harness integration.

1. **Offline preprocessing** (`preprocess` CLI command) — streams each shard's
   HDF5 + geometry, extracts the compact per-molecule example, computes
   `scaffold_id`, and serializes a versioned per-shard cache `cache/shard_XXX.pt`.
2. **Preload dataset loader** — reads a selected range of compact shard caches
   into one in-memory `MinimalQuantumDataset`, consumed by the existing trainer.
3. **Per-batch scaffold mask** — built in the trainer from `scaffold_id`,
   replacing the global mask.
4. **Scaffold-hash holdout** — deterministic, scaffold-disjoint split used by the
   validation harness.
5. **Scaling experiment** — a config + harness path that trains the proven
   backbone at increasing shard counts and reports downstream MAE vs. scale.

### Component 1 — Compact format + `preprocess` CLI

For each shard id in a requested range:

- If `cache/shard_XXX.pt` exists and is valid (loads, version matches), skip.
- Otherwise call the existing single-shard extraction (`load_quantum_zinc_dataset`),
  which already drops `dm`/`fukui_p`, yielding `list[MinimalQuantumExample]`.
- Compute the Murcko scaffold SMILES per molecule (reusing the scaffold inference
  already in `eval.py`) and reduce it to a globally stable `scaffold_key` (8-byte
  blake2b digest as an int).
- `torch.save` a versioned payload: `{"version": 1, "examples": [...],
  "scaffold_keys": [...]}` to `cache/shard_XXX.pt` (~90 MB).

Resumable (skip-if-exists), parallelizable across shards. No new extraction logic.

### Component 2 — Preload dataset loader

```python
def load_compact_shards(cache_dir, shard_ids) -> MinimalQuantumDataset:
```

Reads each requested compact shard cache, validates its version, concatenates the
examples (and their `scaffold_key`s) into one in-memory `MinimalQuantumDataset`.
The trainer's existing random-index batching works unchanged. Selecting a shard
sub-range (1, 10, 50, all) is just a different `shard_ids` list.

### Component 3 — Per-batch scaffold mask

`build_scaffold_negative_mask` is reworked (or superseded) to operate on a batch's
`scaffold_key` vector rather than a global example list:

```python
def scaffold_mask_from_keys(scaffold_keys: torch.Tensor) -> torch.Tensor:
    eq = scaffold_keys[:, None] == scaffold_keys[None, :]
    eq.fill_diagonal_(False)
    return eq
```

In `contrastive_pretrain_on_dataset`, the pre-loop `full_mask` and per-batch
`global_idx` slicing are removed; the per-batch mask is built from the batch's
`scaffold_key`s (sliced by `coords_index`, like the rest of the conformer path).
`use_scaffold_negmask=False` skips construction entirely (unchanged default
behaviour). The masked-InfoNCE path in `losses.py` is untouched.

### Component 4 — Scaffold-hash holdout

```python
def scaffold_hash_holdout(dataset, k: int) -> tuple[pretrain, holdout]:
```

A molecule is holdout iff `scaffold_key % k == 0` (the same globally stable
`scaffold_key` from preprocessing). Deterministic, scaffold-disjoint, no global
shuffle needed. Replaces `split_holdout` in the validation harness's
pretrain/holdout construction. The downstream adapt eval (external ESOL data) is
unaffected.

### Component 5 — Scaling experiment

A validation config (or a thin sweep wrapper) that runs the proven backbone at
shard counts **1, 10, 50, all** × 3 seeds, then the existing ESOL downstream
probes (`mlp_head`, `finetune`, `engine`). The report adds a **MAE-vs-scale**
view per probe. Optionally a scaffold-on/off arm at the largest scale, to test
whether scaffold masking separates from noise once batches routinely contain
scaffold mates (it could not at 276 molecules).

## Data flow

```
One-time preprocessing (resumable):
  for shard 000..NNN:
    if cache/shard_XXX.pt valid: skip
    HDF5 + geometry → extract compact example (drop dm/fukui_p)
    Murcko scaffold SMILES → globally stable scaffold_key per molecule
    torch.save({version, examples, scaffold_keys}) → cache/shard_XXX.pt

Each training run:
  load_compact_shards(cache_dir, shard_ids) → in-memory dataset (~22GB for all 250)
  scaffold_hash_holdout(dataset, k) → (pretrain, holdout)   # scaffold-disjoint
  contrastive_pretrain_on_dataset(pretrain)                 # backbone unchanged
    └ per batch: mask = scaffold_mask_from_keys(batch scaffold_keys)

Scaling experiment:
  for scale in [1, 10, 50, all] × seeds:
    train proven backbone at `scale` shards
    run ESOL probes (mlp_head / finetune / engine)
  report MAE vs scale per probe
```

## Error handling

- **Partial preprocessing** — skip-if-exists per shard makes the run resumable
  after interruption.
- **Corrupt/half-written cache** — detected by load-and-validate (version +
  structural check); re-extracted rather than trusted.
- **Missing/corrupt HDF5 or geometry** — logged and skipped via the existing
  `skipped_mol_ids` mechanism; never crashes a multi-shard run.
- **Schema drift** — cache payload carries a format `version`; a mismatch triggers
  re-extraction, never silent wrong data.
- **Empty holdout** for a scaffold-sparse shard range — validated with a clear
  error before training starts.
- Existing per-cell isolation, non-finite backbone guard, and skip-if-exists
  backbone caching in the validation harness are inherited unchanged.

## Testing

TDD, reusing `tests/_validation_fixtures.py` and a small 2-shard fixture.

- **Preprocess unit** — 2-shard fixture round-trips: extract → save → reload
  yields identical examples; `dm` absent from the cache payload; `scaffold_key`
  present, deterministic across repeated runs, and equal for two molecules sharing
  a Murcko scaffold across different shards.
- **Preload unit** — loading 2 compact shards yields one dataset whose length and
  indexing equal the concatenation; selecting a shard sub-range works.
- **Scaffold-hash holdout** — deterministic for a fixed `k`; pretrain and holdout
  are scaffold-disjoint; both non-empty on the fixture.
- **Per-batch scaffold mask** — `scaffold_mask_from_keys` equals the previous
  global-mask slice on a small batch (regression guard for the rewrite); diagonal
  always False; symmetric.
- **Integration** — preprocess → preload → `contrastive_pretrain_on_dataset`
  returns a finite loss history on the 2-shard fixture; the backbone checkpoint
  loads with the existing `export-embeddings` path.
- **Regression** — full suite stays green; the single-shard path reproduces
  today's result (compact cache is behaviourally identical to direct extraction).

## Risks

1. **Scaling may still not separate from noise.** Possible but unlikely at ~900×
   data; if downstream MAE is flat from 1→all shards, that is itself a decisive,
   publishable finding and redirects effort away from data toward architecture.
2. **Preprocessing wall-clock.** Parsing ~1.7 TB of HDF5 once is hours, not
   minutes; mitigated by resumability and across-shard parallelism. It is paid
   once; every subsequent run loads the 22 GB compact cache in seconds.
3. **Preload RAM ceiling.** Fine at 250K (~22 GB ≪ 114 GB); growth toward ~1 M
   molecules (~90 GB) is the documented trigger to switch to Phase 2 shard-block
   streaming.
4. **Scaffold-hash holdout fraction drift.** Hash-bucket holdout gives an
   approximate, not exact, holdout fraction; acceptable for evaluation and
   deterministic.

## Out of scope (Phase 2 / future)

- Shard-block streaming loader and shuffle buffer (constant-RAM, unbounded scale).
- MoCo-style momentum-encoder memory-queue negatives.
- Equivariant 3D teacher or richer 2D atom/bond features (representation upgrade).
- Any change to the inference contract, checkpoint format, VICReg arm, or adapt
  subsystem.
