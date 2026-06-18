# Scaled Pretraining Tutorial

Train the contrastive GNN backbone on the full ZINC-250K dataset (~250 K molecules,
~250 shards), measure downstream accuracy vs. data scale, and export trained
backbones for property prediction.

**Why scale?** Previous experiments used a single shard (~276 molecules). Every
contrastive objective tweak landed "within noise" at that scale because the signal
was buried under seed variance. Scaling ~900× is the single largest untried lever
for accuracy improvement.

---

## Hardware requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| Disk (raw data) | 7 GB (1 shard) | 1.7 TB (all shards) |
| Disk (compact cache) | 90 MB (1 shard) | ~22 GB (all shards) |
| RAM | 2 GB | 64 GB+ (to preload many shards) |
| GPU | optional | strongly recommended for training |

The preprocessing step converts each ~6.8 GB raw shard into a ~90 MB compact
cache (drops the density matrix, which is never used in training). You only pay
that disk I/O once; every subsequent training run loads the compact cache in seconds.

---

## Dataset layout expected

```
zinc-250k/
├── subsets/          # shard CSVs: subset_000.csv, subset_001.csv, ...
├── geometries/       # conformer pickles: coords_000.pkl, coords_001.pkl, ...
└── results/          # quantum HDF5: results_000.h5, results_001.h5, ...
```

The preprocessing step reads from this layout and writes to a separate
`compact_cache/` directory you choose.

---

## Step 1 — One-time preprocessing

Preprocessing converts raw shards into compact, scaffold-keyed caches. It is
**resumable**: shards with a valid existing cache are skipped.

```bash
# Quick smoke-test: preprocess shard 0 only (~90 MB output, ~a few minutes)
bash scripts/preprocess.sh zinc-250k zinc-250k/compact_cache "0"

# First 10 shards (~900 MB, good for initial training experiments)
bash scripts/preprocess.sh zinc-250k zinc-250k/compact_cache "0-9"

# All 250 shards (~22 GB, hours on first run; instant on re-run)
bash scripts/preprocess.sh zinc-250k zinc-250k/compact_cache "0-249"
```

If the run is interrupted, re-run the same command. Completed shards are skipped.

To force re-extraction (e.g., after a corrupt write):
```bash
bash scripts/preprocess.sh zinc-250k zinc-250k/compact_cache "12,37" --overwrite
```

What preprocessing does:
- Loads each shard through the existing extractor (drops the `dm` density matrix)
- Attaches a globally stable Murcko scaffold key to each molecule
- Saves `compact_cache/shard_NNN.pt` (versioned, validates on load)

---

## Step 2 — Train a backbone

### Option A: Train directly from raw data (single shard, fast)

Use `scripts/train.sh` when you want to train on a small subset without
preprocessing first. This reads from raw HDF5 and is limited to one shard's
worth of molecules.

```bash
# Single seed, default shard 0, output to checkpoints/
bash scripts/train.sh zinc-250k checkpoints "0"
```

The script trains with the best configuration found in ablation experiments:
quantum teacher + scaffold-aware InfoNCE negative masking, 200 epochs, hidden
dim 64, batch size 16.

### Option B: Train from compact caches (multi-shard, recommended for scale)

After preprocessing, use the validation harness with `configs/validate_scaled.yaml`.
This is the production path for multi-shard training.

**Edit `configs/validate_scaled.yaml`** to set which shards to train on:

```yaml
pretrain:
  cache_dir: zinc-250k/compact_cache
  subset_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # <-- adjust this list
  epochs: 200
  hidden_dim: 64
  batch_size: 16
```

Then run:
```bash
python -m qchem_gnn.validation --config configs/validate_scaled.yaml
```

This trains 3 seeds × 1 arm, runs ESOL downstream probes (mlp_head + finetune),
and writes a report to `runs/validate_scaled/report.json`.

**Holdout:** the config uses scaffold-disjoint holdout (`k: 10`), which withholds
~1/10 of scaffolds from pretraining — a stricter, leakage-free split.

The trained backbone checkpoints are saved to `runs/validate_scaled/`:
```
runs/validate_scaled/
├── quantum_scaffold_s0.pt    # backbone, seed 0
├── quantum_scaffold_s1.pt    # backbone, seed 1
├── quantum_scaffold_s2.pt    # backbone, seed 2
└── report.json               # downstream MAE, R², per probe and seed
```

---

## Step 3 — Export embeddings

Given any backbone checkpoint, export one embedding vector per molecule:

```bash
bash scripts/infer.sh runs/validate_scaled/quantum_scaffold_s0.pt embeddings.npy
```

The output is a NumPy `.npy` file with shape `[N_molecules, hidden_dim]`.

---

## Step 4 — Adapt to a new property

Train a lightweight MLP head on top of a frozen backbone for any property CSV:

```bash
# Copy the template and edit it
cp configs/adapt_example.yaml configs/adapt_mydata.yaml
```

Edit `configs/adapt_mydata.yaml`:
```yaml
backbone: runs/validate_scaled/quantum_scaffold_s0.pt

dataset:
  csv: data/my_property.csv     # CSV with SMILES and property columns
  smiles_col: smiles
  targets: [my_property]

task: regression

adapter:
  hidden_dims: [128, 64]
  dropout: 0.1

training:
  epochs: 100
  lr: 1.0e-3
  batch_size: 32
  patience: 10
  seed: 42

split:
  test_frac: 0.1
  val_frac: 0.1
  seed: 42
  stratify: false

output: results/
```

Then run:
```bash
bash scripts/adapt.sh configs/adapt_mydata.yaml
```

Optional inline overrides (no YAML edit required):
```bash
bash scripts/adapt.sh configs/adapt_mydata.yaml training.epochs=200 training.lr=5e-4
```

Reproduce the published ESOL benchmark:
```bash
bash scripts/adapt.sh configs/adapt_mlp_head_solubility.yaml
bash scripts/adapt.sh configs/adapt_finetune_solubility.yaml
```

---

## Step 5 — Scaling sweep (optional)

Measure how downstream MAE changes as you increase the number of pretraining
shards. This answers: "does more data actually help?"

```bash
# Train at 1, 10, and 50 shards and print a MAE-vs-scale table
bash scripts/scaling_sweep.sh zinc-250k zinc-250k/compact_cache "1 10 50"
```

The sweep:
1. Preprocesses shards (skip-if-exists — safe to re-run)
2. Trains the backbone at each scale point with 3 seeds
3. Runs ESOL downstream probes at each scale
4. Prints a table like:

```
 scale     method               arm   mae_mean
     1   mlp_head  quantum_scaffold     0.9200
    10   mlp_head  quantum_scaffold     0.8700
    50   mlp_head  quantum_scaffold     0.8100
     1    finetune  quantum_scaffold     0.8500
    10    finetune  quantum_scaffold     0.7900
    50    finetune  quantum_scaffold     0.7400
```

To aggregate from existing report files manually:
```bash
python scripts/aggregate_scaling.py \
  1=runs/scaling_s1/report.json \
  10=runs/scaling_s10/report.json \
  50=runs/scaling_s50/report.json
```

---

## Full end-to-end example (10 shards)

```bash
# 0. Activate environment
source .venv/bin/activate

# 1. Preprocess 10 shards (~900 MB, ~10 minutes)
bash scripts/preprocess.sh zinc-250k zinc-250k/compact_cache "0-9"

# 2. Train backbone from compact cache (3 seeds, ~hours)
python -m qchem_gnn.validation --config configs/validate_scaled.yaml

# 3. Check the downstream report
python -c "
import json
r = json.load(open('runs/validate_scaled/report.json'))
for method, arms in r['aggregate']['extrinsic'].items():
    for arm, stats in arms.items():
        print(f'{method:10s} {arm:20s}  MAE={stats[\"mae_mean\"]:.4f}')
"

# 4. Export embeddings from the best seed
bash scripts/infer.sh runs/validate_scaled/quantum_scaffold_s0.pt embeddings.npy

# 5. Adapt to solubility
bash scripts/adapt.sh configs/adapt_finetune_solubility.yaml
```

---

## Troubleshooting

**`KeyError: 'dataset_root'` when running validation**
Make sure your YAML has `cache_dir` (not `dataset_root`) under `pretrain:` when
using compact caches. These are two separate loading paths.

**`ValueError: shard cache version mismatch`**
A cache file was written by an older version. Re-extract it:
```bash
bash scripts/preprocess.sh zinc-250k zinc-250k/compact_cache "N" --overwrite
```

**`ValueError: scaffold_hash_holdout produced an empty side`**
The `k` value in `holdout.k` is routing all molecules to one side. Use a larger
`k` (the default `k: 10` works for any dataset with ≥ 10 distinct scaffolds).

**Training OOM (out of memory)**
Reduce `batch_size` in the config (try 8), or reduce `subset_ids` to fewer shards.

**`import h5py` error**
Install the optional h5py dependency: `pip install h5py`.
