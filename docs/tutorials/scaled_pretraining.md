# Scaled Pretraining Tutorial

Train the contrastive GNN backbone on the full ZINC-250K dataset (~250K molecules,
~250 shards), then adapt the trained backbone to any property prediction task.

**Why scale?** Previous experiments used a single shard (~276 molecules). Every
contrastive objective tweak landed "within noise" at that scale because the signal
was buried under seed variance. Scaling ~900× is the single largest untried lever
for accuracy improvement.

**Pipeline overview:**

```
raw ZINC data
      │
      ▼  Step 1 — preprocess.sh (once)
compact shard cache (.pt files, ~90 MB/shard)
      │
      ▼  Step 2 — train.sh
backbone checkpoints (quantum_scaffold_s{N}.pt)
      │
      ▼  Step 3 — adapt.sh
downstream task model + metrics
```

---

## Installation

### Requirements

- Python 3.13 or newer
- CUDA-capable GPU (optional but strongly recommended for training)

### Create a virtual environment

```bash
# From the project root
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### Install dependencies

```bash
pip install numpy pandas pyyaml scikit-learn scipy h5py pytest
pip install torch                        # see https://pytorch.org for CUDA builds
pip install torch-geometric              # graph neural network library
pip install rdkit                        # molecular informatics
```

> **GPU build of PyTorch:** the default `pip install torch` installs the CPU build.
> For CUDA 12.x, use the PyTorch install selector at pytorch.org to get the right
> `--index-url` flag. Training is ~10–50× faster on GPU.

> **RDKit:** if `pip install rdkit` fails on your platform, install it via conda:
> `conda install -c conda-forge rdkit` and activate that environment instead.

### Install the package (editable)

```bash
pip install -e .
```

### Verify the install

```bash
pytest tests/ -q
```

Expected: all tests pass in under a minute.

---

## Hardware requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| Disk (raw ZINC data) | 7 GB (1 shard) | 1.7 TB (all 250 shards) |
| Disk (compact cache) | 90 MB (1 shard) | ~22 GB (all 250 shards) |
| RAM | 2 GB | 64 GB+ (to preload many shards) |
| GPU | optional | strongly recommended for training |

Preprocessing converts each ~6.8 GB raw shard into a ~90 MB compact cache
(drops the density matrix, never used in training). You pay that I/O cost once;
every subsequent training run loads the compact cache in seconds.

---

## Dataset layout

The raw ZINC data directory must have this structure:

```
zinc-250k/
├── subsets/          # shard CSVs: subset_000.csv, subset_001.csv, ...
├── geometries/       # conformer pickles: coords_000.pkl, coords_001.pkl, ...
└── results/          # quantum HDF5: results_000.h5, results_001.h5, ...
```

Preprocessing reads from this layout and writes compact `.pt` files to a
separate `compact_cache/` directory you choose.

---

## Step 1 — Preprocess (once)

Convert raw shards into compact, scaffold-keyed caches. This step is
**resumable**: shards with a valid existing cache are skipped automatically.

```bash
# Smoke-test: one shard (~90 MB, a few minutes)
bash scripts/preprocess.sh zinc-250k zinc-250k/compact_cache "0"

# First 10 shards (~900 MB, good for initial training experiments)
bash scripts/preprocess.sh zinc-250k zinc-250k/compact_cache "0-9"

# All 250 shards (~22 GB, hours on first run; instant on re-run)
bash scripts/preprocess.sh zinc-250k zinc-250k/compact_cache "0-249"
```

If interrupted, re-run the same command — completed shards are skipped.

To force re-extraction (e.g. after a corrupt write):
```bash
bash scripts/preprocess.sh zinc-250k zinc-250k/compact_cache "12,37" --overwrite
```

What preprocessing does:
- Loads each shard through the existing extractor (drops the `dm` density matrix)
- Attaches a globally stable Murcko scaffold key to each molecule
- Saves `compact_cache/shard_NNN.pt` (versioned, validated on load)

---

## Step 2 — Train

Read the compact cache and produce trained backbone checkpoints.

```bash
# Train on first 10 shards (default output: runs/validate_scaled/)
bash scripts/train.sh zinc-250k/compact_cache "0-9"

# Train on all 250 shards, custom output directory
bash scripts/train.sh zinc-250k/compact_cache "0-249" runs/train_250k

# Single shard smoke-test
bash scripts/train.sh zinc-250k/compact_cache "0" runs/train_smoke
```

The script:
1. Verifies all requested shard `.pt` files exist (errors early if any are missing)
2. Builds a temp config from `configs/validate_scaled.yaml` with your shard range
3. Calls `python -m qchem_gnn.validation --config` to train 3 seeds × 1 arm
4. Runs ESOL downstream probes (mlp_head + finetune) after each seed
5. Writes a report to `<output_dir>/report.json`

Backbone checkpoints are saved to `<output_dir>/`:
```
runs/validate_scaled/
├── quantum_scaffold_s0.pt    # backbone, seed 0
├── quantum_scaffold_s1.pt    # backbone, seed 1
├── quantum_scaffold_s2.pt    # backbone, seed 2
└── report.json               # downstream MAE, R², per probe and seed
```

**Holdout:** scaffold-disjoint (`k: 10`) — ~1/10 of scaffolds are withheld from
pretraining, a stricter leakage-free split than random holdout.

### Resuming an interrupted run

Training writes a rolling checkpoint `{output_dir}/{arm}_s{seed}.ckpt.pt` every
10 epochs (configurable). To continue a killed run:

```bash
RESUME=true bash scripts/train.sh zinc-250k/compact_cache "0-9"
```

Resume refuses if structural hyperparameters changed since the checkpoint was
written. Only raising `epochs` (extending training) is allowed. `--overwrite`
ignores and deletes any checkpoint, restarting from epoch 0.

### Advanced: override config directly

For fine-grained control (custom seeds, arms, probes) edit
`configs/validate_scaled.yaml` and call the validation harness directly:

```bash
python -m qchem_gnn.validation --config configs/validate_scaled.yaml
python -m qchem_gnn.validation --config configs/validate_scaled.yaml --overwrite
```

---

## Step 3 — Adapt to a downstream task

Train a lightweight adapter on top of any frozen backbone. Three adapter
methods are available: `mlp_head` (frozen backbone, fast), `finetune`
(backbone updated at a lower lr), `engine` (side-structure, backbone untouched).

### Quick start

```bash
cp configs/adapt_example.yaml configs/adapt_mydata.yaml
```

Edit `configs/adapt_mydata.yaml`:

```yaml
command: adapt
method: mlp_head    # or: finetune, engine
backbone: runs/validate_scaled/quantum_scaffold_s0.pt

task: regression

dataset:
  csv: data/my_property.csv
  smiles_col: smiles
  targets: [my_property]

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

outputs:
  adapter: runs/mlp_head_mydata.pt
  report:  runs/mlp_head_mydata_metrics.json
```

Then run:
```bash
bash scripts/adapt.sh configs/adapt_mydata.yaml
```

Optional inline overrides (no YAML edit required):
```bash
bash scripts/adapt.sh configs/adapt_mydata.yaml training.epochs=200 training.lr=5e-4
```

### Worked example: bond dissociation energy (BDE-db)

Ready-to-run configs are provided for predicting homolytic bond dissociation
enthalpies (kcal/mol) from the BDE-db dataset (290K bonds, M06-2X/def2-TZVP).

```bash
# Download the data (~31 MB)
bash data/download_bde.sh

# Run all three methods
bash scripts/adapt.sh configs/adapt_mlp_head_bde.yaml
bash scripts/adapt.sh configs/adapt_finetune_bde.yaml
bash scripts/adapt.sh configs/adapt_engine_bde.yaml
```

Benchmark results on 484K bonds (80/9/10 split):

| Method | MAE (kcal/mol) | R² |
|--------|---------------|-----|
| mlp_head | 7.59 | 0.133 |
| finetune | 7.55 | 0.137 |
| engine | 7.89 | 0.087 |

### Solubility benchmark (Delaney ESOL)

```bash
bash scripts/adapt.sh configs/adapt_mlp_head_solubility.yaml
bash scripts/adapt.sh configs/adapt_finetune_solubility.yaml
```

---

## Step 4 — Export embeddings (optional)

Export one embedding vector per molecule from any backbone checkpoint:

```bash
bash scripts/infer.sh runs/validate_scaled/quantum_scaffold_s0.pt embeddings.npy
```

Output: NumPy `.npy` file with shape `[N_molecules, hidden_dim]`.

---

## Step 5 — Scaling sweep (optional)

Measure how downstream MAE changes as you increase pretraining data volume:

```bash
# Train at 1, 10, and 50 shards and print a MAE-vs-scale table
bash scripts/scaling_sweep.sh zinc-250k zinc-250k/compact_cache "1 10 50"
```

The sweep preprocesses (skip-if-exists), trains at each scale with 3 seeds,
runs ESOL probes, and prints:

```
 scale     method               arm   mae_mean
     1   mlp_head  quantum_scaffold     0.9200
    10   mlp_head  quantum_scaffold     0.8700
    50   mlp_head  quantum_scaffold     0.8100
```

To aggregate from existing reports manually:
```bash
python scripts/aggregate_scaling.py \
  1=runs/scaling_s1/report.json \
  10=runs/scaling_s10/report.json \
  50=runs/scaling_s50/report.json
```

---

## Full end-to-end example (10 shards)

```bash
# 0. Set up and activate environment
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas pyyaml scikit-learn scipy h5py torch torch-geometric rdkit
pip install -e .

# 1. Preprocess 10 shards (~900 MB, ~10 minutes)
bash scripts/preprocess.sh zinc-250k zinc-250k/compact_cache "0-9"

# 2. Train backbone from compact cache (3 seeds, ~hours)
bash scripts/train.sh zinc-250k/compact_cache "0-9"

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

# 5. Adapt to a new property (e.g. solubility)
bash scripts/adapt.sh configs/adapt_finetune_solubility.yaml
```

---

## Troubleshooting

**Missing shard files when running `train.sh`**
Run `bash scripts/preprocess.sh` first. `train.sh` checks for all requested
shard `.pt` files before starting and errors with the exact missing paths.

**`ValueError: shard cache version mismatch`**
A cache file was written by an older version. Re-extract it:
```bash
bash scripts/preprocess.sh zinc-250k zinc-250k/compact_cache "N" --overwrite
```

**`CheckpointMismatchError` when resuming**
The config changed since the checkpoint was written. Either revert the config
change, or restart from scratch with `--overwrite`.

**`ValueError: scaffold_hash_holdout produced an empty side`**
The `k` value in `holdout.k` is routing all molecules to one side. The default
`k: 10` works for any dataset with ≥ 10 distinct scaffolds.

**Training OOM (out of memory)**
Reduce `batch_size` in `configs/validate_scaled.yaml` (try 8), or reduce
`subset_ids` to fewer shards.

**`import h5py` error**
```bash
pip install h5py
```

**RDKit install fails**
```bash
conda install -c conda-forge rdkit
```
