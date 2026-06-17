# Getting Started: Environment Setup, Training, and Inference

This tutorial covers the full workflow from a fresh checkout to making
property predictions with a pretrained molecular GNN.

---

## 1. Prerequisites

- Python 3.13 or newer
- A CUDA-capable GPU is recommended for pretraining; CPU is fine for adaptation
  and inference on small datasets

---

## 2. Install the Virtual Environment

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install \
    numpy pandas torch pyyaml \
    rdkit scikit-learn scipy h5py pytest
```

> **Note:** If `rdkit` is not available via pip on your platform, install it
> through conda (`conda install -c conda-forge rdkit`) and keep the rest of
> the deps in the venv.

Verify the environment:

```bash
python -c "import torch, rdkit, qchem_gnn; print('OK', torch.__version__)"
```

Run the test suite to confirm everything is wired up:

```bash
python -m pytest tests -q
# Expected: 120 passed
```

---

## 3. Backbone Pretraining

The adaptation methods in this project all start from a **pretrained backbone
checkpoint**. If you already have one (e.g. `runs/example_contrastive.pt`),
skip to Section 4.

### 3a. Standard supervised pretraining

```bash
python -m qchem_gnn.cli train \
    --config configs/minimal_train.yaml
# Checkpoint saved to runs/minimal_train.pt
```

### 3b. Contrastive (2D/3D) pretraining (recommended)

Distills 3D conformer geometry into the 2D encoder. Produces representations
that transfer better to downstream property prediction.

```bash
python -m qchem_gnn.cli contrastive-pretrain \
    --config configs/minimal_contrastive_pretrain.yaml
# Checkpoint saved to runs/example_contrastive.pt (path set in the config)
```

### Overriding YAML values from the command line

Any key in the YAML can be overridden:

```bash
python -m qchem_gnn.cli train \
    --config configs/minimal_train.yaml \
    --epochs 100 \
    --output runs/my_backbone.pt
```

---

## 4. Downstream Adaptation

The `adapt` command trains a lightweight adapter on top of a frozen (or
fine-tuned) backbone for **any property CSV you provide**. Three methods are
available:

| Method | What updates | When to use |
|--------|-------------|-------------|
| `mlp_head` | Head only (backbone frozen) | Fast; good baseline |
| `finetune` | Backbone + head (dual LR) | Best accuracy; slower |
| `engine` | Per-layer side structure + alpha gates | Multi-exit inference |

All three share the same YAML schema and CLI entry point.

### 4a. Run a single adaptation

Edit one of the example configs in `configs/` to point at your checkpoint and
CSV, then:

```bash
python -m qchem_gnn.cli adapt configs/adapt_mlp_head_solubility.yaml
```

The config controls every hyperparameter:

```yaml
# configs/adapt_mlp_head_solubility.yaml
command: adapt
method: mlp_head
backbone: runs/example_contrastive.pt   # pretrained checkpoint
task: regression                         # or: classification

dataset:
  csv: data/delaney-processed.csv
  smiles_col: auto        # auto-detect the SMILES column
  targets: auto           # auto-detect the target column(s)

adapter:
  hidden_dims: [128, 64]
  dropout: 0.1

training:
  epochs: 300
  lr: 1.0e-3
  batch_size: 128
  patience: 40            # early stopping
  seed: 42

split:
  test_frac: 0.2
  val_frac: 0.25          # fraction of the non-test pool
  seed: 42
  stratify: true

outputs:
  adapter: runs/mlp_head_solubility.pt
  report: runs/mlp_head_metrics.json
```

Override individual values without editing the file:

```bash
python -m qchem_gnn.cli adapt configs/adapt_mlp_head_solubility.yaml \
    -O training.epochs=500 training.lr=5e-4
```

### 4b. Fine-tuning (backbone + head jointly)

```bash
python -m qchem_gnn.cli adapt configs/adapt_finetune_solubility.yaml
```

Fine-tuning config adds `head_lr` / `backbone_lr` / `grad_clip`:

```yaml
training:
  epochs: 200
  head_lr: 1.0e-3
  backbone_lr: 5.0e-5
  grad_clip: 1.0
  patience: 30
```

### 4c. ENGINE adapter

```bash
python -m qchem_gnn.cli adapt configs/adapt_engine_solubility.yaml
```

ENGINE trains per-exit heads at every message-passing layer, learning a
learned gate (α) for each. At inference you can choose ensemble or
early-exit mode (see Section 5).

### 4d. Adapt to your own dataset

Create a CSV with at least one SMILES column and one property column:

```
smiles,logP
CCO,-0.31
c1ccccc1,2.13
...
```

Then write a minimal config:

```yaml
command: adapt
method: mlp_head
backbone: runs/example_contrastive.pt
task: regression

dataset:
  csv: data/my_property.csv
  smiles_col: smiles         # or: auto
  targets: [logP]            # list one or more column names, or: auto

adapter:
  hidden_dims: [128, 64]
  dropout: 0.1

training:
  epochs: 200
  lr: 1.0e-3
  batch_size: 64
  patience: 30
  seed: 42

split:
  test_frac: 0.2
  val_frac: 0.25
  seed: 42
  stratify: true

outputs:
  adapter: runs/my_property_adapter.pt
  report: runs/my_property_metrics.json
```

```bash
python -m qchem_gnn.cli adapt configs/adapt_my_property.yaml
```

Unparseable SMILES are silently skipped with a reported count; the run never
crashes on bad inputs.

### 4e. Multi-target regression

List several column names under `targets`:

```yaml
dataset:
  targets: [logP, MW, TPSA]
```

The adapter head grows to `len(targets)` output units automatically. Metrics
(MAE, RMSE, R²) are reported per target and as a macro average.

### 4f. Binary classification

```yaml
task: classification
dataset:
  targets: [active]     # column of 0/1 labels
```

Loss switches to BCE-with-logits; predictions are returned as probabilities;
metrics reported are AUC, accuracy, and F1.

---

## 5. Hyperparameter Sweeps

Add a `sweep:` block to any adaptation config. The `grid` maps dotted config
keys to lists of values; the Cartesian product is run automatically and a CSV
comparison table is written.

```yaml
# configs/adapt_engine_epoch_sweep.yaml
command: adapt
method: engine
backbone: runs/example_contrastive.pt
task: regression
dataset:
  csv: data/delaney-processed.csv
  smiles_col: auto
  targets: auto
adapter:
  hidden_dims: [128, 64]
  dropout: 0.1
training:
  lr: 1.0e-3
  batch_size: 64
  patience: 30
  seed: 42
split:
  test_frac: 0.2
  val_frac: 0.25
  seed: 42
  stratify: true

sweep:
  grid:
    training.epochs: [10, 50, 100, 200, 400]
  report: runs/engine_epoch_sweep.csv
```

```bash
python -m qchem_gnn.cli adapt configs/adapt_engine_epoch_sweep.yaml
# Runs 5 cells; prints a comparison table; writes runs/engine_epoch_sweep.csv
```

You can sweep over any combination:

```yaml
sweep:
  grid:
    training.epochs: [100, 200]
    adapter.dropout: [0.0, 0.1, 0.2]
  report: runs/sweep.csv
# → 6 cells (2 × 3)
```

---

## 6. Inference

After training, the adapter `.pt` file is self-contained — it carries the
backbone checkpoint path, all normalisation statistics, and the adapter type.

### Predict from SMILES strings

```bash
python scripts/predict_property.py \
    --adapter runs/mlp_head_solubility.pt \
    "CCO" "c1ccccc1" "CC(=O)O" "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
```

Example output:

```
Adapter type : MLP head (frozen backbone)
Trained on   : delaney-processed.csv  (678 train / 224 test)
Target       : measured log solubility in mols per litre
Best val MAE : ?  Test MAE: ?  R²: ?

Scoring 4 molecule(s) …

  SMILES                          measured log solubility in mols per litre
  ---------------------------------------------------------------------------
  CCO                             +0.673
  c1ccccc1                        -1.797
  CC(=O)O                         -0.829
  CN1C=NC2=C1C(=O)N(C(=O)N2C)C    -1.044
```

### Predict from a CSV

```bash
python scripts/predict_property.py \
    --adapter runs/mlp_head_solubility.pt \
    --csv data/delaney-processed.csv \
    --smiles-col smiles \
    --output predictions.csv
```

### ENGINE-specific inference modes

```bash
# Ensemble (default): weighted average across all exits
python scripts/predict_property.py \
    --adapter runs/engine_solubility.pt \
    --mode ensemble \
    "CCO" "c1ccccc1"

# Early-exit: stop at the first exit where prediction std < threshold
python scripts/predict_property.py \
    --adapter runs/engine_solubility.pt \
    --mode early_exit \
    --exit-tol 0.05 \
    "CCO" "c1ccccc1"
```

### Predict from Python

```python
from qchem_gnn.adapt import predict_smiles

smiles = ["CCO", "c1ccccc1", "invalid!!"]
preds, valid_idx = predict_smiles(smiles, "runs/mlp_head_solubility.pt")

# preds: np.ndarray [N_valid, n_targets]
# valid_idx: list of positions in smiles that were parseable
for i, p in zip(valid_idx, preds):
    print(smiles[i], p)
```

For ENGINE mode:

```python
preds, valid_idx = predict_smiles(
    smiles,
    "runs/engine_solubility.pt",
    mode="early_exit",
    exit_tol=0.05,
)
```

---

## 7. Diagnostic Evaluation on the Pretraining Dataset

The `downstream` command evaluates a checkpoint's learned representations on
the **native pretraining dataset** (not an external CSV). It is a research
diagnostic, not an adaptation tool:

```bash
python -m qchem_gnn.cli downstream \
    --checkpoint runs/example_contrastive.pt \
    --kind linear-probe
```

Available modes: `linear-probe`, `fine-tune`, `morgan-baseline`,
`sample-efficiency`. Results are printed to stdout only; nothing is saved.

---

## 8. Quick Reference

```bash
# Pretrain backbone
python -m qchem_gnn.cli contrastive-pretrain --config configs/minimal_contrastive_pretrain.yaml

# Adapt (frozen MLP head)
python -m qchem_gnn.cli adapt configs/adapt_mlp_head_solubility.yaml

# Adapt (fine-tune backbone + head)
python -m qchem_gnn.cli adapt configs/adapt_finetune_solubility.yaml

# Adapt (ENGINE side structure)
python -m qchem_gnn.cli adapt configs/adapt_engine_solubility.yaml

# Sweep epochs
python -m qchem_gnn.cli adapt configs/adapt_engine_epoch_sweep.yaml

# Predict (command line)
python scripts/predict_property.py --adapter runs/mlp_head_solubility.pt "CCO"

# Predict (Python API)
from qchem_gnn.adapt import predict_smiles
preds, valid_idx = predict_smiles(["CCO"], "runs/mlp_head_solubility.pt")

# Tests
python -m pytest tests -q
```
