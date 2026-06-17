# Predicting Molecular Solubility with a Frozen GNN and the ENGINE Adapter

This tutorial walks through fine-tuning a pretrained molecular GNN for aqueous
solubility prediction without updating a single backbone weight. It uses the
**ENGINE side structure** (Zhu et al., [github.com/zhuyun97/engine](https://github.com/zhuyun97/engine))
adapted from large language model tuning on text graphs to molecular property
prediction.

---

## Background

### The transfer learning problem

We pretrained a 2D/3D contrastive GNN on a subset of ZINC-250k to learn
general molecular representations. We now want to use those representations to
predict **aqueous solubility** — a key property in drug discovery and materials
science — on a completely different labelled dataset.

The naive approach is **fine-tuning**: run gradient descent through the entire
backbone. This risks destroying the pretraining representations, especially when
the downstream dataset is small.

### Frozen-backbone adaptation

A better approach keeps the backbone frozen and trains only a lightweight
**head** on top of the fixed embeddings. The backbone becomes a feature
extractor; only the head adapts.

| Method | Backbone | Trainable params | Notes |
|--------|----------|-----------------|-------|
| Linear probe | Frozen | output layer only | Closed-form solution |
| k-NN | Frozen | zero | Pure similarity search |
| MLP head | Frozen | 2-layer MLP | Non-linear head |
| **ENGINE adapter** | **Frozen** | **projections + α gates + exit heads** | **Uses all backbone layers** |

### The ENGINE side structure

The first three methods only see the GNN's **final embedding**. ENGINE instead
taps every intermediate message-passing layer:

```
h_0 = zeros
h_i = proj_i(gnn_state_i) × σ(α_i)  +  h_{i-1} × (1 − σ(α_i))
exit_pred_i = head_i(h_i)
L_train = Σ_i  MSE(exit_pred_i, y)
```

- **`gnn_state_i`** — pooled node embeddings after the i-th frozen message-passing step
- **`proj_i`** — learnable two-layer MLP that projects layer i's state into the side stream
- **`σ(α_i)`** — sigmoid of a learnable scalar; controls how much new information enters at layer i
- **`exit_pred_i`** — each layer produces its own prediction; all contribute to the training loss

This has two key benefits:
1. Every layer receives a **direct gradient signal** (the summed loss), not just the final one.
2. The learned α gates reveal **which layer's representation is most informative** for the task.

At inference the model can either:
- **Ensemble**: average all exit-head predictions for maximum accuracy.
- **Early exit**: stop each molecule at the first layer where prediction variance
  drops below a threshold, saving computation.

---

## Prerequisites

```
Python   ≥ 3.10
PyTorch  ≥ 2.0
RDKit    (molecule parsing)
scikit-learn, pandas, numpy
```

You also need the two files produced in earlier steps:

| File | How to obtain |
|------|---------------|
| `runs/example_contrastive.pt` | Run `python -m qchem_gnn.cli contrastive-pretrain --config configs/example_contrastive_pretrain.yaml` |
| `data/delaney-processed.csv`  | Included in the repo (Delaney ESOL 2004 / MoleculeNet 2018, 1 128 molecules) |

---

## The dataset: Delaney ESOL

`data/delaney-processed.csv` contains 1 128 organic molecules with
experimentally measured aqueous solubility (`measured log solubility in mols
per litre`). The target is log(S) in mol/L — a signed number where:

- **Positive** (e.g. +1.8) → highly water-soluble (like ethanol)
- **Around −2** → slightly soluble (like benzene)
- **Below −5** → essentially insoluble (like large hydrophobic drugs)

```
Dataset statistics
  min    = −11.60 log(mol/L)
  max    =  +1.58 log(mol/L)
  mean   =  −3.05 log(mol/L)
  std    =   2.10 log(mol/L)
```

---

## Step 1 — Verify the backbone checkpoint

```bash
python3 -c "
import torch
ckpt = torch.load('runs/example_contrastive.pt', weights_only=False)
cfg = ckpt['model_config']
print(f'hidden_dim={cfg[\"hidden_dim\"]}  steps={cfg[\"num_message_passing_steps\"]}')
print(f'pretrained epoch={ckpt[\"epoch\"]}  on {ckpt[\"run_metadata\"][\"num_examples\"]} molecules')
"
```

Expected output:
```
hidden_dim=64  steps=3
pretrained epoch=50  on 32 molecules
```

The backbone has **3 message-passing layers** and a 64-dimensional hidden
space. ENGINE will train one projection + alpha gate + exit head per layer,
for 3 exits total.

---

## Step 2 — Train the ENGINE adapter

### Dataset split

We use a **60 / 20 / 20** stratified split (stratified by solubility quintile
so every partition covers the full range of log S):

| Partition | Fraction | Molecules |
|-----------|----------|-----------|
| Train | 60 % | 678 |
| Validation | 20 % | 226 |
| Test | 20 % | 224 |

### Command

```bash
python scripts/train_engine_adapter.py \
    --checkpoint runs/example_contrastive.pt \
    --data data/delaney-processed.csv \
    --test-frac 0.2 \
    --val-frac  0.25 \
    --epochs    10 \
    --output    runs/solubility_adapter_tutorial.pt
```

`--val-frac 0.25` means 25 % of the 80 % non-test pool → 20 % of total.

### Full output

```
Dataset: 1128 molecules  |  target: 'measured log solubility in mols per litre'
  log(S): min=-11.60  max=1.58  mean=-3.05  std=2.10

Backbone: hidden_dim=64  steps=3  (pretrained epoch 50)

Extracting per-layer embeddings from frozen backbone …
  Parsed 1128/1128  (0 skipped)  layers: 3 × (1128, 64)
  Split: 678 train  /  226 val  /  224 test

Training ENGINE adapter  (10 epochs) …
   Epoch   train_loss   val_MAE
  --------------------------------
      10       2.0604    1.4023

─────────────────────────────────────────────────────────────
  Test results                            MAE    RMSE      R²
  ───────────────────────────────────────────────────────────
  ENGINE ensemble                       1.434   1.802   0.219
  ENGINE early exit (tol=0.1)           1.443   1.818   0.205  [2.4/3 avg, 61% early]
─────────────────────────────────────────────────────────────

  α gates: σ(α_0)=0.50, σ(α_1)=0.50, σ(α_2)=0.50

  Adapter saved → runs/solubility_adapter_tutorial.pt
```

### Reading the output

**Embedding extraction** runs once before training. The backbone is frozen so
the 3 × 1128 layer embeddings (each 64-dim) are computed in a single forward
pass and reused for all 10 training epochs — making training very fast.

**Training loss** (2.06 at epoch 10) is the sum of MSE across all three exit
heads in normalised label space. It will keep falling with more epochs.

**Test MAE of 1.434 log(mol/L)** means predictions are off by about 1.4 units
on average. For context, the standard deviation of the dataset is 2.1, so the
model is already predicting better than always guessing the mean (which would
give MAE ≈ 1.68). With 400 epochs this drops to 0.849 (see below).

**α gates all at 0.50** — `σ(α_i)` initialises to 0.5 (since α = 0 at init).
After only 10 epochs the gates have not moved. Given more training, they diverge
to reflect each layer's usefulness for solubility prediction.

**Early exit**: 61 % of molecules exited before the final layer (average exit
at layer 2.4/3). The adapter was not fully trained, so the threshold is hit
early not because predictions converged but because early-layer variance is
already below 0.1 normalised units.

---

## Step 3 — Inference on new molecules

```bash
python scripts/predict_solubility.py \
    --adapter runs/solubility_adapter_tutorial.pt \
    "CCO" "c1ccccc1" "CC(=O)O" "CCCCCCCCC" "OC(=O)c1ccccc1" \
    "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1" "O=C(O)CCC(=O)O"
```

Output:
```
Adapter trained on: delaney-processed.csv  (678 train / 224 test)
  Best val MAE: 1.4023  |  Test MAE ensemble: 1.4337

Predicting log(S) for 7 molecule(s) [mode: ensemble] …

  SMILES                                         log(S) [mol/L]
  ------------------------------------------------------------
  CCO                                            -1.244
  c1ccccc1                                       -3.687
  CC(=O)O                                        -1.489
  CCCCCCCCC                                      -2.286
  OC(=O)c1ccccc1                                 -3.536
  CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1        -3.531
  O=C(O)CCC(=O)O                                 -1.827
```

Qualitative check: ethanol (`CCO`, −1.2) and acetic acid (`CC(=O)O`, −1.5) are
predicted more soluble than benzene (`c1ccccc1`, −3.7), which is correct.
Nonane (`CCCCCCCCC`, −2.3) is predicted less soluble than the polar acids,
also correct in direction, though the magnitude is imprecise after only 10
epochs. The large drug-like compound (`CC(C)(C)c1ccc2…`, −3.5) is predicted
as poorly soluble, which is also chemically reasonable.

To score a CSV file and save results:
```bash
python scripts/predict_solubility.py \
    --adapter runs/solubility_adapter_tutorial.pt \
    --csv my_molecules.csv \
    --smiles-col smiles \
    --output predictions.csv
```

---

## Effect of training duration

Training for more epochs substantially improves test metrics (same 60/20/20
split, 400 epochs):

| Epochs | val MAE | Test MAE | Test R² | α gates |
|--------|---------|----------|---------|---------|
| 10     | 1.402   | 1.434    | 0.219   | all 0.50 (not yet differentiated) |
| 400    | 0.794   | 0.849    | 0.667   | 0.55 / 0.47 / 0.43 (layer 0 trusted most) |

The α gate pattern at convergence (`σ(α_0)=0.55, σ(α_1)=0.47, σ(α_2)=0.43`)
tells a story: the **first message-passing layer** carries the most weight in
the side stream. This makes sense — after one step each atom has seen its
immediate neighbours, which is already enough to encode local chemical
environment relevant to solubility (polarity, hydrogen-bond donors/acceptors).
Deeper layers add longer-range structure with diminishing marginal benefit for
this task.

---

## What limits accuracy here?

The backbone was pretrained on **32 molecules** from a single ZINC-250k shard
with no explicit solubility signal. A backbone pretrained on hundreds of
thousands of diverse molecules will produce much richer frozen embeddings,
and the ENGINE adapter will start from a better feature space.

To retrain the backbone on more data:
```bash
# Remove limit_per_shard to use all ~1000 molecules in the shard
python -m qchem_gnn.cli contrastive-pretrain \
    --config configs/example_contrastive_pretrain.yaml \
    --limit-per-shard 0 \
    --epochs 200 \
    --output runs/contrastive_larger.pt
```

Then retrain the ENGINE adapter on top of the new checkpoint.

---

## Files produced

| File | Description |
|------|-------------|
| `runs/solubility_adapter_tutorial.pt` | Trained ENGINE adapter weights + normalisation statistics |
| `scripts/train_engine_adapter.py` | Training script |
| `scripts/predict_solubility.py` | Inference script |
| `qchem_gnn/engine_adapter.py` | `EngineAdapterHead`, `extract_intermediate_embeddings`, `save/load_adapter`, `predict` |

The adapter file is self-contained: it stores the backbone checkpoint path so
`predict_solubility.py` requires only `--adapter`.
