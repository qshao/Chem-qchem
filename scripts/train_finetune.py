#!/usr/bin/env python3
"""
Fine-tune a pretrained GNN backbone + MLP head jointly for property prediction.

Unlike the MLP head and ENGINE adapters, fine-tuning updates the backbone GNN
weights alongside the head.  The backbone uses a much lower learning rate
(backbone_lr) to preserve the pretraining representation while adapting it to
the downstream task.  Gradient clipping prevents exploding gradients.

Because the backbone is updated each epoch, molecule graphs must be forwarded
through it at every training step.  Training is therefore slower than the
frozen-backbone methods but can recover task-specific signal that the
pretraining stage did not learn.

Usage:
  python scripts/train_finetune.py \\
      --checkpoint runs/example_contrastive.pt \\
      --data data/delaney-processed.csv \\
      --output runs/finetune_solubility.pt

Optional flags:
  --smiles-col     SMILES column name  (auto-detected if omitted)
  --target-col     Target column name  (auto-detected if omitted)
  --test-frac      0.2    held-out test fraction  (default 0.2)
  --val-frac       0.25   validation fraction of remaining data  (default 0.25)
  --epochs         200    maximum training epochs  (default 200)
  --head-lr        1e-3   learning rate for the MLP head  (default 1e-3)
  --backbone-lr    5e-5   learning rate for the backbone  (default 5e-5)
  --hidden-dims    128 64 hidden layer widths  (default: 128 64)
  --dropout        0.1    dropout probability  (default 0.1)
  --batch-size     64     training minibatch size  (default 64)
  --patience       30     early-stop patience in epochs  (default 30)
  --grad-clip      1.0    gradient clipping max norm  (default 1.0)
  --seed           42     random seed  (default 42)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from qchem_gnn.adapters import save_finetune_adapter, train_finetune
from qchem_gnn.checkpoint import load_checkpoint
from qchem_gnn.graph import GraphData, build_graph_from_smiles
from qchem_gnn.model import MolecularQuantumGNN


def _detect_columns(df: pd.DataFrame, smiles_col: str | None, target_col: str | None):
    if smiles_col is None:
        smiles_col = next(c for c in df.columns if c.lower() == "smiles")
    if target_col is None:
        candidates = [
            c for c in df.columns
            if "solubility" in c.lower()
            and "sd" not in c.lower()
            and "esol" not in c.lower()
        ]
        target_col = candidates[0] if candidates else df.columns[-1]
    return smiles_col, target_col


def _stratified_split(
    y: np.ndarray,
    test_frac: float,
    val_frac: float,
    rng: np.random.Generator,
    n_bins: int = 5,
) -> tuple[list[int], list[int], list[int]]:
    bounds = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
    bins   = np.digitize(y, bounds)

    train_val_idx: list[int] = []
    test_idx:      list[int] = []
    for q in range(n_bins):
        mask = np.where(bins == q)[0]
        rng.shuffle(mask)
        cut = max(1, int(len(mask) * test_frac))
        test_idx.extend(mask[:cut].tolist())
        train_val_idx.extend(mask[cut:].tolist())

    tv = np.array(train_val_idx)
    rng.shuffle(tv)
    val_cut   = max(1, int(len(tv) * val_frac))
    val_idx   = tv[:val_cut].tolist()
    train_idx = tv[val_cut:].tolist()
    return train_idx, val_idx, test_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--data",        required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--smiles-col",  default=None)
    parser.add_argument("--target-col",  default=None)
    parser.add_argument("--test-frac",   type=float,       default=0.2)
    parser.add_argument("--val-frac",    type=float,       default=0.25)
    parser.add_argument("--epochs",      type=int,         default=200)
    parser.add_argument("--head-lr",     type=float,       default=1e-3)
    parser.add_argument("--backbone-lr", type=float,       default=5e-5)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout",     type=float,       default=0.1)
    parser.add_argument("--batch-size",  type=int,         default=64)
    parser.add_argument("--patience",    type=int,         default=30)
    parser.add_argument("--grad-clip",   type=float,       default=1.0)
    parser.add_argument("--seed",        type=int,         default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(args.data)
    smiles_col, target_col = _detect_columns(df, args.smiles_col, args.target_col)
    df = df[[smiles_col, target_col]].dropna()
    all_smiles = df[smiles_col].tolist()
    all_y      = df[target_col].to_numpy(dtype=np.float32)

    print(f"\nDataset: {len(df)} molecules  |  target: '{target_col}'")
    print(f"  values: min={all_y.min():.2f}  max={all_y.max():.2f}  "
          f"mean={all_y.mean():.2f}  std={all_y.std():.2f}")

    # ── Backbone ──────────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    ckpt  = load_checkpoint(ckpt_path)
    model = MolecularQuantumGNN(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    h_dim = ckpt["model_config"]["hidden_dim"]
    print(f"\nBackbone: hidden_dim={h_dim}  "
          f"steps={ckpt['model_config']['num_message_passing_steps']}  "
          f"(pretrained epoch {ckpt.get('epoch')})")

    # ── Build graphs (parsed once, forwarded each epoch) ──────────────────────
    print("\nBuilding molecule graphs …")
    valid_idx: list[int] = []
    graphs:    list[GraphData] = []
    for i, smi in enumerate(all_smiles):
        try:
            graphs.append(build_graph_from_smiles(smi))
            valid_idx.append(i)
        except Exception:
            pass
    y = all_y[valid_idx]
    n_skip = len(all_smiles) - len(valid_idx)
    print(f"  Parsed {len(valid_idx)}/{len(all_smiles)}  ({n_skip} skipped)")

    # ── Split ──────────────────────────────────────────────────────────────────
    train_idx, val_idx, test_idx = _stratified_split(
        y, args.test_frac, args.val_frac, rng
    )
    print(f"  Split: {len(train_idx)} train  /  {len(val_idx)} val  "
          f"/  {len(test_idx)} test")

    graphs_tr  = [graphs[i] for i in train_idx]
    graphs_val = [graphs[i] for i in val_idx]
    graphs_te  = [graphs[i] for i in test_idx]
    y_tr  = y[train_idx]
    y_val = y[val_idx]
    y_te  = y[test_idx]

    # ── Fine-tune ──────────────────────────────────────────────────────────────
    model, head, label_mu, label_sig, log = train_finetune(
        model, graphs_tr, y_tr, graphs_val, y_val, graphs_te, y_te,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        epochs=args.epochs,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        batch_size=args.batch_size,
        patience=args.patience,
        grad_clip=args.grad_clip,
        seed=args.seed,
        print_every=50,
    )

    # ── Test results ───────────────────────────────────────────────────────────
    W = 30
    print(f"\n{'─' * (W + 28)}")
    print(f"  {'Test results':<{W}}  {'MAE':>6}  {'RMSE':>6}  {'R²':>6}")
    print(f"  {'─' * (W + 26)}")
    print(f"  {'Fine-tune (backbone + head)':<{W}}  "
          f"{log['test_mae']:>6.3f}  {log['test_rmse']:>6.3f}  {log['test_r2']:>6.3f}")
    print(f"{'─' * (W + 28)}")
    print(f"\n  Best val MAE: {log['best_val_mae']:.4f}")
    print(f"  head_lr={args.head_lr}  backbone_lr={args.backbone_lr}  "
          f"grad_clip={args.grad_clip}")

    # ── Save ───────────────────────────────────────────────────────────────────
    save_finetune_adapter(
        path=Path(args.output),
        model=model,
        head=head,
        label_mu=label_mu,
        label_sig=label_sig,
        model_config=ckpt["model_config"],
        backbone_ckpt=str(ckpt_path.resolve()),
        training_info={
            "dataset":       str(Path(args.data).name),
            "target_col":    target_col,
            "n_train":       len(train_idx),
            "n_val":         len(val_idx),
            "n_test":        len(test_idx),
            "epochs":        args.epochs,
            "hidden_dims":   args.hidden_dims,
            "dropout":       args.dropout,
            "head_lr":       args.head_lr,
            "backbone_lr":   args.backbone_lr,
            "grad_clip":     args.grad_clip,
            "best_val_mae":  log["best_val_mae"],
            "test_mae":      log["test_mae"],
            "test_rmse":     log["test_rmse"],
            "test_r2":       log["test_r2"],
        },
    )
    print(f"\n  Adapter saved → {args.output}")
    print(f"  Run inference:  python scripts/predict_property.py "
          f"--adapter {args.output} 'CCO' 'c1ccccc1'")


if __name__ == "__main__":
    main()
