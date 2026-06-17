#!/usr/bin/env python3
"""
Train an MLP head on top of a frozen pretrained GNN backbone.

The backbone GNN is never updated — its weights are fixed at the pretrained
values.  A small feedforward MLP is trained on the backbone's final molecule
embeddings to predict the downstream property.

Because the backbone is frozen, all molecule embeddings are computed once
before training begins and reused every epoch, making this method very fast.

Usage:
  python scripts/train_mlp_head.py \\
      --checkpoint runs/example_contrastive.pt \\
      --data data/delaney-processed.csv \\
      --output runs/mlp_head_solubility.pt

Optional flags:
  --smiles-col     SMILES column name  (auto-detected if omitted)
  --target-col     Target column name  (auto-detected if omitted)
  --test-frac      0.2    held-out test fraction  (default 0.2)
  --val-frac       0.25   validation fraction of remaining data  (default 0.25)
  --epochs         300    maximum training epochs  (default 300)
  --lr             1e-3   learning rate  (default 1e-3)
  --hidden-dims    128 64 hidden layer widths  (default: 128 64)
  --dropout        0.1    dropout probability  (default 0.1)
  --batch-size     128    training minibatch size  (default 128)
  --patience       40     early-stop patience in epochs  (default 40)
  --seed           42     random seed  (default 42)
  --embed-batch    256    molecules per embedding batch  (default 256)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from qchem_gnn.adapters import embed_smiles, save_mlp_adapter, train_mlp_head
from qchem_gnn.checkpoint import load_checkpoint
from qchem_gnn.model import MolecularQuantumGNN


def _detect_columns(df: pd.DataFrame, smiles_col: str | None, target_col: str | None):
    if smiles_col is None:
        smiles_col = next(c for c in df.columns if c.lower() == "smiles")
    if target_col is None:
        # Prefer columns with "solubility" that are not SD or ESOL-computed
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
    """Split by property quintile for balanced coverage in every partition."""
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
    parser.add_argument("--checkpoint", required=True, help="Pretrained backbone .pt")
    parser.add_argument("--data",       required=True, help="CSV with SMILES + target")
    parser.add_argument("--output",     required=True, help="Save adapter to this path")
    parser.add_argument("--smiles-col", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--test-frac",  type=float,       default=0.2)
    parser.add_argument("--val-frac",   type=float,       default=0.25)
    parser.add_argument("--epochs",     type=int,         default=300)
    parser.add_argument("--lr",         type=float,       default=1e-3)
    parser.add_argument("--hidden-dims",type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout",    type=float,       default=0.1)
    parser.add_argument("--batch-size", type=int,         default=128)
    parser.add_argument("--patience",   type=int,         default=40)
    parser.add_argument("--seed",       type=int,         default=42)
    parser.add_argument("--embed-batch",type=int,         default=256)
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

    # ── Embed (one pass, frozen backbone) ─────────────────────────────────────
    print("\nExtracting frozen embeddings …")
    emb, valid_idx = embed_smiles(all_smiles, model, batch_size=args.embed_batch)
    y       = all_y[valid_idx]
    n_skip  = len(all_smiles) - len(valid_idx)
    print(f"  Parsed {len(valid_idx)}/{len(all_smiles)}  ({n_skip} skipped)  "
          f"→  embeddings {emb.shape}")

    # ── Split ──────────────────────────────────────────────────────────────────
    train_idx, val_idx, test_idx = _stratified_split(
        y, args.test_frac, args.val_frac, rng
    )
    print(f"  Split: {len(train_idx)} train  /  {len(val_idx)} val  "
          f"/  {len(test_idx)} test")

    X_tr  = emb[train_idx];  y_tr  = y[train_idx]
    X_val = emb[val_idx];    y_val = y[val_idx]
    X_te  = emb[test_idx];   y_te  = y[test_idx]

    # ── Train ──────────────────────────────────────────────────────────────────
    head, feat_mu, feat_sig, label_mu, label_sig, log = train_mlp_head(
        X_tr, y_tr, X_val, y_val, X_te, y_te,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
        print_every=50,
    )

    # ── Test results ───────────────────────────────────────────────────────────
    W = 30
    print(f"\n{'─' * (W + 28)}")
    print(f"  {'Test results':<{W}}  {'MAE':>6}  {'RMSE':>6}  {'R²':>6}")
    print(f"  {'─' * (W + 26)}")
    print(f"  {'MLP head':<{W}}  "
          f"{log['test_mae']:>6.3f}  {log['test_rmse']:>6.3f}  {log['test_r2']:>6.3f}")
    print(f"{'─' * (W + 28)}")
    print(f"\n  Best val MAE: {log['best_val_mae']:.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    save_mlp_adapter(
        path=Path(args.output),
        head=head,
        feat_mu=feat_mu,
        feat_sig=feat_sig,
        label_mu=label_mu,
        label_sig=label_sig,
        backbone_ckpt=str(ckpt_path.resolve()),
        training_info={
            "dataset":      str(Path(args.data).name),
            "target_col":   target_col,
            "n_train":      len(train_idx),
            "n_val":        len(val_idx),
            "n_test":       len(test_idx),
            "epochs":       args.epochs,
            "hidden_dims":  args.hidden_dims,
            "dropout":      args.dropout,
            "best_val_mae": log["best_val_mae"],
            "test_mae":     log["test_mae"],
            "test_rmse":    log["test_rmse"],
            "test_r2":      log["test_r2"],
        },
    )
    print(f"\n  Adapter saved → {args.output}")
    print(f"  Run inference:  python scripts/predict_property.py "
          f"--adapter {args.output} 'CCO' 'c1ccccc1'")


if __name__ == "__main__":
    main()
