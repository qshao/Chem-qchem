#!/usr/bin/env python3
"""
Train an ENGINE side adapter on top of a frozen pretrained GNN for
aqueous solubility prediction, then save the adapter for inference.

The backbone GNN weights are never updated. Only the ENGINE side structure
(per-layer projections, alpha gates, exit heads) is trained.

Downloaded Delaney ESOL dataset (included in data/delaney-processed.csv):
  1128 molecules, measured log solubility in mol/L
  Wu et al. MoleculeNet 2018 / Delaney ESOL 2004

Usage:
  python scripts/train_engine_adapter.py \\
      --checkpoint runs/example_contrastive.pt \\
      --data data/delaney-processed.csv \\
      --output runs/solubility_adapter.pt

Optional flags:
  --smiles-col   SMILES column name  (auto-detected if omitted)
  --target-col   Target column name  (auto-detected if omitted)
  --test-frac    0.2   held-out test fraction (default 0.2)
  --val-frac     0.1   validation fraction of training data (default 0.1)
  --epochs       400   training epochs (default 400)
  --lr           3e-3  learning rate (default 3e-3)
  --seed         42    random seed (default 42)
  --batch        256   embedding batch size (default 256)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from qchem_gnn.checkpoint import load_checkpoint
from qchem_gnn.engine_adapter import (
    EngineAdapterHead,
    extract_intermediate_embeddings,
    save_adapter,
)
from qchem_gnn.model import MolecularQuantumGNN


def _detect_columns(df: pd.DataFrame, smiles_col: str | None, target_col: str | None):
    if smiles_col is None:
        smiles_col = next(c for c in df.columns if c.lower() == "smiles")
    if target_col is None:
        target_col = next(
            c for c in df.columns
            if "solubility" in c.lower() and "sd" not in c.lower() and "esol" not in c.lower()
        )
    return smiles_col, target_col


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smiles-col", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--val-frac",  type=float, default=0.1)
    parser.add_argument("--epochs",    type=int,   default=400)
    parser.add_argument("--lr",        type=float, default=3e-3)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--batch",     type=int,   default=256)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # --- Data ---
    df = pd.read_csv(args.data)
    smiles_col, target_col = _detect_columns(df, args.smiles_col, args.target_col)
    df = df[[smiles_col, target_col]].dropna()
    all_smiles = df[smiles_col].tolist()
    all_y = df[target_col].to_numpy(dtype=np.float32)
    print(f"\nDataset: {len(df)} molecules  |  target: '{target_col}'")
    print(f"  log(S): min={all_y.min():.2f}  max={all_y.max():.2f}  "
          f"mean={all_y.mean():.2f}  std={all_y.std():.2f}")

    # --- Backbone ---
    ckpt_path = Path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path)
    model = MolecularQuantumGNN(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    h_dim   = ckpt["model_config"]["hidden_dim"]
    n_steps = ckpt["model_config"]["num_message_passing_steps"]
    print(f"\nBackbone: hidden_dim={h_dim}  steps={n_steps}  "
          f"(pretrained epoch {ckpt.get('epoch')})")

    # --- Extract layer embeddings (one pass, frozen backbone) ---
    print("\nExtracting per-layer embeddings from frozen backbone …")
    layer_embs, _, valid_idx = extract_intermediate_embeddings(
        all_smiles, model, batch_size=args.batch
    )
    y = all_y[valid_idx]
    n_skip = len(all_smiles) - len(valid_idx)
    print(f"  Parsed {len(valid_idx)}/{len(all_smiles)}  ({n_skip} skipped)  "
          f"layers: {n_steps} × {layer_embs[0].shape}")

    # --- Stratified split (by solubility quintile) ---
    quintiles = np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
    train_val_idx: list[int] = []
    test_idx: list[int] = []
    for q in range(5):
        mask = np.where(quintiles == q)[0]
        rng.shuffle(mask)
        cut = max(1, int(len(mask) * args.test_frac))
        test_idx.extend(mask[:cut].tolist())
        train_val_idx.extend(mask[cut:].tolist())

    # Val from within train_val
    rng.shuffle(train_val_idx := np.array(train_val_idx))
    val_cut = max(1, int(len(train_val_idx) * args.val_frac))
    val_idx   = train_val_idx[:val_cut].tolist()
    train_idx = train_val_idx[val_cut:].tolist()

    print(f"  Split: {len(train_idx)} train  /  {len(val_idx)} val  "
          f"/  {len(test_idx)} test")

    # --- Per-layer z-score normalisation (fit on training set only) ---
    layer_scalers: list[tuple[np.ndarray, np.ndarray]] = []
    tr_tensors, val_tensors, te_tensors = [], [], []
    for emb in layer_embs:
        mu  = emb[train_idx].mean(0)
        sig = emb[train_idx].std(0).clip(1e-8)
        layer_scalers.append((mu, sig))
        tr_tensors.append(torch.as_tensor((emb[train_idx] - mu) / sig, dtype=torch.float32))
        val_tensors.append(torch.as_tensor((emb[val_idx]   - mu) / sig, dtype=torch.float32))
        te_tensors.append(torch.as_tensor((emb[test_idx]  - mu) / sig, dtype=torch.float32))

    # Normalise labels
    y_mu  = float(y[train_idx].mean())
    y_sig = float(y[train_idx].std()) or 1.0
    y_tr  = torch.as_tensor((y[train_idx] - y_mu) / y_sig, dtype=torch.float32).unsqueeze(1)
    y_val_np = y[val_idx]
    y_te_np  = y[test_idx]

    # --- Build ENGINE adapter ---
    adapter = EngineAdapterHead(hidden_dim=h_dim, num_layers=n_steps)
    opt  = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    crit = nn.MSELoss()

    best_val_mae = float("inf")
    best_state   = None

    print(f"\nTraining ENGINE adapter  ({args.epochs} epochs) …")
    print(f"  {'Epoch':>6}  {'train_loss':>11}  {'val_MAE':>8}")
    print("  " + "-" * 32)

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        opt.zero_grad()
        exit_preds = adapter(tr_tensors)
        loss = sum(crit(p, y_tr) for p in exit_preds)
        loss.backward()
        opt.step()
        sched.step()

        if epoch % 50 == 0 or epoch == args.epochs:
            adapter.eval()
            pred_val = adapter.predict_ensemble(val_tensors) * y_sig + y_mu
            val_mae  = float(np.mean(np.abs(pred_val - y_val_np)))
            print(f"  {epoch:>6}  {loss.item():>11.4f}  {val_mae:>8.4f}")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state   = {k: v.clone() for k, v in adapter.state_dict().items()}

    # Restore best weights
    adapter.load_state_dict(best_state)
    adapter.eval()

    # --- Test evaluation ---
    pred_ens = adapter.predict_ensemble(te_tensors) * y_sig + y_mu
    tol_norm = 0.1 / y_sig
    pred_ee, exit_layers = adapter.predict_early_exit(te_tensors, tolerance=tol_norm)
    pred_ee = pred_ee * y_sig + y_mu

    def mae(a, b):  return float(np.mean(np.abs(a - b)))
    def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
    def r2(a, b):
        ss_res = np.sum((a - b) ** 2)
        ss_tot = np.sum((a - a.mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot else 0.0

    W = 35
    print(f"\n{'─' * (W + 26)}")
    print(f"  {'Test results':<{W}}  {'MAE':>6}  {'RMSE':>6}  {'R²':>6}")
    print(f"  {'─' * (W + 24)}")
    print(f"  {'ENGINE ensemble':<{W}}  {mae(y_te_np,pred_ens):6.3f}"
          f"  {rmse(y_te_np,pred_ens):6.3f}  {r2(y_te_np,pred_ens):6.3f}")
    avg_exit = float(exit_layers.mean()) + 1
    pct_ee   = float((exit_layers < n_steps - 1).mean() * 100)
    print(f"  {'ENGINE early exit (tol=0.1)':<{W}}  {mae(y_te_np,pred_ee):6.3f}"
          f"  {rmse(y_te_np,pred_ee):6.3f}  {r2(y_te_np,pred_ee):6.3f}"
          f"  [{avg_exit:.1f}/{n_steps} avg, {pct_ee:.0f}% early]")
    print(f"{'─' * (W + 26)}")
    print(f"\n  α gates: " + ", ".join(
        f"σ(α_{i})={torch.sigmoid(a).item():.2f}"
        for i, a in enumerate(adapter.alphas)
    ))

    # --- Save ---
    save_adapter(
        path=Path(args.output),
        adapter=adapter,
        layer_scalers=layer_scalers,
        label_mu=y_mu,
        label_sig=y_sig,
        backbone_checkpoint=str(ckpt_path.resolve()),
        training_info={
            "dataset": str(Path(args.data).name),
            "target_col": target_col,
            "n_train": len(train_idx),
            "n_val":   len(val_idx),
            "n_test":  len(test_idx),
            "epochs":  args.epochs,
            "best_val_mae": round(best_val_mae, 4),
            "test_mae_ensemble":  round(mae(y_te_np, pred_ens), 4),
            "test_mae_early_exit": round(mae(y_te_np, pred_ee),  4),
        },
    )
    print(f"\n  Adapter saved → {args.output}")
    print(f"  Run inference with:  python scripts/predict_solubility.py "
          f"--adapter {args.output} 'CCO' 'c1ccccc1' ...")


if __name__ == "__main__":
    main()
