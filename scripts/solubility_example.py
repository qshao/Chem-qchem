#!/usr/bin/env python3
"""
Solubility prediction from a frozen 2D/3D contrastive GNN checkpoint.

Four frozen-backbone adaptation methods (backbone weights never updated):

  1. Linear probe   — Ridge regression on frozen final embeddings (closed-form)
  2. k-NN           — k=7, cosine distance; zero parameters, pure retrieval
  3. MLP head       — 2-layer MLP trained on top of frozen final embeddings
  4. ENGINE adapter — Tunable side structure across all GNN layers (see below)

ENGINE side structure (Zhu et al., https://github.com/zhuyun97/engine):
  At each frozen message-passing step i, a learnable projection + alpha gate
  blends the layer's pooled state into an accumulating side embedding h_i:

      h_i = proj_i(gnn_state_i) * sigmoid(α_i) + h_{i-1} * (1 - sigmoid(α_i))
      exit_pred_i = head_i(h_i)
      L_train = Σ_i MSE(exit_pred_i, y)          # sum loss over all exits

  Inference modes:
    · ensemble — average predictions from all exit heads
    · early exit — stop at the first layer where inter-head prediction
                   std < tolerance (avoids unnecessary computation)

Compared against a Morgan ECFP4 + Ridge baseline (no GNN).

Download AqSolDB from Kaggle first:
  https://www.kaggle.com/datasets/sorkun/aqsoldb-a-curated-aqueous-solubility-dataset
  Unzip → curated-solubility-dataset.csv

Usage:
  python scripts/solubility_example.py \\
      --checkpoint runs/example_contrastive.pt \\
      --data curated-solubility-dataset.csv

Optional flags:
  --test-frac   0.2    fraction held out (default 0.2)
  --seed        42     random seed (default 42)
  --batch       256    embedding batch size (default 256)
  --exit-tol    0.1    ENGINE early-exit tolerance in log(mol/L) (default 0.1)
  --output      runs/solubility_results.json  (default: print only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from qchem_gnn.checkpoint import load_checkpoint
from qchem_gnn.engine_adapter import EngineAdapterHead, extract_intermediate_embeddings
from qchem_gnn.model import MolecularQuantumGNN


def morgan_features(smiles_list: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    out = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            DataStructs.ConvertToNumpyArray(fp, out[i])
    return out


# ---------------------------------------------------------------------------
# Adaptation methods 1-3 (operate on final frozen embedding only)
# ---------------------------------------------------------------------------

def _zscore(X_tr: np.ndarray, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sigma = X_tr.mean(0), X_tr.std(0).clip(1e-8)
    return (X_tr - mu) / sigma, (X_te - mu) / sigma


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def adapt_linear_probe(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    alpha: float = 1.0,
) -> dict:
    """Ridge regression (closed-form). No gradient descent needed."""
    from sklearn.linear_model import Ridge
    X_tr_s, X_te_s = _zscore(X_tr, X_te)
    pred = Ridge(alpha=alpha).fit(X_tr_s, y_tr).predict(X_te_s)
    return _metrics(y_te, pred)


def adapt_knn(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    k: int = 7,
) -> dict:
    """k-NN on cosine similarity — zero learnable parameters."""
    from sklearn.neighbors import KNeighborsRegressor
    X_tr_s, X_te_s = _zscore(X_tr, X_te)
    pred = KNeighborsRegressor(n_neighbors=k, metric="cosine").fit(X_tr_s, y_tr).predict(X_te_s)
    return _metrics(y_te, pred)


def adapt_mlp_head(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    hidden: int = 128,
    epochs: int = 300,
    lr: float = 3e-3,
    seed: int = 42,
) -> dict:
    """2-layer MLP head; backbone stays frozen throughout."""
    torch.manual_seed(seed)
    X_tr_s, X_te_s = _zscore(X_tr, X_te)
    y_mu, y_sig = float(y_tr.mean()), float(y_tr.std()) or 1.0
    X_t = torch.as_tensor(X_tr_s, dtype=torch.float32)
    y_t = torch.as_tensor((y_tr - y_mu) / y_sig, dtype=torch.float32).unsqueeze(1)
    head = nn.Sequential(nn.Linear(X_tr_s.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    crit = nn.MSELoss()
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        crit(head(X_t), y_t).backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        pred = head(torch.as_tensor(X_te_s, dtype=torch.float32)).squeeze(1).numpy() * y_sig + y_mu
    return _metrics(y_te, pred)


# ---------------------------------------------------------------------------
# Adaptation method 4: ENGINE side structure
# ---------------------------------------------------------------------------

def adapt_engine(
    layer_embs_tr: list[np.ndarray],
    y_tr: np.ndarray,
    layer_embs_te: list[np.ndarray],
    y_te: np.ndarray,
    hidden_dim: int,
    epochs: int = 300,
    lr: float = 3e-3,
    seed: int = 42,
    exit_tolerance: float = 0.1,
) -> dict:
    """
    Train ENGINE side structure on frozen per-layer GNN embeddings.

    Each layer gets its own projection + alpha gate + exit head.
    Loss during training is the sum of MSE across every exit.
    """
    torch.manual_seed(seed)
    num_layers = len(layer_embs_tr)

    # Z-score normalise each layer independently using training statistics
    norm_tr, norm_te = [], []
    for i in range(num_layers):
        mu = layer_embs_tr[i].mean(0)
        sig = layer_embs_tr[i].std(0).clip(1e-8)
        norm_tr.append((layer_embs_tr[i] - mu) / sig)
        norm_te.append((layer_embs_te[i] - mu) / sig)

    tr_tensors = [torch.as_tensor(x, dtype=torch.float32) for x in norm_tr]
    te_tensors = [torch.as_tensor(x, dtype=torch.float32) for x in norm_te]

    # Normalise labels on training split
    y_mu, y_sig = float(y_tr.mean()), float(y_tr.std()) or 1.0
    y_t = torch.as_tensor((y_tr - y_mu) / y_sig, dtype=torch.float32).unsqueeze(1)

    adapter = EngineAdapterHead(hidden_dim=hidden_dim, num_layers=num_layers)
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
    crit = nn.MSELoss()

    adapter.train()
    for _ in range(epochs):
        opt.zero_grad()
        exit_preds = adapter(tr_tensors)
        loss = sum(crit(p, y_t) for p in exit_preds)
        loss.backward()
        opt.step()

    adapter.eval()

    # Ensemble: average all exit-head predictions
    pred_ens = adapter.predict_ensemble(te_tensors) * y_sig + y_mu

    # Early exit: stop when inter-head std < tolerance (in normalised units)
    tol_norm = exit_tolerance / y_sig          # convert to normalised space
    pred_ee, exit_layers = adapter.predict_early_exit(te_tensors, tolerance=tol_norm)
    pred_ee = pred_ee * y_sig + y_mu

    # Summarise exit behaviour
    avg_exit = float(exit_layers.mean()) + 1   # 1-indexed for display
    pct_early = float((exit_layers < num_layers - 1).mean() * 100)

    return {
        "ensemble": _metrics(y_te, pred_ens),
        "early_exit": {
            **_metrics(y_te, pred_ee),
            "avg_exit_layer": round(avg_exit, 2),
            "pct_early_exit": round(pct_early, 1),
        },
        "alpha_gates": [float(torch.sigmoid(a).item()) for a in adapter.alphas],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--exit-tol", type=float, default=0.1,
                        help="ENGINE early-exit tolerance in log(mol/L)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # --- AqSolDB ---
    data_path = Path(args.data)
    if not data_path.exists():
        print(
            f"ERROR: {data_path} not found.\n\n"
            "Download AqSolDB:\n"
            "  https://www.kaggle.com/datasets/sorkun/aqsoldb-a-curated-aqueous-solubility-dataset\n"
            "Unzip → curated-solubility-dataset.csv  then pass --data <path>",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(data_path)
    smiles_col = next(c for c in df.columns if c.upper() == "SMILES")
    sol_col = next(
        c for c in df.columns
        if "solubility" in c.lower() and "sd" not in c.lower() and "log" not in c.lower()
    )
    df = df[[smiles_col, sol_col]].dropna()
    all_smiles = df[smiles_col].tolist()
    all_y = df[sol_col].to_numpy(dtype=np.float32)

    print(f"\nAqSolDB: {len(df)} molecules")
    print(f"  Solubility (log mol/L): min={all_y.min():.2f}  max={all_y.max():.2f}  "
          f"mean={all_y.mean():.2f}  std={all_y.std():.2f}")

    # --- Checkpoint ---
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    ckpt = load_checkpoint(ckpt_path)
    model = MolecularQuantumGNN(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    h_dim = ckpt["model_config"]["hidden_dim"]
    n_steps = ckpt["model_config"]["num_message_passing_steps"]
    print(f"\nCheckpoint: epoch {ckpt.get('epoch')}, hidden_dim={h_dim}, steps={n_steps}")

    # --- Embed (collect final embedding + all intermediate layer embeddings) ---
    print("\nEmbedding molecules with frozen 2D GNN (collecting all layers) …")
    layer_embs, final_embs, valid_idx = extract_intermediate_embeddings(
        all_smiles, model, batch_size=args.batch
    )
    valid_smiles = [all_smiles[i] for i in valid_idx]
    y = all_y[valid_idx]
    n_failed = len(all_smiles) - len(valid_idx)
    print(f"  Parsed {len(valid_idx)}/{len(all_smiles)} SMILES  ({n_failed} skipped)")
    print(f"  Final embeddings: {final_embs.shape}   "
          f"Layer embeddings: {n_steps} × {layer_embs[0].shape}")

    # --- Stratified train/test split ---
    rng = np.random.default_rng(args.seed)
    quintiles = np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
    train_idx: list[int] = []
    test_idx: list[int] = []
    for q in range(5):
        mask = np.where(quintiles == q)[0]
        rng.shuffle(mask)
        cut = max(1, int(len(mask) * args.test_frac))
        test_idx.extend(mask[:cut].tolist())
        train_idx.extend(mask[cut:].tolist())

    X_tr, X_te = final_embs[train_idx], final_embs[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    L_tr = [e[train_idx] for e in layer_embs]
    L_te = [e[test_idx] for e in layer_embs]
    print(f"  Split: {len(train_idx)} train  /  {len(test_idx)} test")

    # --- Morgan baseline ---
    print("\nComputing Morgan fingerprints (ECFP4, 2048 bits) …")
    mfp = morgan_features(valid_smiles)
    mfp_tr, mfp_te = mfp[train_idx], mfp[test_idx]

    # --- Results table ---
    W = 42
    print("\n" + "=" * (W + 25))
    print(f"  {'Method':<{W}}  {'MAE':>6}  {'RMSE':>6}  {'R²':>6}")
    print("  " + "-" * (W + 23))

    results: dict[str, dict] = {}

    def _row(name: str, m: dict, extra: str = "") -> None:
        line = f"  {name:<{W}}  {m['mae']:6.3f}  {m['rmse']:6.3f}  {m['r2']:6.3f}"
        print(line + (f"  {extra}" if extra else ""))
        results[name] = m

    _row("Morgan ECFP4 + Ridge (baseline)", adapt_linear_probe(mfp_tr, y_tr, mfp_te, y_te))
    print("  " + "-" * (W + 23))

    _row("GNN frozen │ 1. Linear probe (Ridge)", adapt_linear_probe(X_tr, y_tr, X_te, y_te))
    _row("GNN frozen │ 2. k-NN  (k=7, cosine)", adapt_knn(X_tr, y_tr, X_te, y_te))

    print(f"  {'GNN frozen │ 3. MLP head  (training…)':<{W}}", end="", flush=True)
    mlp_m = adapt_mlp_head(X_tr, y_tr, X_te, y_te, seed=args.seed)
    print(f"\r", end="")
    _row("GNN frozen │ 3. MLP head", mlp_m)

    # ENGINE side structure
    print("  " + "-" * (W + 23))
    print(f"  {'GNN frozen │ 4. ENGINE side (training…)':<{W}}", end="", flush=True)
    eng = adapt_engine(L_tr, y_tr, L_te, y_te, hidden_dim=h_dim,
                       seed=args.seed, exit_tolerance=args.exit_tol)
    print(f"\r", end="")

    ee = eng["early_exit"]
    _row("GNN frozen │ 4. ENGINE  (ensemble)", eng["ensemble"])
    _row(
        "GNN frozen │ 4. ENGINE  (early exit)",
        {k: v for k, v in ee.items() if k in {"mae", "rmse", "r2"}},
        extra=(
            f"avg exit layer {ee['avg_exit_layer']:.1f}/{n_steps}  "
            f"({ee['pct_early_exit']:.0f}% exit early)"
        ),
    )

    print("=" * (W + 25))
    print(
        f"\n  Units: MAE and RMSE in log(mol/L).\n"
        f"  Learned α gates (layer blend weight per step): "
        + ", ".join(f"σ(α_{i})={v:.2f}" for i, v in enumerate(eng["alpha_gates"]))
        + "\n"
        f"  GNN pretrained on {ckpt.get('run_metadata', {}).get('num_examples', '?')} molecules.\n"
        f"  Transfer quality scales with pretraining corpus size."
    )

    # --- Save ---
    if args.output:
        out = {
            "checkpoint": str(ckpt_path),
            "data": str(data_path),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "gnn_hidden_dim": h_dim,
            "gnn_num_steps": n_steps,
            "exit_tolerance": args.exit_tol,
            "results": results,
            "engine_alpha_gates": eng["alpha_gates"],
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\n  Results saved → {args.output}")


if __name__ == "__main__":
    main()
