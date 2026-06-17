#!/usr/bin/env python3
"""
Predict a molecular property using any saved adapter.

Supports all three adaptation methods (auto-detected from the adapter file):
  mlp_head   — frozen backbone + MLP head
  finetune   — fine-tuned backbone + MLP head
  engine     — ENGINE side-structure adapter

Usage:
  # Predict directly from SMILES strings:
  python scripts/predict_property.py \\
      --adapter runs/mlp_head_solubility.pt \\
      "CCO" "c1ccccc1" "CC(=O)O"

  # Predict from a CSV and save results:
  python scripts/predict_property.py \\
      --adapter runs/finetune_solubility.pt \\
      --csv molecules.csv \\
      --smiles-col smiles \\
      --output predictions.csv

  # ENGINE early-exit mode:
  python scripts/predict_property.py \\
      --adapter runs/solubility_adapter.pt \\
      --mode early_exit \\
      "CCO" "c1ccccc1"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from qchem_gnn.adapt import predict_smiles


def _adapter_summary(adapter_path: Path) -> tuple[str, dict]:
    """Read adapter_type and training_info without loading model weights."""
    state = torch.load(adapter_path, map_location="cpu", weights_only=False)
    return state.get("adapter_type", "engine"), state.get("training_info", {})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("smiles",       nargs="*", help="SMILES strings to score")
    parser.add_argument("--adapter",    required=True, help="Path to saved adapter .pt")
    parser.add_argument("--csv",        default=None,  help="CSV with a SMILES column")
    parser.add_argument("--smiles-col", default=None,  help="SMILES column name in CSV")
    parser.add_argument("--output",     default=None,  help="Write predictions to CSV")
    parser.add_argument("--mode",       default="ensemble",
                        choices=["ensemble", "early_exit"],
                        help="Inference mode (ENGINE adapter only)")
    parser.add_argument("--exit-tol",  type=float, default=0.05,
                        help="ENGINE early-exit std threshold in normalised units")
    parser.add_argument("--batch",     type=int,   default=256)
    args = parser.parse_args()

    # ── Collect SMILES ─────────────────────────────────────────────────────────
    smiles_list: list[str] = list(args.smiles)

    if args.csv:
        df = pd.read_csv(args.csv)
        col = args.smiles_col or next(
            (c for c in df.columns if c.lower() == "smiles"), df.columns[0]
        )
        smiles_list.extend(df[col].dropna().tolist())

    if not smiles_list:
        print("ERROR: provide SMILES on the command line or via --csv", file=sys.stderr)
        sys.exit(1)

    # ── Adapter summary ────────────────────────────────────────────────────────
    adapter_path = Path(args.adapter)
    adapter_type, info = _adapter_summary(adapter_path)

    method_label = {
        "mlp_head": "MLP head (frozen backbone)",
        "finetune": "Fine-tuned backbone + MLP head",
        "engine":   f"ENGINE side adapter ({args.mode})",
    }.get(adapter_type, adapter_type)

    print(f"\nAdapter type : {method_label}")
    if info:
        print(f"Trained on   : {info.get('dataset', '?')}  "
              f"({info.get('n_train', '?')} train / {info.get('n_test', '?')} test)")
        target = info.get("target_col", "property")
        print(f"Target       : {target}")
        print(f"Best val MAE : {info.get('best_val_mae', '?')}  "
              f"Test MAE: {info.get('test_mae', '?')}  "
              f"R²: {info.get('test_r2', '?')}")

    # ── Predict ────────────────────────────────────────────────────────────────
    print(f"\nScoring {len(smiles_list)} molecule(s) …")

    preds, valid_idx = predict_smiles(
        smiles_list, adapter_path,
        mode=args.mode, exit_tolerance=args.exit_tol, batch_size=args.batch,
    )
    # preds is [N_valid, T]; determine target column names from adapter metadata
    state_meta = torch.load(adapter_path, map_location="cpu", weights_only=False)
    saved_target_names = state_meta.get("target_names")

    multi_target = preds.ndim == 2 and preds.shape[1] > 1
    if not multi_target:
        # single-target: flatten to 1D for simple display
        preds_1d = preds[:, 0] if preds.ndim == 2 else preds
        target_label = info.get("target_col", "prediction") if info else "prediction"
    else:
        # multi-target: use saved target names or generic col names
        n_t = preds.shape[1]
        if saved_target_names and len(saved_target_names) == n_t:
            target_cols = saved_target_names
        else:
            target_cols = [f"pred_{i}" for i in range(n_t)]

    # ── Output ─────────────────────────────────────────────────────────────────
    valid_set = set(valid_idx)
    rows = []
    vi = 0
    if not multi_target:
        for i, smi in enumerate(smiles_list):
            if i in valid_set:
                rows.append({"smiles": smi, target_label: round(float(preds_1d[vi]), 4)})
                vi += 1
            else:
                rows.append({"smiles": smi, target_label: None})
    else:
        for i, smi in enumerate(smiles_list):
            if i in valid_set:
                row = {"smiles": smi}
                row.update({col: round(float(preds[vi, t]), 4) for t, col in enumerate(target_cols)})
                rows.append(row)
                vi += 1
            else:
                rows.append({"smiles": smi, **{col: None for col in target_cols}})
    result_df = pd.DataFrame(rows)

    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"  Saved {len(result_df)} rows → {args.output}")
    else:
        col_w = min(50, max(len(s) for s in smiles_list) + 2)
        if not multi_target:
            print(f"\n  {'SMILES':<{col_w}}  {target_label}")
            print("  " + "-" * (col_w + len(target_label) + 4))
            for _, row in result_df.iterrows():
                val = f"{row[target_label]:+.3f}" if row[target_label] is not None else "FAILED"
                print(f"  {str(row['smiles']):<{col_w}}  {val}")
        else:
            header = "  ".join(f"{c:>10}" for c in target_cols)
            print(f"\n  {'SMILES':<{col_w}}  {header}")
            print("  " + "-" * (col_w + 12 * len(target_cols) + 4))
            for _, row in result_df.iterrows():
                vals = "  ".join(
                    f"{row[c]:>+10.3f}" if row[c] is not None else f"{'FAILED':>10}"
                    for c in target_cols
                )
                print(f"  {str(row['smiles']):<{col_w}}  {vals}")

    fail_col = target_label if not multi_target else target_cols[0]
    n_fail = result_df[fail_col].isna().sum()
    if n_fail:
        print(f"\n  {n_fail} SMILES could not be parsed.")


if __name__ == "__main__":
    main()
