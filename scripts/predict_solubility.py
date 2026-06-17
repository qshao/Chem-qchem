#!/usr/bin/env python3
"""
Predict aqueous solubility with a trained ENGINE adapter.

Loads a frozen pretrained GNN backbone and the trained side adapter
to predict log(S) in mol/L for any set of SMILES strings.

Usage:
  # Pass SMILES directly on the command line:
  python scripts/predict_solubility.py \\
      --adapter runs/solubility_adapter.pt \\
      "CCO" "c1ccccc1" "CC(=O)O"

  # Or pass a CSV file with a SMILES column:
  python scripts/predict_solubility.py \\
      --adapter runs/solubility_adapter.pt \\
      --csv my_molecules.csv \\
      --smiles-col smiles \\
      --output predictions.csv

  # Early exit instead of ensemble:
  python scripts/predict_solubility.py \\
      --adapter runs/solubility_adapter.pt \\
      --mode early_exit \\
      "CCO"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from qchem_gnn.engine_adapter import load_adapter, predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("smiles", nargs="*", help="SMILES strings to score")
    parser.add_argument("--adapter",    required=True, help="Path to trained adapter .pt")
    parser.add_argument("--backbone",   default=None,
                        help="Override backbone checkpoint path (default: from adapter)")
    parser.add_argument("--csv",        default=None, help="CSV file with a SMILES column")
    parser.add_argument("--smiles-col", default=None, help="Column name for SMILES in CSV")
    parser.add_argument("--output",     default=None, help="Write predictions to this CSV")
    parser.add_argument("--mode",       default="ensemble",
                        choices=["ensemble", "early_exit"])
    parser.add_argument("--exit-tol",   type=float, default=0.05,
                        help="Early-exit std tolerance in normalised units (default 0.05)")
    parser.add_argument("--batch",      type=int, default=256)
    args = parser.parse_args()

    # --- Collect SMILES ---
    smiles_list: list[str] = list(args.smiles)
    source_labels: list[str] = list(args.smiles)

    if args.csv:
        df = pd.read_csv(args.csv)
        col = args.smiles_col or next(c for c in df.columns if c.lower() == "smiles")
        csv_smiles = df[col].dropna().tolist()
        smiles_list.extend(csv_smiles)
        source_labels.extend(csv_smiles)

    if not smiles_list:
        print("ERROR: provide SMILES on the command line or via --csv", file=sys.stderr)
        sys.exit(1)

    # --- Load adapter metadata ---
    _, meta = load_adapter(args.adapter)
    backbone = args.backbone or meta["backbone_checkpoint"]
    info = meta.get("training_info", {})
    if info:
        print(f"\nAdapter trained on: {info.get('dataset', '?')}  "
              f"({info.get('n_train', '?')} train / {info.get('n_test', '?')} test)")
        print(f"  Best val MAE: {info.get('best_val_mae', '?')}  |  "
              f"Test MAE ensemble: {info.get('test_mae_ensemble', '?')}")

    # --- Predict ---
    print(f"\nPredicting log(S) for {len(smiles_list)} molecule(s) "
          f"[mode: {args.mode}] …")

    preds, valid_idx = predict(
        smiles_list,
        backbone_ckpt=backbone,
        adapter_path=args.adapter,
        mode=args.mode,
        exit_tolerance=args.exit_tol,
        batch_size=args.batch,
    )

    # --- Output ---
    rows = []
    valid_set = set(valid_idx)
    vi = 0
    for i, smi in enumerate(smiles_list):
        if i in valid_set:
            rows.append({"smiles": smi, "log_S_pred": round(float(preds[vi]), 4)})
            vi += 1
        else:
            rows.append({"smiles": smi, "log_S_pred": None})

    result_df = pd.DataFrame(rows)

    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"  Saved {len(result_df)} predictions → {args.output}")
    else:
        print(f"\n  {'SMILES':<45}  log(S) [mol/L]")
        print("  " + "-" * 60)
        for _, row in result_df.iterrows():
            pred_str = f"{row['log_S_pred']:+.3f}" if row["log_S_pred"] is not None else "FAILED"
            print(f"  {str(row['smiles']):<45}  {pred_str}")

    n_fail = result_df["log_S_pred"].isna().sum()
    if n_fail:
        print(f"\n  {n_fail} SMILES could not be parsed and were skipped.")


if __name__ == "__main__":
    main()
