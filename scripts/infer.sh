#!/usr/bin/env bash
# Export molecular embeddings from a trained backbone checkpoint.
#
# Usage:
#   bash scripts/infer.sh <checkpoint.pt> <output.npy>
#
# Arguments:
#   checkpoint  Path to a backbone .pt file produced by train.sh
#   output      Output path for the embeddings NumPy array (.npy)
#
# Example:
#   bash scripts/infer.sh checkpoints/backbone_s0.pt embeddings_s0.npy
set -euo pipefail

CHECKPOINT="${1:?Usage: infer.sh <checkpoint.pt> <output.npy>}"
OUTPUT="${2:?Usage: infer.sh <checkpoint.pt> <output.npy>}"

python -m qchem_gnn.cli export-embeddings \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT"

echo "Embeddings written to $OUTPUT"
