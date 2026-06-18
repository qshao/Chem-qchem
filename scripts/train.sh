#!/usr/bin/env bash
# Contrastive pretraining with scaffold-aware negative masking (best config from experiments).
#
# Usage:
#   bash scripts/train.sh <dataset_root> [output_dir] [seeds]
#
# Arguments:
#   dataset_root  Directory containing subsets/, geometries/, and results/
#   output_dir    Where to write backbone_s{N}.pt files (default: checkpoints/)
#   seeds         Space-separated seed list (default: "0 1 2")
#
# Example:
#   bash scripts/train.sh /data/zinc_shard checkpoints "0 1 2 3 4 5 6"
set -euo pipefail

DATASET_ROOT="${1:?Usage: train.sh <dataset_root> [output_dir] [seeds]}"
OUTPUT_DIR="${2:-checkpoints}"
SEEDS="${3:-0 1 2}"

mkdir -p "$OUTPUT_DIR"

for SEED in $SEEDS; do
    echo "=== Training seed $SEED ==="
    python -m qchem_gnn.cli contrastive-pretrain \
        --dataset-root "$DATASET_ROOT" \
        --epochs 200 \
        --hidden-dim 64 \
        --hidden-dim-3d 64 \
        --message-passing-steps 3 \
        --message-passing-steps-3d 3 \
        --batch-size 16 \
        --learning-rate 1e-3 \
        --teacher-weight 1.0 \
        --contrastive-weight 1.0 \
        --temperature 0.1 \
        --conformer-pool-mode energy \
        --energy-temperature 298.15 \
        --contrastive-loss infonce \
        --use-scaffold-negmask \
        --seed "$SEED" \
        --output "$OUTPUT_DIR/backbone_s${SEED}.pt"
    echo "Saved $OUTPUT_DIR/backbone_s${SEED}.pt"
done

echo "=== All seeds done. Backbones in $OUTPUT_DIR/ ==="
