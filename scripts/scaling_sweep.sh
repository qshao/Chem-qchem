#!/usr/bin/env bash
# Measure downstream MAE vs. pretraining data scale.
#
# Usage:
#   bash scripts/scaling_sweep.sh <dataset_root> <cache_dir> [scales]
#
# Arguments:
#   dataset_root  ZINC root with subsets/, geometries/, results/
#   cache_dir     Where compact shard caches live / will be written
#   scales        Space-separated shard counts (default: "1 10 50")
#
# Example:
#   bash scripts/scaling_sweep.sh zinc-250k zinc-250k/compact_cache "1 10 50"
set -euo pipefail

DATASET_ROOT="${1:?Usage: scaling_sweep.sh <dataset_root> <cache_dir> [scales]}"
CACHE_DIR="${2:?Usage: scaling_sweep.sh <dataset_root> <cache_dir> [scales]}"
SCALES="${3:-1 10 50}"
BASE_CONFIG="configs/validate_scaled.yaml"

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# Preprocess enough shards for the largest scale (skip-if-exists makes this cheap).
MAX_SCALE=$(echo "$SCALES" | tr ' ' '\n' | sort -n | tail -1)
MAX_IDX=$((MAX_SCALE - 1))
IDS=$(seq -s, 0 "$MAX_IDX")
python -m qchem_gnn.cli preprocess --dataset-root "$DATASET_ROOT" --subset-ids "$IDS" --cache-dir "$CACHE_DIR"

REPORT_ARGS=()
for SCALE in $SCALES; do
    LAST_IDX=$((SCALE - 1))
    IDS=$(seq -s, 0 "$LAST_IDX")
    OUT_DIR="runs/scaling_s${SCALE}"
    # Override subset_ids and output dir for this scale via a temp config.
    python - "$BASE_CONFIG" "$CACHE_DIR" "$IDS" "$OUT_DIR" "$TMPDIR/scaling_cfg_${SCALE}.yaml" <<'PY'
import sys, yaml
base, cache_dir, ids, out_dir, cfg_out = sys.argv[1:6]
cfg = yaml.safe_load(open(base))
cfg["pretrain"]["cache_dir"] = cache_dir
cfg["pretrain"]["subset_ids"] = [int(x) for x in ids.split(",")]
cfg["outputs"]["dir"] = out_dir
cfg["outputs"]["report"] = f"{out_dir}/report"
yaml.safe_dump(cfg, open(cfg_out, "w"))
PY
    python -m qchem_gnn.validation --config "$TMPDIR/scaling_cfg_${SCALE}.yaml"
    REPORT_ARGS+=("${SCALE}=${OUT_DIR}/report.json")
done

echo "=== MAE vs scale ==="
python scripts/aggregate_scaling.py "${REPORT_ARGS[@]}"
