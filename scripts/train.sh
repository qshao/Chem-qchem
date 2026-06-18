#!/usr/bin/env bash
# Train from compact shard caches (run scripts/preprocess.sh first).
#
# Usage:
#   bash scripts/train.sh <cache_dir> [shard_range] [output_dir] [--overwrite]
#
# Arguments:
#   cache_dir     Directory containing compact shard .pt files (from preprocess.sh)
#   shard_range   Which shards to train on; three formats accepted:
#                   "0-9"      shards 0 through 9 inclusive (default: "0-9")
#                   "0,1,5,7"  explicit comma-separated list
#                   "0"        single shard
#   output_dir    Where to write backbones and the validation report
#                 (default: runs/validate_scaled)
#   --overwrite   Re-train even if backbones already exist
#
# Examples:
#   # Quick 1-shard smoke-test:
#   bash scripts/train.sh zinc-250k/compact_cache "0" runs/train_smoke
#
#   # Train on first 10 shards with 3 seeds (default config):
#   bash scripts/train.sh zinc-250k/compact_cache "0-9"
#
#   # Train on all 250 shards:
#   bash scripts/train.sh zinc-250k/compact_cache "0-249" runs/train_250k
#
#   # Resume an interrupted run:
#   RESUME=true bash scripts/train.sh zinc-250k/compact_cache "0-9"
#
# The script builds a temporary config from configs/validate_scaled.yaml,
# overriding cache_dir, subset_ids, and output paths, then calls:
#   python -m qchem_gnn.validation --config <tmp_config>
set -euo pipefail

CACHE_DIR="${1:?Usage: train.sh <cache_dir> [shard_range] [output_dir] [--overwrite]}"
SHARD_ARG="${2:-0-9}"
OUTPUT_DIR="${3:-runs/validate_scaled}"
OVERWRITE_FLAG="${4:-}"
BASE_CONFIG="configs/validate_scaled.yaml"

# Build comma-separated id list from the shard_range argument.
if [[ "$SHARD_ARG" == *"-"* && "$SHARD_ARG" != *","* ]]; then
    START="${SHARD_ARG%-*}"
    END="${SHARD_ARG#*-}"
    IDS=$(seq -s, "$START" "$END")
    N_SHARDS=$(( END - START + 1 ))
else
    IDS="$SHARD_ARG"
    N_SHARDS=$(echo "$IDS" | tr ',' '\n' | wc -l | tr -d ' ')
fi

echo "=== Training from compact cache ==="
echo "    cache_dir  : $CACHE_DIR"
echo "    shards     : $IDS ($N_SHARDS shard(s))"
echo "    output_dir : $OUTPUT_DIR"
echo ""

# Verify cache files exist before starting.
MISSING=0
for ID in $(echo "$IDS" | tr ',' ' '); do
    F="$CACHE_DIR/$(printf 'shard_%03d.pt' "$ID")"
    if [[ ! -f "$F" ]]; then
        echo "  ERROR: missing cache file $F"
        echo "  Run: bash scripts/preprocess.sh <dataset_root> $CACHE_DIR $SHARD_ARG"
        MISSING=1
    fi
done
if [[ "$MISSING" -eq 1 ]]; then
    exit 1
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT
TMP_CONFIG="$TMPDIR/train_cfg.yaml"

# Build a temp config overriding cache_dir, subset_ids, and output paths.
RESUME="${RESUME:-false}"
python - "$BASE_CONFIG" "$CACHE_DIR" "$IDS" "$OUTPUT_DIR" "$RESUME" "$TMP_CONFIG" <<'PY'
import sys, yaml
base, cache_dir, ids, out_dir, resume, cfg_out = sys.argv[1:7]
cfg = yaml.safe_load(open(base))
cfg["pretrain"]["cache_dir"] = cache_dir
cfg["pretrain"]["subset_ids"] = [int(x) for x in ids.split(",")]
cfg["pretrain"]["resume"] = resume.lower() == "true"
cfg["outputs"]["dir"] = out_dir
cfg["outputs"]["report"] = f"{out_dir}/report"
yaml.safe_dump(cfg, open(cfg_out, "w"))
PY

OVERWRITE_ARG=""
if [[ "$OVERWRITE_FLAG" == "--overwrite" ]]; then
    OVERWRITE_ARG="--overwrite"
fi

python -m qchem_gnn.validation --config "$TMP_CONFIG" $OVERWRITE_ARG

echo ""
echo "=== Done. Backbones and report in $OUTPUT_DIR/ ==="
ls "$OUTPUT_DIR"/*.pt 2>/dev/null | head -10 || true
