#!/usr/bin/env bash
# One-time preprocessing: extract ZINC shards into compact scaffold-keyed caches.
#
# Drops the density matrix (~6.8 GB/shard → ~90 MB/shard) and attaches a globally
# stable Murcko scaffold key to every molecule. The result is a compact_cache/
# directory that the training harness and scaling sweep use directly.
#
# Usage:
#   bash scripts/preprocess.sh <dataset_root> <cache_dir> [shard_range]
#
# Arguments:
#   dataset_root   ZINC root containing subsets/, geometries/, results/
#   cache_dir      Output directory for compact shard caches
#   shard_range    Which shards to process; three formats accepted:
#                    "0-49"        process shards 0 through 49 inclusive (default: "0-9")
#                    "0,1,5,7"     explicit comma-separated list
#                    "0"           single shard
#
# The command is resumable: shards with a valid existing cache are skipped.
# Re-run after interruption without --overwrite to continue where you left off.
#
# Examples:
#   # Preprocess the first 10 shards (quick smoke-test):
#   bash scripts/preprocess.sh zinc-250k zinc-250k/compact_cache "0-9"
#
#   # Preprocess all 250 shards (hours; runs once, then instant on re-run):
#   bash scripts/preprocess.sh zinc-250k zinc-250k/compact_cache "0-249"
#
#   # Re-extract specific shards (e.g., after a corrupt write):
#   bash scripts/preprocess.sh zinc-250k zinc-250k/compact_cache "12,37,88" --overwrite
set -euo pipefail

DATASET_ROOT="${1:?Usage: preprocess.sh <dataset_root> <cache_dir> [shard_range]}"
CACHE_DIR="${2:?Usage: preprocess.sh <dataset_root> <cache_dir> [shard_range]}"
SHARD_ARG="${3:-0-9}"
OVERWRITE="${4:-}"

# --- Build the comma-separated id list from the shard_range argument ---
if [[ "$SHARD_ARG" == *"-"* && "$SHARD_ARG" != *","* ]]; then
    # Range format: "0-49"
    START="${SHARD_ARG%-*}"
    END="${SHARD_ARG#*-}"
    IDS=$(seq -s, "$START" "$END")
    N_SHARDS=$(( END - START + 1 ))
else
    # Explicit list: "0,1,5,7" or "0"
    IDS="$SHARD_ARG"
    N_SHARDS=$(echo "$IDS" | tr ',' '\n' | wc -l | tr -d ' ')
fi

echo "=== Preprocessing $N_SHARDS shard(s) ==="
echo "    dataset_root : $DATASET_ROOT"
echo "    cache_dir    : $CACHE_DIR"
echo "    shards       : $IDS"
echo "    estimated output: ~$((N_SHARDS * 90)) MB"
echo ""

OVERWRITE_FLAG=""
if [[ "$OVERWRITE" == "--overwrite" ]]; then
    OVERWRITE_FLAG="--overwrite"
    echo "    (--overwrite: re-extracting even if cache exists)"
fi

python -m qchem_gnn preprocess \
    --dataset-root "$DATASET_ROOT" \
    --subset-ids "$IDS" \
    --cache-dir "$CACHE_DIR" \
    ${OVERWRITE_FLAG:+$OVERWRITE_FLAG}

echo ""
echo "=== Done. Cache directory: $CACHE_DIR ==="
du -sh "$CACHE_DIR" 2>/dev/null && echo ""
echo "Re-run with the same arguments to skip existing shards (idempotent)."
