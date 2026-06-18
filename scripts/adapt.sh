#!/usr/bin/env bash
# Adapt a frozen backbone to a new property prediction task via a CSV dataset.
#
# Usage:
#   bash scripts/adapt.sh <adapt_config.yaml> [KEY=VALUE ...]
#
# Arguments:
#   adapt_config  YAML config file (see configs/adapt_example.yaml for template)
#   KEY=VALUE     Optional dotted-key overrides, e.g. training.epochs=200
#
# Example:
#   bash scripts/adapt.sh configs/adapt_example.yaml training.epochs=200
set -euo pipefail

CONFIG="${1:?Usage: adapt.sh <adapt_config.yaml> [KEY=VALUE ...]}"
shift

if [ "$#" -gt 0 ]; then
    python -m qchem_gnn.cli adapt "$CONFIG" --override "$@"
else
    python -m qchem_gnn.cli adapt "$CONFIG"
fi
