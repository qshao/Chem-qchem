#!/usr/bin/env bash
# Download the BDE-db bond dissociation enthalpy dataset.
#
# Source: https://github.com/nsf-c-cas/BDE-db
# Paper:  Schwaller et al., 290,664 Homolytic Bond Dissociation Enthalpies
#         for Small Organic Molecules (10 or fewer heavy atoms, C/H/O/N only)
#         Computed at M06-2X/def2-TZVP via Gaussian 16.
#
# Output: data/bde.csv (~31 MB)
#
# Schema:
#   rid         integer row id
#   molecule    parent molecule SMILES  (use as smiles_col)
#   bond_index  index of the broken bond within the molecule
#   fragment1   SMILES of the first radical fragment
#   fragment2   SMILES of the second radical fragment
#   bde         homolytic BDE in kcal/mol  (use as prediction target)
#   bond_type   bond type string, e.g. C-H, C-C, C-O, H-O, C-N
#
# Usage:
#   bash data/download_bde.sh
set -euo pipefail

OUT="$(dirname "$0")/bde.csv"

if [ -f "$OUT" ]; then
    echo "Already exists: $OUT  ($(wc -l < "$OUT") lines)"
    echo "Delete it to re-download."
    exit 0
fi

echo "Downloading BDE-db (~31 MB) ..."
curl -L --progress-bar \
    -o "$OUT" \
    "https://raw.githubusercontent.com/nsf-c-cas/BDE-db/master/rdf_data_190531.csv"

echo ""
echo "Saved to $OUT  ($(wc -l < "$OUT") rows)"
echo ""
echo "Columns: $(head -1 "$OUT")"
