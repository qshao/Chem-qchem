#!/usr/bin/env bash

#SBATCH --job-name=qchem-scaled
#SBATCH --account=gol_qsh226_uksr
#SBATCH --partition=SKY32M192_D
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Edit these variables if the checkout, config, logs, runs, or virtualenv path moves.
# PROJECT_ROOT defaults to the directory you submit from; change the fallback if the repo moves.
# CONFIG_FILE should point at the YAML you want to run.
# LOG_DIR and RUN_DIR default to subdirectories under PROJECT_ROOT.
# VENV_DIR points at the repository virtual environment; set it to a different path if needed.

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/scratch/qsh226/qchem}"
CONFIG_FILE="$PROJECT_ROOT/configs/validate_scaled_slurm.yaml"
LOG_DIR="$PROJECT_ROOT/logs"
RUN_DIR="$PROJECT_ROOT/runs"
VENV_DIR="$PROJECT_ROOT/.venv"
cd "$PROJECT_ROOT"

mkdir -p "$LOG_DIR" "$RUN_DIR"

if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

python -m qchem_gnn.validation --config "$CONFIG_FILE"
