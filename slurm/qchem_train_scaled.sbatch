#!/usr/bin/env bash

#SBATCH --job-name=qchem-train
#SBATCH --account=gol_qsh226_uksr
#SBATCH --partition=H8V141_SAP112M2000_L
#SBATCH --gres=gpu:1
#SBATCH --chdir=.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-user=qshao@uky.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# Edit these variables if the checkout, config, logs, runs, or conda paths move.
# PROJECT_ROOT defaults to the directory you submit from; change the fallback if the repo moves.
# CONFIG_FILE should point at the YAML you want to run.
# CONDA_BASE and CONDA_ENV should match the miniconda install and environment name.
# LOG_DIR and RUN_DIR default to subdirectories under PROJECT_ROOT.

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/scratch/qsh226/qchem}"
CONFIG_FILE="$PROJECT_ROOT/configs/train_scaled_slurm.yaml"
LOG_DIR="$PROJECT_ROOT/logs"
RUN_DIR="$PROJECT_ROOT/runs"
cd "$PROJECT_ROOT"

mkdir -p "$LOG_DIR" "$RUN_DIR"

echo "Slurm job startup"
date
hostname
pwd
printenv SLURM_JOB_ID || true
printenv CUDA_VISIBLE_DEVICES || true

module load ccs/cuda/12.2.0_535.54.03
module list
nvidia-smi

CONDA_BASE="/project/qsh226_uksr/qsh226/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
CONDA_ENV="$CONDA_BASE/envs/qchem"
conda activate "$CONDA_ENV"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

python - <<'PY'
import importlib.util
import sys

required = ["rdkit", "yaml", "pandas", "sklearn", "h5py", "torch"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing required Python packages in active conda env: " + ", ".join(missing))

import torch
print("python", sys.executable)
print("torch", torch.__version__)
print("torch cuda available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("PyTorch does not see CUDA in the active conda env")
PY

srun --unbuffered --chdir="$PROJECT_ROOT" --ntasks=1 --cpus-per-task=14 --gres=gpu:1 python -m qchem_gnn.validation --config "$CONFIG_FILE"
