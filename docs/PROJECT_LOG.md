# Project Log

This file records the main work completed in the `qchem_gnn` project so the implementation history does not need to be reconstructed from the codebase.

## What Was Built

- A 2D molecular graph neural network for quantum-informed representation learning.
- Multi-task training and pretraining support for atom, edge, graph, and auxiliary targets.
- Checkpoint save, load, and resume support.
- Embedding export and checkpoint evaluation entry points.
- Downstream evaluation for:
  - fine-tuning
  - linear probing
  - Morgan fingerprint baseline
  - sample efficiency
- YAML configuration support for all main CLI commands.
- Bash launchers for training and inference.

## Key Files

- `qchem_gnn/cli.py` - command-line interface for training and inference.
- `qchem_gnn/config.py` - YAML config loading, validation, and CLI mapping.
- `qchem_gnn/model.py` - 2D molecular GNN encoder and task heads.
- `qchem_gnn/quantum_data.py` - dataset loading and quantum target aggregation.
- `qchem_gnn/minimal.py` - minimal fast-path dataset and training loop.
- `qchem_gnn/pretrain.py` - auxiliary pretraining loop.
- `qchem_gnn/eval.py` - downstream evaluation helpers.
- `configs/` - runnable minimal YAML examples.
- `scripts/train.sh` - training launcher.
- `scripts/infer.sh` - inference launcher.

## Dataset Handling

- The local `zinc-250k/` folder contains the ZINC-250K dataset, geometry pickles, quantum outputs, and project-specific notes.
- That folder is intentionally excluded from the GitHub repository because it is too large to upload.
- The published repository includes the code, configs, tests, and launchers only.

## Validation

- The test suite currently passes with `python -m pytest tests -q`.
- The YAML config flow, training CLI, inference CLI, and launcher scripts were all checked against the current implementation.

## Notes

- The default workflow is intentionally minimal so it can run on a small shard before scaling up.
- HDF5-backed quantum targets are supported but opt-in.
- The scripts default to the minimal YAML configs in `configs/`.
