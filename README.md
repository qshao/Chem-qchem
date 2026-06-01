# qchem_gnn

`qchem_gnn` is a small geometric deep learning project for learning molecular representations from the ZINC-250K dataset. The current implementation trains a 2D molecular graph neural network that learns from quantum-informed targets and exports reusable embeddings for downstream tasks.

## What is implemented

The codebase currently supports:

- training on small ZINC-250K subsets
- quantum-aware pretraining with auxiliary labels
- checkpoint saving and resume
- embedding export from a saved checkpoint
- evaluation of a saved checkpoint
- downstream linear probe, fine-tuning, Morgan fingerprint baseline, and sample-efficiency runs
- YAML-driven configuration for training and inference

The default workflow is intentionally small and fast so you can validate the pipeline on a minimal subset before scaling up.

## Project layout

- `qchem_gnn/` - model, dataset, training, evaluation, checkpoint, and config code
- `configs/` - runnable minimal YAML examples
- `zinc-250k/` - dataset files, shard metadata, geometry pickles, and quantum outputs
- `tests/` - regression tests for config, CLI, training, inference, and downstream evaluation
- `docs/` - plan documents and implementation notes

## Requirements

- Python 3.13 or newer
- `numpy`
- `pandas`
- `torch`
- `rdkit`
- `pyyaml`
- `pytest` for tests

If you use the HDF5-backed quantum targets, you also need `h5py`.

## Install a virtual environment

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas torch pyyaml rdkit-pypi pytest h5py
```

If `rdkit-pypi` is not available on your platform, install RDKit through conda instead and keep the rest of the dependencies in the virtual environment.

## Run the test suite

```bash
python -m pytest tests -q
```

## Training

The recommended fast-start path is to use one of the example YAML configs in `configs/`.

Minimal training example:

```bash
python -m qchem_gnn.cli train --config configs/minimal_train.yaml
```

Minimal pretraining example:

```bash
python -m qchem_gnn.cli pretrain --config configs/minimal_pretrain.yaml
```

You can still override YAML values on the command line. For example:

```bash
python -m qchem_gnn.cli train \
  --config configs/minimal_train.yaml \
  --epochs 50 \
  --output runs/custom_train.pt
```

### Data modes

The CLI supports two dataset layouts:

- single-shard mode with `csv` and `geometry`
- shard-range mode with `dataset-root` and `subset-ids`

The example configs use shard-range mode with a single shard so the run stays small and fast.

## Inference

### Export embeddings

```bash
python -m qchem_gnn.cli export-embeddings --config configs/export_embeddings.yaml
```

This loads a saved checkpoint and exports one embedding per molecule.

### Evaluate a checkpoint

```bash
python -m qchem_gnn.cli eval --config configs/eval.yaml
```

This reports reconstruction-style metrics for the saved dataset stored in the checkpoint.

### Downstream evaluation

You can run downstream tasks from a checkpoint as well:

```bash
python -m qchem_gnn.cli downstream \
  --checkpoint runs/minimal_train.pt \
  --kind linear-probe
```

Other supported downstream modes are:

- `fine-tune`
- `linear-probe`
- `morgan-baseline`
- `sample-efficiency`

## YAML configuration

The project supports YAML configuration files for all main entry points:

- `train`
- `pretrain`
- `export-embeddings`
- `eval`
- `downstream`

Example configs live in `configs/` and are intended as runnable starting points rather than placeholders.

Explicit command-line arguments override values loaded from YAML.

## Dataset notes

The ZINC-250K dataset directory contains:

- the main CSV table of molecular properties
- shard CSVs under `subsets/`
- conformer geometry pickles under `geometries/`
- quantum result files under `results/`

The dataset and file structure are documented in `zinc-250k/DATASET_LOG.md`.

## Reproducibility

Training and pretraining checkpoints store:

- model state
- optimizer state
- scheduler state when available
- loss history
- target normalization
- dataset and split metadata
- run metadata, including a run ID and git commit when available

## Notes

- The current model is a 2D graph neural network that learns from quantum-informed supervision during training.
- HDF5-backed quantum targets are optional and only used when explicitly enabled.
- The minimal configs are designed to keep the first run small and fast.

