import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from qchem_gnn.cli import main


def _write_tiny_dataset(tmp_path: Path, *, smiles_a: str = 'C', smiles_b: str = 'CC'):
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / 'subset_000.csv'
    geo_path = tmp_path / 'coords_000.pkl'

    pd.DataFrame(
        [
            {'smiles': smiles_a, 'logP': 0.1, 'qed': 0.2, 'SAS': 0.3},
            {'smiles': smiles_b, 'logP': 0.4, 'qed': 0.5, 'SAS': 0.6},
        ]
    ).to_csv(csv_path, index=False)

    with geo_path.open('wb') as handle:
        pickle.dump(
            {
                'subset_0_idx_0': {
                    'smiles': smiles_a,
                    'charge': 0,
                    'symbols': ['C', 'H', 'H', 'H', 'H'],
                    'atomic_nums': [6, 1, 1, 1, 1],
                    'conformers': [np.zeros((5, 3), dtype=np.float32)],
                },
                'subset_0_idx_1': {
                    'smiles': smiles_b,
                    'charge': 0,
                    'symbols': ['C', 'C'] + ['H'] * 6,
                    'atomic_nums': [6, 6] + [1] * 6,
                    'conformers': [np.zeros((8, 3), dtype=np.float32)],
                },
            },
            handle,
        )

    return csv_path, geo_path


def _train_checkpoint(
    tmp_path: Path,
    *,
    csv_path: Path,
    geo_path: Path,
    output_name: str,
    extra_args: list[str] | None = None,
) -> Path:
    output_path = tmp_path / output_name
    args = [
        'train',
        '--csv',
        str(csv_path),
        '--geometry',
        str(geo_path),
        '--limit',
        '2',
        '--epochs',
        '2',
        '--output',
        str(output_path),
    ]
    if extra_args:
        args[1:1] = extra_args
    exit_code = main(args)
    assert exit_code == 0
    assert output_path.exists()
    return output_path



def test_train_cli_config_overrides_yaml_dataset_mode(tmp_path):
    csv_path, geo_path = _write_tiny_dataset(tmp_path / 'cli-dataset')
    cli_output = tmp_path / 'cli-dataset-mode.pt'
    config_path = tmp_path / 'train-dataset-root.yaml'
    config_path.write_text(
        f"""
command: train
dataset:
  dataset_root: {tmp_path / 'yaml-root'}
  subset_ids: [0]
  limit_per_shard: 1
model:
  hidden_dim: 24
  message_passing_steps: 1
training:
  epochs: 3
  learning_rate: 0.01
outputs:
  checkpoint: {tmp_path / 'yaml-dataset-root.pt'}
""",
        encoding='utf-8',
    )

    exit_code = main(
        [
            'train',
            '--config',
            str(config_path),
            '--csv',
            str(csv_path),
            '--geometry',
            str(geo_path),
            '--limit',
            '2',
            '--epochs',
            '4',
            '--output',
            str(cli_output),
        ]
    )

    assert exit_code == 0
    payload = torch.load(cli_output, map_location='cpu')
    assert payload['run_metadata']['resolved_config']['dataset']['csv'] == str(csv_path)
    assert payload['run_metadata']['resolved_config']['dataset']['dataset_root'] is None
    assert payload['run_metadata']['resolved_config']['dataset']['subset_ids'] == []
    assert payload['run_metadata']['resolved_config']['model']['hidden_dim'] == 24
    assert payload['run_metadata']['resolved_config']['outputs']['checkpoint'] == str(cli_output)


def test_train_cli_uses_yaml_config_and_cli_overrides(tmp_path):
    csv_path, geo_path = _write_tiny_dataset(tmp_path)
    yaml_output = tmp_path / 'yaml-train.pt'
    cli_output = tmp_path / 'cli-train.pt'
    yaml_metrics = tmp_path / 'yaml-train.json'
    cli_metrics = tmp_path / 'cli-train.json'
    config_path = tmp_path / 'train.yaml'
    config_path.write_text(
        f"""
command: train
dataset:
  csv: {csv_path}
  geometry: {geo_path}
  limit: 1
  use_results: true
model:
  hidden_dim: 24
  message_passing_steps: 1
training:
  epochs: 3
  learning_rate: 0.01
outputs:
  checkpoint: {yaml_output}
  metrics: {yaml_metrics}
""",
        encoding='utf-8',
    )

    exit_code = main(
        [
            'train',
            '--config',
            str(config_path),
            '--limit',
            '2',
            '--epochs',
            '4',
            '--output',
            str(cli_output),
            '--metrics-output',
            str(cli_metrics),
        ]
    )

    assert exit_code == 0
    assert cli_output.exists()
    assert not yaml_output.exists()
    payload = torch.load(cli_output, map_location='cpu')
    assert len(payload['loss_history']) == 4
    assert payload['embeddings'].shape[0] == 2
    assert payload['run_metadata']['resolved_config']['training']['epochs'] == 4
    assert payload['run_metadata']['resolved_config']['dataset']['limit'] == 2
    assert payload['run_metadata']['resolved_config']['dataset']['use_results'] is True
    assert payload['run_metadata']['resolved_config']['model']['hidden_dim'] == 24
    assert payload['run_metadata']['resolved_config']['outputs']['checkpoint'] == str(cli_output)
    assert cli_metrics.exists()


def test_train_cli_resume_respects_cli_overrides(tmp_path):
    base_csv, base_geo = _write_tiny_dataset(tmp_path / 'base-dataset')
    resume_csv, resume_geo = _write_tiny_dataset(tmp_path / 'resume-dataset', smiles_a='N', smiles_b='NC')
    checkpoint_path = tmp_path / 'base-checkpoint.pt'
    resumed_output = tmp_path / 'resumed-overrides.pt'

    base_exit = main(
        [
            'train',
            '--csv',
            str(base_csv),
            '--geometry',
            str(base_geo),
            '--limit',
            '2',
            '--epochs',
            '1',
            '--output',
            str(checkpoint_path),
        ]
    )
    assert base_exit == 0

    resumed_exit = main(
        [
            'train',
            '--resume-from',
            str(checkpoint_path),
            '--csv',
            str(resume_csv),
            '--geometry',
            str(resume_geo),
            '--epochs',
            '1',
            '--output',
            str(resumed_output),
        ]
    )
    assert resumed_exit == 0
    payload = torch.load(resumed_output, map_location='cpu')
    assert payload['run_metadata']['resolved_config']['dataset']['csv'] == str(resume_csv)
    assert payload['run_metadata']['resolved_config']['dataset']['dataset_root'] is None
    assert payload['run_metadata']['resolved_config']['model']['hidden_dim'] == 32
    assert payload['run_metadata']['resolved_config']['training']['resume_from'] == str(checkpoint_path)
    assert payload['epoch'] == 2


def test_pretrain_cli_uses_yaml_config_and_cli_overrides(tmp_path):
    csv_path, geo_path = _write_tiny_dataset(tmp_path)
    yaml_output = tmp_path / 'yaml-pretrain.pt'
    cli_output = tmp_path / 'cli-pretrain.pt'
    config_path = tmp_path / 'pretrain.yaml'
    config_path.write_text(
        f"""
command: pretrain
dataset:
  csv: {csv_path}
  geometry: {geo_path}
  limit: 1
  use_results: true
model:
  hidden_dim: 20
  message_passing_steps: 1
training:
  epochs: 2
  learning_rate: 0.01
  aux_weight: 0.25
outputs:
  checkpoint: {yaml_output}
""",
        encoding='utf-8',
    )

    exit_code = main(
        [
            'pretrain',
            '--config',
            str(config_path),
            '--limit',
            '2',
            '--epochs',
            '5',
            '--output',
            str(cli_output),
        ]
    )

    assert exit_code == 0
    assert cli_output.exists()
    assert not yaml_output.exists()
    payload = torch.load(cli_output, map_location='cpu')
    assert len(payload['loss_history']) == 5
    assert payload['embeddings'].shape[0] == 2
    assert payload['run_metadata']['resolved_config']['training']['epochs'] == 5
    assert payload['run_metadata']['resolved_config']['training']['aux_weight'] == 0.25
    assert payload['run_metadata']['resolved_config']['dataset']['limit'] == 2
    assert payload['run_metadata']['resolved_config']['outputs']['checkpoint'] == str(cli_output)


def test_export_eval_and_downstream_cli_use_yaml_configs_and_cli_overrides(tmp_path):
    csv_path, geo_path = _write_tiny_dataset(tmp_path)
    checkpoint_path = _train_checkpoint(
        tmp_path,
        csv_path=csv_path,
        geo_path=geo_path,
        output_name='checkpoint.pt',
    )

    export_yaml_output = tmp_path / 'yaml-embeddings.pt'
    export_cli_output = tmp_path / 'cli-embeddings.pt'
    export_config = tmp_path / 'export.yaml'
    export_config.write_text(
        f"""
command: export-embeddings
inference:
  checkpoint: {checkpoint_path}
  output: {export_yaml_output}
""",
        encoding='utf-8',
    )
    export_exit = main(
        [
            'export-embeddings',
            '--config',
            str(export_config),
            '--output',
            str(export_cli_output),
        ]
    )
    assert export_exit == 0
    assert export_cli_output.exists()
    assert not export_yaml_output.exists()
    exported = torch.load(export_cli_output, map_location='cpu')
    assert exported['embeddings'].shape == (2, 32)
    assert exported['mol_ids'] == ['subset_0_idx_0', 'subset_0_idx_1']

    eval_yaml_output = tmp_path / 'yaml-eval.pt'
    eval_cli_output = tmp_path / 'cli-eval.pt'
    eval_config = tmp_path / 'eval.yaml'
    eval_config.write_text(
        f"""
command: eval
inference:
  checkpoint: {checkpoint_path}
  output: {eval_yaml_output}
""",
        encoding='utf-8',
    )
    eval_exit = main(
        [
            'eval',
            '--config',
            str(eval_config),
            '--output',
            str(eval_cli_output),
        ]
    )
    assert eval_exit == 0
    assert eval_cli_output.exists()
    assert not eval_yaml_output.exists()
    eval_metrics = torch.load(eval_cli_output, map_location='cpu')
    assert eval_metrics['num_examples'] == 2
    assert 'loss' in eval_metrics

    downstream_yaml_output = tmp_path / 'yaml-downstream.pt'
    downstream_cli_output = tmp_path / 'cli-downstream.pt'
    downstream_config = tmp_path / 'downstream.yaml'
    downstream_config.write_text(
        f"""
command: downstream
inference:
  checkpoint: {checkpoint_path}
  output: {downstream_yaml_output}
downstream:
  kind: linear-probe
  epochs: 3
  learning_rate: 0.02
  fractions: [0.1, 1.0]
  seed: 11
""",
        encoding='utf-8',
    )
    downstream_exit = main(
        [
            'downstream',
            '--config',
            str(downstream_config),
            '--kind',
            'sample-efficiency',
            '--output',
            str(downstream_cli_output),
            '--fractions',
            '1.0',
            '--seed',
            '7',
        ]
    )
    assert downstream_exit == 0
    assert downstream_cli_output.exists()
    assert not downstream_yaml_output.exists()
    downstream_metrics = torch.load(downstream_cli_output, map_location='cpu')
    assert set(downstream_metrics) == {1.0}
    assert downstream_metrics[1.0]['train']


def test_train_cli_writes_checkpoint(tmp_path):
    csv_path = tmp_path / 'subset_000.csv'
    geo_path = tmp_path / 'coords_000.pkl'
    output_path = tmp_path / 'checkpoint.pt'
    metrics_path = tmp_path / 'metrics.json'

    pd.DataFrame(
        [
            {'smiles': 'C', 'logP': 0.1, 'qed': 0.2, 'SAS': 0.3},
            {'smiles': 'CC', 'logP': 0.4, 'qed': 0.5, 'SAS': 0.6},
        ]
    ).to_csv(csv_path, index=False)

    with geo_path.open('wb') as handle:
        pickle.dump(
            {
                'subset_0_idx_0': {
                    'smiles': 'C',
                    'charge': 0,
                    'symbols': ['C', 'H', 'H', 'H', 'H'],
                    'atomic_nums': [6, 1, 1, 1, 1],
                    'conformers': [np.zeros((5, 3), dtype=np.float32)],
                },
                'subset_0_idx_1': {
                    'smiles': 'CC',
                    'charge': 0,
                    'symbols': ['C', 'C'] + ['H'] * 6,
                    'atomic_nums': [6, 6] + [1] * 6,
                    'conformers': [np.zeros((8, 3), dtype=np.float32)],
                },
            },
            handle,
        )

    exit_code = main(
        [
            'train',
            '--csv',
            str(csv_path),
            '--geometry',
            str(geo_path),
            '--limit',
            '2',
            '--epochs',
            '20',
            '--use-results',
            '--metrics-output',
            str(metrics_path),
            '--output',
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert metrics_path.exists()
    payload = torch.load(output_path, map_location='cpu')
    assert payload['embeddings'].shape == (2, 32)
    assert len(payload['loss_history']) == 20
    assert payload['target_normalization']['node_mean'].shape == (2,)
    metrics = __import__('json').loads(metrics_path.read_text())
    assert metrics['num_examples'] == 2
    assert metrics['final_loss'] == metrics['loss_history'][-1]
    assert metrics['run_id']


def test_train_cli_accepts_dataset_root_use_results_and_custom_results_dir(tmp_path, monkeypatch):
    dataset_root = tmp_path / 'zinc-250k'
    subsets_dir = dataset_root / 'subsets'
    geometries_dir = dataset_root / 'geometries'
    results_dir = tmp_path / 'custom-results'
    subsets_dir.mkdir(parents=True)
    geometries_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    output_path = tmp_path / 'checkpoint_root.pt'
    metrics_path = tmp_path / 'metrics_root.json'

    for subset_id, smiles in [(0, 'C'), (1, 'CC')]:
        pd.DataFrame(
            [
                {'smiles': smiles, 'logP': 0.1, 'qed': 0.2, 'SAS': 0.3},
            ]
        ).to_csv(subsets_dir / f'subset_{subset_id:03d}.csv', index=False)
        with (geometries_dir / f'coords_{subset_id:03d}.pkl').open('wb') as handle:
            pickle.dump(
                {
                    f'subset_{subset_id}_idx_0': {
                        'smiles': smiles,
                        'charge': 0,
                        'symbols': ['C', 'H', 'H', 'H', 'H'] if subset_id == 0 else ['C', 'C'] + ['H'] * 6,
                        'atomic_nums': [6, 1, 1, 1, 1] if subset_id == 0 else [6, 6] + [1] * 6,
                        'conformers': [
                            np.zeros((5, 3), dtype=np.float32)
                            if subset_id == 0
                            else np.zeros((8, 3), dtype=np.float32)
                        ],
                    }
                },
                handle,
            )
        (results_dir / f'results_{subset_id:03d}.h5').touch()

    fake_results = {
        'subset_0_idx_0': {
            'conf_0': {
                'coords': np.zeros((5, 3), dtype=np.float32),
                'chelpg': np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
                'iao_pops': np.array([1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float32),
                'wbi': np.full((5, 5), 1.0, dtype=np.float32),
            }
        }
    }

    fake_h5py = types.ModuleType('h5py')
    fake_h5py.File = lambda path, mode: fake_results
    monkeypatch.setitem(sys.modules, 'h5py', fake_h5py)

    exit_code = main(
        [
            'train',
            '--dataset-root',
            str(dataset_root),
            '--subset-ids',
            '0,1',
            '--results',
            str(results_dir),
            '--limit-per-shard',
            '1',
            '--epochs',
            '10',
            '--use-results',
            '--metrics-output',
            str(metrics_path),
            '--output',
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = torch.load(output_path, map_location='cpu')
    assert payload['run_metadata']['num_examples'] == 2
    assert payload['run_metadata']['run_id']
    assert payload['embeddings'].shape[0] == 2
    assert payload['target_normalization']['graph_std'].shape == (2,)
    metrics = __import__('json').loads(metrics_path.read_text())
    assert metrics['num_examples'] == 2


def test_train_cli_can_resume_and_export_embeddings(tmp_path, monkeypatch):
    dataset_root = tmp_path / 'zinc-250k'
    subsets_dir = dataset_root / 'subsets'
    geometries_dir = dataset_root / 'geometries'
    results_dir = tmp_path / 'custom-results'
    subsets_dir.mkdir(parents=True)
    geometries_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    output_path = tmp_path / 'checkpoint_root.pt'
    metrics_path = tmp_path / 'metrics_root.json'
    resumed_output = tmp_path / 'resumed.pt'
    embeddings_path = tmp_path / 'embeddings.pt'

    pd.DataFrame(
        [
            {'smiles': 'C', 'logP': 0.1, 'qed': 0.2, 'SAS': 0.3},
        ]
    ).to_csv(subsets_dir / 'subset_000.csv', index=False)
    with (geometries_dir / 'coords_000.pkl').open('wb') as handle:
        pickle.dump(
            {
                'subset_0_idx_0': {
                    'smiles': 'C',
                    'charge': 0,
                    'symbols': ['C', 'H', 'H', 'H', 'H'],
                    'atomic_nums': [6, 1, 1, 1, 1],
                    'conformers': [np.zeros((5, 3), dtype=np.float32)],
                }
            },
            handle,
        )
    (results_dir / 'results_000.h5').touch()

    fake_results = {
        'subset_0_idx_0': {
            'conf_0': {
                'coords': np.zeros((5, 3), dtype=np.float32),
                'chelpg': np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
                'iao_pops': np.array([1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float32),
                'wbi': np.full((5, 5), 1.0, dtype=np.float32),
            }
        }
    }

    fake_h5py = types.ModuleType('h5py')
    fake_h5py.File = lambda path, mode: fake_results
    monkeypatch.setitem(sys.modules, 'h5py', fake_h5py)

    first_exit = main(
        [
            'train',
            '--dataset-root',
            str(dataset_root),
            '--subset-ids',
            '0',
            '--results',
            str(results_dir),
            '--limit-per-shard',
            '1',
            '--epochs',
            '2',
            '--use-results',
            '--metrics-output',
            str(metrics_path),
            '--output',
            str(output_path),
        ]
    )
    assert first_exit == 0

    resumed_exit = main(
        [
            'train',
            '--dataset-root',
            str(dataset_root),
            '--subset-ids',
            '0',
            '--results',
            str(results_dir),
            '--limit-per-shard',
            '1',
            '--epochs',
            '1',
            '--use-results',
            '--resume-from',
            str(output_path),
            '--output',
            str(resumed_output),
        ]
    )
    assert resumed_exit == 0
    resumed_payload = torch.load(resumed_output, map_location='cpu')
    assert resumed_payload['epoch'] == 3
    assert resumed_payload['global_step'] == 3

    export_exit = main(
        [
            'export-embeddings',
            '--checkpoint',
            str(output_path),
            '--output',
            str(embeddings_path),
        ]
    )
    assert export_exit == 0
    exported = torch.load(embeddings_path, map_location='cpu')
    assert exported['embeddings'].shape == (1, 32)
    assert exported['mol_ids'] == ['subset_0_idx_0']
    assert exported['checkpoint_run_metadata']['run_id']
    metrics = __import__('json').loads(metrics_path.read_text())
    assert metrics['num_examples'] == 1


def test_train_cli_can_resume_without_restating_dataset_args(tmp_path, monkeypatch):
    dataset_root = tmp_path / 'zinc-250k'
    subsets_dir = dataset_root / 'subsets'
    geometries_dir = dataset_root / 'geometries'
    results_dir = tmp_path / 'custom-results'
    subsets_dir.mkdir(parents=True)
    geometries_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    checkpoint_path = tmp_path / 'checkpoint.pt'
    resumed_output = tmp_path / 'resumed.pt'

    pd.DataFrame(
        [
            {'smiles': 'C', 'logP': 0.1, 'qed': 0.2, 'SAS': 0.3},
        ]
    ).to_csv(subsets_dir / 'subset_000.csv', index=False)
    with (geometries_dir / 'coords_000.pkl').open('wb') as handle:
        pickle.dump(
            {
                'subset_0_idx_0': {
                    'smiles': 'C',
                    'charge': 0,
                    'symbols': ['C', 'H', 'H', 'H', 'H'],
                    'atomic_nums': [6, 1, 1, 1, 1],
                    'conformers': [np.zeros((5, 3), dtype=np.float32)],
                }
            },
            handle,
        )
    (results_dir / 'results_000.h5').touch()

    fake_results = {
        'subset_0_idx_0': {
            'conf_0': {
                'coords': np.zeros((5, 3), dtype=np.float32),
                'chelpg': np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
                'iao_pops': np.array([1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float32),
                'wbi': np.full((5, 5), 1.0, dtype=np.float32),
            }
        }
    }
    fake_h5py = types.ModuleType('h5py')
    fake_h5py.File = lambda path, mode: fake_results
    monkeypatch.setitem(sys.modules, 'h5py', fake_h5py)

    first_exit = main(
        [
            'train',
            '--dataset-root',
            str(dataset_root),
            '--subset-ids',
            '0',
            '--results',
            str(results_dir),
            '--limit-per-shard',
            '1',
            '--epochs',
            '1',
            '--use-results',
            '--output',
            str(checkpoint_path),
        ]
    )
    assert first_exit == 0

    resumed_exit = main(
        [
            'train',
            '--resume-from',
            str(checkpoint_path),
            '--epochs',
            '1',
            '--output',
            str(resumed_output),
        ]
    )
    assert resumed_exit == 0
    resumed_payload = torch.load(resumed_output, map_location='cpu')
    assert resumed_payload['epoch'] == 2
    assert resumed_payload['global_step'] == 2

def test_eval_cli_writes_metrics(tmp_path, monkeypatch):
    dataset_root = tmp_path / 'zinc-250k'
    subsets_dir = dataset_root / 'subsets'
    geometries_dir = dataset_root / 'geometries'
    results_dir = tmp_path / 'custom-results'
    subsets_dir.mkdir(parents=True)
    geometries_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    checkpoint_path = tmp_path / 'checkpoint.pt'
    metrics_path = tmp_path / 'metrics.pt'

    pd.DataFrame(
        [
            {'smiles': 'C', 'logP': 0.1, 'qed': 0.2, 'SAS': 0.3},
        ]
    ).to_csv(subsets_dir / 'subset_000.csv', index=False)
    with (geometries_dir / 'coords_000.pkl').open('wb') as handle:
        pickle.dump(
            {
                'subset_0_idx_0': {
                    'smiles': 'C',
                    'charge': 0,
                    'symbols': ['C', 'H', 'H', 'H', 'H'],
                    'atomic_nums': [6, 1, 1, 1, 1],
                    'conformers': [np.zeros((5, 3), dtype=np.float32)],
                }
            },
            handle,
        )
    (results_dir / 'results_000.h5').touch()

    fake_results = {
        'subset_0_idx_0': {
            'conf_0': {
                'coords': np.zeros((5, 3), dtype=np.float32),
                'chelpg': np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
                'iao_pops': np.array([1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float32),
                'wbi': np.full((5, 5), 1.0, dtype=np.float32),
            }
        }
    }
    fake_h5py = types.ModuleType('h5py')
    fake_h5py.File = lambda path, mode: fake_results
    monkeypatch.setitem(sys.modules, 'h5py', fake_h5py)

    train_exit = main(
        [
            'train',
            '--dataset-root',
            str(dataset_root),
            '--subset-ids',
            '0',
            '--results',
            str(results_dir),
            '--limit-per-shard',
            '1',
            '--epochs',
            '1',
            '--use-results',
            '--output',
            str(checkpoint_path),
        ]
    )
    assert train_exit == 0

    eval_exit = main(
        [
            'eval',
            '--checkpoint',
            str(checkpoint_path),
            '--output',
            str(metrics_path),
        ]
    )
    assert eval_exit == 0
    metrics = torch.load(metrics_path, map_location='cpu')
    assert metrics['num_examples'] == 1
    assert 'loss' in metrics
    assert 'embedding_norm_mean' in metrics


def _write_contrastive_inputs(tmp_path: Path):
    csv_path = tmp_path / "subset_000.csv"
    geo_path = tmp_path / "coords_000.pkl"
    pd.DataFrame(
        [
            {"smiles": "C", "logP": 0.1, "qed": 0.2, "SAS": 0.3},
            {"smiles": "CC", "logP": 0.4, "qed": 0.5, "SAS": 0.6},
            {"smiles": "CCC", "logP": 0.2, "qed": 0.3, "SAS": 0.4},
            {"smiles": "CCO", "logP": 0.5, "qed": 0.6, "SAS": 0.7},
        ]
    ).to_csv(csv_path, index=False)
    import pickle as _pickle
    _pickle.dump(
        {
            "subset_0_idx_0": {"smiles": "C", "charge": 0, "atomic_nums": [6, 1, 1, 1, 1], "conformers": [np.zeros((5, 3), dtype=np.float32)]},
            "subset_0_idx_1": {"smiles": "CC", "charge": 0, "atomic_nums": [6, 6] + [1] * 6, "conformers": [np.zeros((8, 3), dtype=np.float32)]},
            "subset_0_idx_2": {"smiles": "CCC", "charge": 0, "atomic_nums": [6, 6, 6] + [1] * 8, "conformers": [np.zeros((11, 3), dtype=np.float32)]},
            "subset_0_idx_3": {"smiles": "CCO", "charge": 0, "atomic_nums": [6, 6, 8] + [1] * 6, "conformers": [np.zeros((9, 3), dtype=np.float32)]},
        },
        geo_path.open("wb"),
    )
    return csv_path, geo_path


def test_cli_contrastive_pretrain_then_export(tmp_path: Path):
    csv_path, geo_path = _write_contrastive_inputs(tmp_path)
    checkpoint = tmp_path / "contrastive.pt"
    embeddings = tmp_path / "emb.pt"

    code = main(
        [
            "contrastive-pretrain",
            "--csv", str(csv_path),
            "--geometry", str(geo_path),
            "--limit", "4",
            "--epochs", "20",
            "--hidden-dim", "16",
            "--batch-size", "4",
            "--output", str(checkpoint),
        ]
    )
    assert code == 0
    assert checkpoint.exists()

    # The checkpoint must work with the unchanged export-embeddings path.
    code = main(["export-embeddings", "--checkpoint", str(checkpoint), "--output", str(embeddings)])
    assert code == 0
    assert embeddings.exists()
