from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import torch

from qchem_gnn.dataset_index import DatasetIndex, iter_dataset_indices
from qchem_gnn.minimal import load_quantum_zinc_dataset


class FakeDataset:
    def __init__(self, array):
        self.array = np.asarray(array)

    def __array__(self, dtype=None):
        return np.asarray(self.array, dtype=dtype)


class FakeGroup(dict):
    def __init__(self, attrs=None, **items):
        super().__init__(**items)
        self.attrs = attrs or {}


def _methane_wbi(scale: float) -> np.ndarray:
    wbi = np.zeros((5, 5), dtype=np.float32)
    for idx in range(1, 5):
        wbi[0, idx] = scale
        wbi[idx, 0] = scale
    return wbi


def _write_tiny_shard(tmp_path: Path) -> tuple[Path, Path, Path]:
    csv_path = tmp_path / 'subset_000.csv'
    geo_path = tmp_path / 'coords_000.pkl'
    results_root = tmp_path / 'results'
    results_path = results_root / 'results_000.h5'
    results_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {'smiles': 'C', 'logP': 0.1, 'qed': 0.2, 'SAS': 0.3},
            {'smiles': 'CC', 'logP': 0.4, 'qed': 0.5, 'SAS': 0.6},
        ]
    ).to_csv(csv_path, index=False)

    import pickle

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

    results_path.touch()
    return csv_path, geo_path, results_path


def _fake_hdf5_results():
    return FakeGroup(
        **{
            'subset_0_idx_0': FakeGroup(
                attrs={'smiles': 'C'},
                conf_0=FakeGroup(
                    attrs={'energy': -1.0, 'alpha': 3.0},
                    coords=FakeDataset(np.zeros((5, 3))),
                    chelpg=FakeDataset(np.array([0.1, 0.2, 0.3, 0.4, 0.5])),
                    fukui_plus=FakeDataset(np.array([0.5, 0.4, 0.3, 0.2, 0.1])),
                    iao_pops=FakeDataset(np.array([1.0, 1.1, 1.2, 1.3, 1.4])),
                    wbi=FakeDataset(_methane_wbi(1.0)),
                ),
                conf_1=FakeGroup(
                    attrs={'energy': -2.0, 'alpha': 6.0},
                    coords=FakeDataset(np.ones((5, 3))),
                    chelpg=FakeDataset(np.array([1.1, 1.2, 1.3, 1.4, 1.5])),
                    fukui_plus=FakeDataset(np.array([1.5, 1.4, 1.3, 1.2, 1.1])),
                    iao_pops=FakeDataset(np.array([2.0, 2.1, 2.2, 2.3, 2.4])),
                    wbi=FakeDataset(_methane_wbi(2.0)),
                ),
            )
        }
    )


def test_dataset_index_enumerates_actual_shard_paths_and_molecule_ids(tmp_path):
    csv_path, geo_path, results_path = _write_tiny_shard(tmp_path)
    dataset_root = tmp_path

    indices = list(iter_dataset_indices(dataset_root, subset_ids=[0], limit_per_shard=2, use_results=True))

    assert indices == [
        DatasetIndex(
            subset_id=0,
            csv_path=csv_path,
            geometry_path=geo_path,
            results_path=results_path,
            mol_ids=['subset_0_idx_0', 'subset_0_idx_1'],
        )
    ]


def test_load_quantum_zinc_dataset_uses_proxy_targets_when_results_are_disabled(tmp_path):
    csv_path, geo_path, results_path = _write_tiny_shard(tmp_path)

    index = DatasetIndex(
        subset_id=0,
        csv_path=csv_path,
        geometry_path=geo_path,
        results_path=results_path,
        mol_ids=['subset_0_idx_0', 'subset_0_idx_1'],
    )

    dataset = load_quantum_zinc_dataset(
        index,
        results_handle=_fake_hdf5_results(),
        use_results=False,
    )

    assert len(dataset) == 2
    example = dataset[0]
    assert example.mol_id == 'subset_0_idx_0'
    assert torch.allclose(
        example.node_target,
        torch.tensor(
            [
                [0.6, 0.3],
                [0.1, 0.05],
                [0.1, 0.05],
                [0.1, 0.05],
                [0.1, 0.05],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.allclose(example.graph_target, torch.tensor([0.5, 0.8]))


def test_load_quantum_zinc_dataset_aggregates_fake_hdf5_targets(tmp_path):
    csv_path, geo_path, results_path = _write_tiny_shard(tmp_path)

    dataset = load_quantum_zinc_dataset(
        DatasetIndex(
            subset_id=0,
            csv_path=csv_path,
            geometry_path=geo_path,
            results_path=results_path,
            mol_ids=['subset_0_idx_0'],
        ),
        results_handle=_fake_hdf5_results(),
        use_results=True,
    )

    assert len(dataset) == 1
    example = dataset[0]
    assert example.mol_id == 'subset_0_idx_0'
    assert example.conformer_count == 2
    assert torch.allclose(
        example.node_target,
        torch.tensor(
            [
                [0.6, 1.5],
                [0.7, 1.6],
                [0.8, 1.7],
                [0.9, 1.8],
                [1.0, 1.9],
            ],
            dtype=torch.float32,
        ),
    )
    assert example.edge_target.shape == (8, 1)
    assert torch.allclose(example.edge_target, torch.full((8, 1), 1.5))
    assert torch.allclose(example.graph_target, torch.tensor([-1.5, 4.5]))
