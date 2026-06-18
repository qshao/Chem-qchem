import torch

from qchem_gnn.graph import GraphData
from qchem_gnn.shard_cache import SHARD_CACHE_VERSION
from qchem_gnn.validation import _load_dataset
from qchem_gnn.minimal import MinimalQuantumExample


def _ex(mol_id, smiles):
    g = GraphData(
        atomic_numbers=torch.zeros(1, dtype=torch.long),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, dtype=torch.long),
    )
    return MinimalQuantumExample(
        mol_id=mol_id,
        smiles=smiles,
        graph=g,
        charge=0,
        conformer_count=0,
        node_target=torch.zeros(1, 1),
        edge_target=torch.zeros(0, 1),
        graph_target=torch.zeros(2),
        scaffold_key=1,
    )


def test_load_dataset_reads_compact_cache_when_cache_dir_present(tmp_path):
    for sid, mols in ((0, ["CCO"]), (1, ["CCN", "CCC"])):
        torch.save(
            {
                "version": SHARD_CACHE_VERSION,
                "examples": [_ex(f"s{sid}_{i}", s) for i, s in enumerate(mols)],
                "skipped_mol_ids": (),
            },
            tmp_path / f"shard_{sid:03d}.pt",
        )
    cfg = {"cache_dir": str(tmp_path), "subset_ids": [0, 1]}
    ds = _load_dataset(cfg)
    assert len(ds) == 3
    assert [e.mol_id for e in ds.examples] == ["s0_0", "s1_0", "s1_1"]
