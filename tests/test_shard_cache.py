from pathlib import Path

import torch

from qchem_gnn.shard_cache import (
    SHARD_CACHE_VERSION,
    load_compact_shard,
    load_compact_shards,
    preprocess_shard,
)
from qchem_gnn.minimal import MinimalQuantumDataset, MinimalQuantumExample
from qchem_gnn.graph import GraphData


def _fake_example(mol_id, smiles):
    g = GraphData(  # minimal serialisable graph; not inspected by cache I/O
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
    )


def test_example_has_scaffold_key_field_default_none():
    ex = _fake_example("m0", "CCO")
    assert ex.scaffold_key is None


def test_load_compact_shard_roundtrip(tmp_path):
    examples = [_fake_example("m0", "CCO"), _fake_example("m1", "Cc1ccccc1")]
    path = tmp_path / "shard_007.pt"
    torch.save(
        {"version": SHARD_CACHE_VERSION, "examples": examples, "skipped_mol_ids": ("m9",)},
        path,
    )
    loaded, skipped = load_compact_shard(path)
    assert [e.mol_id for e in loaded] == ["m0", "m1"]
    assert skipped == ("m9",)


def test_load_compact_shard_rejects_bad_version(tmp_path):
    path = tmp_path / "shard_000.pt"
    torch.save({"version": 999, "examples": [], "skipped_mol_ids": ()}, path)
    try:
        load_compact_shard(path)
    except ValueError as exc:
        assert "version" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for version mismatch")


def test_load_compact_shards_concatenates(tmp_path):
    for sid, smis in ((0, ["CCO"]), (3, ["CCN", "CCC"])):
        examples = [_fake_example(f"s{sid}_{i}", s) for i, s in enumerate(smis)]
        torch.save(
            {"version": SHARD_CACHE_VERSION, "examples": examples, "skipped_mol_ids": ()},
            tmp_path / f"shard_{sid:03d}.pt",
        )
    ds = load_compact_shards(tmp_path, [0, 3])
    assert isinstance(ds, MinimalQuantumDataset)
    assert len(ds) == 3
    assert [e.mol_id for e in ds.examples] == ["s0_0", "s3_0", "s3_1"]


def test_preprocess_shard_writes_compact_cache_with_scaffold_keys(tmp_path, monkeypatch):
    # Stub the heavy loader so the test stays fast and offline.
    import qchem_gnn.shard_cache as sc

    fake = MinimalQuantumDataset(
        examples=[_fake_example("m0", "Cc1ccccc1"), _fake_example("m1", "Nc1ccccc1")],
        skipped_mol_ids=("m2",),
    )
    monkeypatch.setattr(sc, "load_quantum_zinc_subset_range", lambda *a, **k: fake)

    out = preprocess_shard(tmp_path / "root", 5, tmp_path / "cache")
    assert out == tmp_path / "cache" / "shard_005.pt"

    examples, skipped = load_compact_shard(out)
    assert skipped == ("m2",)
    # toluene + aniline share the benzene Murcko scaffold -> equal keys
    assert examples[0].scaffold_key == examples[1].scaffold_key
    assert examples[0].scaffold_key is not None


def test_preprocess_shard_skips_existing(tmp_path, monkeypatch):
    import qchem_gnn.shard_cache as sc

    calls = {"n": 0}

    def _counting_loader(*a, **k):
        calls["n"] += 1
        return MinimalQuantumDataset(examples=[_fake_example("m0", "CCO")], skipped_mol_ids=())

    monkeypatch.setattr(sc, "load_quantum_zinc_subset_range", _counting_loader)

    preprocess_shard(tmp_path / "root", 1, tmp_path / "cache")
    preprocess_shard(tmp_path / "root", 1, tmp_path / "cache")  # should skip
    assert calls["n"] == 1
