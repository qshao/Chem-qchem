import qchem_gnn.cli as cli


def test_preprocess_command_writes_one_cache_per_shard(tmp_path, monkeypatch):
    written = []

    def _fake_preprocess_shard(dataset_root, subset_id, cache_dir, *, overwrite=False):
        path = tmp_path / "cache" / f"shard_{subset_id:03d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        written.append(subset_id)
        return path

    monkeypatch.setattr(cli, "preprocess_shard", _fake_preprocess_shard)

    rc = cli.main([
        "preprocess",
        "--dataset-root", str(tmp_path / "root"),
        "--subset-ids", "0,2,5",
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert rc == 0
    assert written == [0, 2, 5]
    assert (tmp_path / "cache" / "shard_000.pt").exists()
    assert (tmp_path / "cache" / "shard_005.pt").exists()
