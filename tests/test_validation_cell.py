import json

import qchem_gnn.validation as validation
from qchem_gnn.validation import run_one_cell, split_holdout
from tests._validation_fixtures import make_tiny_quantum_dataset


def _tiny_adapt_cfg(tmp_path):
    csv = tmp_path / "esol_tiny.csv"
    rows = ["smiles,y"]
    for i, smi in enumerate(["CO", "CCO", "CCN", "CCC", "CCCC", "CCCO",
                             "CCCN", "CCCCO", "c1ccccc1", "CC", "CCCl", "CCBr"]):
        rows.append(f"{smi},{0.1 * i:.2f}")
    csv.write_text("\n".join(rows) + "\n")
    return {
        "dataset": {"csv": str(csv), "smiles_col": "smiles", "targets": ["y"]},
        "task": "regression",
        "adapter": {"hidden_dims": [8], "dropout": 0.0},
        "training": {"epochs": 2, "lr": 1.0e-3, "batch_size": 4, "patience": 5, "seed": 42},
        "split": {"test_frac": 0.25, "val_frac": 0.25, "seed": 42, "stratify": False},
    }


def _pretrain_cfg():
    return {"hidden_dim": 16, "message_passing_steps": 2, "total_steps": 2,
            "learning_rate": 0.01, "batch_size": 4, "hidden_dim_3d": 16,
            "message_passing_steps_3d": 2}


def test_run_one_cell_produces_rows_and_artifacts(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    pretrain_ds, holdout = split_holdout(dataset, fraction=0.5, seed=0)
    out_dir = tmp_path / "out"
    cell = run_one_cell(
        "quantum", {"teacher_weight": 1.0, "conformer_pool_mode": "energy"}, 0,
        pretrain_ds, holdout.examples, _pretrain_cfg(),
        [{"method": "mlp_head"}], _tiny_adapt_cfg(tmp_path), out_dir,
    )
    assert (out_dir / "quantum_s0.pt").exists()
    assert (out_dir / "quantum_s0_intrinsic.json").exists()
    assert (out_dir / "quantum_s0_mlp_head.json").exists()
    assert cell["intrinsic"]["status"] == "ok"
    assert cell["extrinsic"][0]["method"] == "mlp_head"
    assert cell["extrinsic"][0]["status"] == "ok"
    assert cell["extrinsic"][0]["mae"] is not None


def test_caching_skips_retraining(tmp_path, monkeypatch):
    dataset = make_tiny_quantum_dataset(tmp_path)
    pretrain_ds, holdout = split_holdout(dataset, fraction=0.5, seed=0)
    out_dir = tmp_path / "out"
    args = ("quantum", {"teacher_weight": 1.0, "conformer_pool_mode": "energy"}, 0,
            pretrain_ds, holdout.examples, _pretrain_cfg(),
            [{"method": "mlp_head"}], _tiny_adapt_cfg(tmp_path), out_dir)

    run_one_cell(*args)  # first run trains + caches

    calls = {"n": 0}
    real = validation.contrastive_pretrain_on_dataset

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(validation, "contrastive_pretrain_on_dataset", spy)
    run_one_cell(*args)  # second run should reuse cache
    assert calls["n"] == 0
