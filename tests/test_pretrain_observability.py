import json
from pathlib import Path

import h5py
import pytest
import torch

from qchem_gnn.contrastive_pretrain import contrastive_pretrain_on_dataset
from qchem_gnn.quantum_data import load_quantum_zinc_dataset
from tests._quantum_fixtures import write_synthetic_results_h5
from tests._validation_fixtures import make_tiny_quantum_dataset


def _varied_target_dataset(tmp_path):
    """Tiny dataset whose per-molecule energy and polarizability vary, so the
    Boltzmann-averaged supervised stats have non-degenerate std (like real data)."""
    mols = [("CO", 6), ("CCO", 9), ("CCN", 10), ("CCC", 11)]
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "subset_044.csv"
    csv_path.write_text(
        "smiles,logP,qed,SAS\n"
        + "".join(f"{smiles},0.0,0.5,2.0\n" for smiles, _ in mols)
    )
    h5_path = tmp_path / "results_044.h5"
    with h5py.File(h5_path, "w") as handle:
        for idx, (smiles, n_atoms) in enumerate(mols):
            base = -115.0 + idx * 5.0
            tmp_single = tmp_path / f"_tmp_{idx}.h5"
            write_synthetic_results_h5(
                tmp_single,
                mol_id=f"subset_44_idx_{idx}",
                smiles=smiles,
                n_atoms=n_atoms,
                energies=(base, base - 0.0005),
                polar_base=10.0 + idx * 3.0,
                seed=idx,
            )
            with h5py.File(tmp_single, "r") as src:
                src.copy(src[f"subset_44_idx_{idx}"], handle, f"subset_44_idx_{idx}")
    with h5py.File(h5_path, "r") as handle:
        return load_quantum_zinc_dataset(
            csv_path,
            geometry_path=tmp_path / "coords_044.pkl",
            limit=len(mols),
            results_handle=handle,
            use_results=True,
        )


def test_teacher_loss_is_normalized_scale(tmp_path):
    # Save and restore the global PyTorch RNG state so this test does not
    # perturb the seed seen by subsequent tests that lack explicit seeding.
    rng_state = torch.get_rng_state()
    try:
        dataset = _varied_target_dataset(tmp_path)
        result = contrastive_pretrain_on_dataset(
            dataset,
            hidden_dim=16,
            hidden_dim_3d=16,
            total_steps=4,
            batch_size=4,
            teacher_weight=1.0,
            conformer_pool_mode="energy",
            seed=0,
        )
        assert result.loss_history
        # Pre-fix the raw-energy teacher MSE pushes this above 1e4; normalized it is O(10).
        assert max(result.loss_history) < 1e3
    finally:
        torch.set_rng_state(rng_state)


def test_metrics_jsonl_written_with_val_keys(tmp_path):
    train_ds = make_tiny_quantum_dataset(tmp_path / "train")
    val_ds = make_tiny_quantum_dataset(tmp_path / "val")
    metrics_path = tmp_path / "run.metrics.jsonl"
    contrastive_pretrain_on_dataset(
        train_ds, val_dataset=val_ds, hidden_dim=16, hidden_dim_3d=16,
        total_steps=4, batch_size=4, log_every=2, seed=0, metrics_path=metrics_path,
    )
    lines = metrics_path.read_text().splitlines()
    assert len(lines) == 2  # steps 2 and 4
    rec = json.loads(lines[0])
    for key in ("step", "total_steps", "train_loss", "train_supervised",
                "train_contrastive", "train_teacher", "steps_per_sec", "wall_seconds",
                "val_loss", "val_supervised", "val_contrastive", "val_teacher"):
        assert key in rec
    assert rec["step"] == 2


def test_metrics_jsonl_omits_val_keys_without_val_dataset(tmp_path):
    train_ds = make_tiny_quantum_dataset(tmp_path)
    metrics_path = tmp_path / "run.metrics.jsonl"
    contrastive_pretrain_on_dataset(
        train_ds, hidden_dim=16, hidden_dim_3d=16, total_steps=2, batch_size=4,
        log_every=2, seed=0, metrics_path=metrics_path,
    )
    rec = json.loads(metrics_path.read_text().splitlines()[0])
    assert "train_loss" in rec
    assert "val_loss" not in rec


def test_per_term_breakdown_isolates_terms(tmp_path):
    # With teacher and contrastive weights zero, total ≈ supervised and the other terms ≈ 0.
    train_ds = make_tiny_quantum_dataset(tmp_path)
    metrics_path = tmp_path / "run.metrics.jsonl"
    contrastive_pretrain_on_dataset(
        train_ds, hidden_dim=16, hidden_dim_3d=16, total_steps=2, batch_size=4,
        log_every=2, seed=0, teacher_weight=0.0, contrastive_weight=0.0,
        metrics_path=metrics_path,
    )
    rec = json.loads(metrics_path.read_text().splitlines()[0])
    assert abs(rec["train_teacher"]) < 1e-6
    assert abs(rec["train_contrastive"]) < 1e-6
    assert rec["train_loss"] == pytest.approx(rec["train_supervised"], rel=1e-5, abs=1e-6)


def test_metrics_jsonl_appends_on_resume(tmp_path):
    train_ds = make_tiny_quantum_dataset(tmp_path)
    metrics_path = tmp_path / "run.metrics.jsonl"
    ckpt = tmp_path / "run.ckpt.pt"
    contrastive_pretrain_on_dataset(
        train_ds, hidden_dim=16, hidden_dim_3d=16, total_steps=2, batch_size=4,
        log_every=2, seed=0, checkpoint_path=ckpt, checkpoint_every=2,
        metrics_path=metrics_path,
    )
    first = len(metrics_path.read_text().splitlines())
    contrastive_pretrain_on_dataset(
        train_ds, hidden_dim=16, hidden_dim_3d=16, total_steps=4, batch_size=4,
        log_every=2, seed=0, checkpoint_path=ckpt, checkpoint_every=2, resume=True,
        metrics_path=metrics_path,
    )
    second = len(metrics_path.read_text().splitlines())
    assert second > first  # appended, not truncated
