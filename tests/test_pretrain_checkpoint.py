import pytest
import torch

from qchem_gnn.contrastive_pretrain import (
    PRETRAIN_CHECKPOINT_VERSION,
    CheckpointMismatchError,
    _build_fingerprint,
    _validate_fingerprint,
    _atomic_save_checkpoint,
    _load_checkpoint,
)


def _fp(**overrides):
    base = {
        "hidden_dim": 16, "num_message_passing_steps": 2,
        "hidden_dim_3d": 16, "num_rbf": 16, "cutoff": 5.0,
        "num_message_passing_steps_3d": 2,
        "batch_size": 4, "learning_rate": 0.01,
        "supervised_weight": 1.0, "contrastive_weight": 1.0, "teacher_weight": 1.0,
        "temperature": 0.1, "energy_temperature": 298.15, "conformer_pool_mode": "mean",
        "contrastive_loss": "infonce",
        "vicreg_sim_weight": 25.0, "vicreg_var_weight": 25.0, "vicreg_cov_weight": 1.0,
        "use_scaffold_negmask": False, "seed": 0, "node_targets": 1, "num_examples": 4,
    }
    base.update(overrides)
    return base


def test_build_fingerprint_selects_only_known_fields():
    fp = _build_fingerprint(dict(_fp(), total_steps=999, extra="ignored"))
    assert "total_steps" not in fp
    assert "extra" not in fp
    assert fp["hidden_dim"] == 16


def test_validate_fingerprint_passes_on_match():
    _validate_fingerprint(_fp(), _fp())  # no raise


def test_validate_fingerprint_raises_and_names_field():
    with pytest.raises(CheckpointMismatchError, match="hidden_dim"):
        _validate_fingerprint(_fp(hidden_dim=16), _fp(hidden_dim=32))


def test_atomic_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "quantum_s0.ckpt.pt"
    payload = {
        "version": PRETRAIN_CHECKPOINT_VERSION, "step": 3,
        "cycle_order": [0, 1, 2, 3], "cycle_start": 0,
        "config_fingerprint": _fp(),
    }
    _atomic_save_checkpoint(path, payload)
    assert path.exists()
    assert not (tmp_path / "quantum_s0.ckpt.pt.tmp").exists()
    loaded = _load_checkpoint(path)
    assert loaded["step"] == 3


def test_load_checkpoint_rejects_wrong_version(tmp_path):
    path = tmp_path / "bad.ckpt.pt"
    torch.save({"version": 999, "step": 1}, path)
    with pytest.raises(CheckpointMismatchError, match="version"):
        _load_checkpoint(path)


def test_load_checkpoint_rejects_unreadable_file(tmp_path):
    path = tmp_path / "corrupt.ckpt.pt"
    path.write_bytes(b"not a torch file")
    with pytest.raises(CheckpointMismatchError):
        _load_checkpoint(path)


from qchem_gnn.contrastive_pretrain import contrastive_pretrain_on_dataset
from tests._validation_fixtures import make_tiny_quantum_dataset


def _ckpt(tmp_path):
    return tmp_path / "quantum_s0.ckpt.pt"


def test_checkpoint_written_every_n_and_at_final(tmp_path):
    # tiny dataset: 4 examples, batch_size=4 → 1 step per cycle
    dataset = make_tiny_quantum_dataset(tmp_path)
    path = _ckpt(tmp_path)
    contrastive_pretrain_on_dataset(
        dataset, hidden_dim=16, hidden_dim_3d=16, total_steps=4, batch_size=4, seed=0,
        checkpoint_path=path, checkpoint_every=2,
    )
    assert path.exists()
    loaded = _load_checkpoint(path)
    assert loaded["step"] == 4
    assert len(loaded["loss_history"]) == 4


def test_resume_continues_and_matches_uninterrupted_run(tmp_path):
    # An interrupted 2+2 step run must equal a single 4-step run (RNG + optimizer restored).
    ds_a = make_tiny_quantum_dataset(tmp_path / "a")
    full = contrastive_pretrain_on_dataset(
        ds_a, hidden_dim=16, hidden_dim_3d=16, total_steps=4, batch_size=4, seed=0,
    )

    ds_b = make_tiny_quantum_dataset(tmp_path / "b")
    path = tmp_path / "b" / "quantum_s0.ckpt.pt"
    contrastive_pretrain_on_dataset(
        ds_b, hidden_dim=16, hidden_dim_3d=16, total_steps=2, batch_size=4, seed=0,
        checkpoint_path=path, checkpoint_every=2,
    )
    resumed = contrastive_pretrain_on_dataset(
        ds_b, hidden_dim=16, hidden_dim_3d=16, total_steps=4, batch_size=4, seed=0,
        checkpoint_path=path, checkpoint_every=2, resume=True,
    )
    assert len(resumed.loss_history) == 4
    assert torch.allclose(resumed.embeddings, full.embeddings, atol=1e-5)


def test_resume_rejects_fingerprint_mismatch(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    path = _ckpt(tmp_path)
    contrastive_pretrain_on_dataset(
        dataset, hidden_dim=16, hidden_dim_3d=16, total_steps=2, batch_size=4, seed=0,
        checkpoint_path=path, checkpoint_every=2,
    )
    with pytest.raises(CheckpointMismatchError, match="hidden_dim"):
        contrastive_pretrain_on_dataset(
            dataset, hidden_dim=32, hidden_dim_3d=16, total_steps=4, batch_size=4, seed=0,
            checkpoint_path=path, checkpoint_every=2, resume=True,
        )


def test_resume_allows_raising_total_steps(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    path = _ckpt(tmp_path)
    contrastive_pretrain_on_dataset(
        dataset, hidden_dim=16, hidden_dim_3d=16, total_steps=2, batch_size=4, seed=0,
        checkpoint_path=path, checkpoint_every=2,
    )
    resumed = contrastive_pretrain_on_dataset(
        dataset, hidden_dim=16, hidden_dim_3d=16, total_steps=5, batch_size=4, seed=0,
        checkpoint_path=path, checkpoint_every=2, resume=True,
    )
    assert len(resumed.loss_history) == 5


def test_resume_past_completion_runs_no_steps(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    path = _ckpt(tmp_path)
    contrastive_pretrain_on_dataset(
        dataset, hidden_dim=16, hidden_dim_3d=16, total_steps=4, batch_size=4, seed=0,
        checkpoint_path=path, checkpoint_every=2,
    )
    with pytest.warns(UserWarning, match="no training will run"):
        resumed = contrastive_pretrain_on_dataset(
            dataset, hidden_dim=16, hidden_dim_3d=16, total_steps=4, batch_size=4, seed=0,
            checkpoint_path=path, checkpoint_every=2, resume=True,
        )
    assert len(resumed.loss_history) == 4
    assert torch.isfinite(resumed.embeddings).all()


def test_resume_missing_checkpoint_warns_and_starts_fresh(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    path = _ckpt(tmp_path)  # does not exist yet
    with pytest.warns(UserWarning, match="no checkpoint"):
        result = contrastive_pretrain_on_dataset(
            dataset, hidden_dim=16, hidden_dim_3d=16, total_steps=2, batch_size=4, seed=0,
            checkpoint_path=path, checkpoint_every=2, resume=True,
        )
    assert len(result.loss_history) == 2


def test_resume_false_ignores_existing_checkpoint(tmp_path):
    # A stale checkpoint with a different fingerprint must NOT be read when resume=False.
    dataset = make_tiny_quantum_dataset(tmp_path)
    path = _ckpt(tmp_path)
    _atomic_save_checkpoint(path, {
        "version": PRETRAIN_CHECKPOINT_VERSION, "step": 99,
        "cycle_order": [0, 1, 2, 3], "cycle_start": 0,
        "config_fingerprint": {"hidden_dim": 999},
    })
    result = contrastive_pretrain_on_dataset(
        dataset, hidden_dim=16, hidden_dim_3d=16, total_steps=2, batch_size=4, seed=0,
        checkpoint_path=path, checkpoint_every=2, resume=False,
    )
    assert len(result.loss_history) == 2  # trained fresh, did not jump to step 99
