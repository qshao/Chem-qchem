import os
from pathlib import Path

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
    fp = _build_fingerprint(dict(_fp(), epochs=999, extra="ignored"))
    assert "epochs" not in fp
    assert "extra" not in fp
    assert fp["hidden_dim"] == 16


def test_validate_fingerprint_passes_on_match():
    _validate_fingerprint(_fp(), _fp())  # no raise


def test_validate_fingerprint_raises_and_names_field():
    with pytest.raises(CheckpointMismatchError, match="hidden_dim"):
        _validate_fingerprint(_fp(hidden_dim=16), _fp(hidden_dim=32))


def test_atomic_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "quantum_s0.ckpt.pt"
    payload = {"version": PRETRAIN_CHECKPOINT_VERSION, "epoch": 3,
               "config_fingerprint": _fp()}
    _atomic_save_checkpoint(path, payload)
    assert path.exists()
    assert not (tmp_path / "quantum_s0.ckpt.pt.tmp").exists()
    loaded = _load_checkpoint(path)
    assert loaded["epoch"] == 3


def test_load_checkpoint_rejects_wrong_version(tmp_path):
    path = tmp_path / "bad.ckpt.pt"
    torch.save({"version": 999, "epoch": 1}, path)
    with pytest.raises(CheckpointMismatchError, match="version"):
        _load_checkpoint(path)


def test_load_checkpoint_rejects_unreadable_file(tmp_path):
    path = tmp_path / "corrupt.ckpt.pt"
    path.write_bytes(b"not a torch file")
    with pytest.raises(CheckpointMismatchError):
        _load_checkpoint(path)
