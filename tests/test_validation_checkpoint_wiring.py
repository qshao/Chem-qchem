import pytest

import qchem_gnn.validation as V
from tests._validation_fixtures import make_tiny_quantum_dataset

_PRETRAIN_CFG = {
    "hidden_dim": 16,
    "message_passing_steps": 2,
    "total_steps": 2,
    "learning_rate": 0.01,
    "resume": True,
    "checkpoint_every": 5,
}


def _run_cell(monkeypatch, tmp_path, overwrite):
    captured = {}

    def fake_pretrain(dataset, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop before real training")

    monkeypatch.setattr(V, "contrastive_pretrain_on_dataset", fake_pretrain)
    dataset = make_tiny_quantum_dataset(tmp_path)
    V.run_one_cell(
        "quantum", {}, 0, dataset, [], _PRETRAIN_CFG, [], {}, tmp_path,
        overwrite=overwrite,
    )
    return captured


def test_wiring_passes_resume_and_checkpoint_path(monkeypatch, tmp_path):
    captured = _run_cell(monkeypatch, tmp_path, overwrite=False)
    assert captured["resume"] is True
    assert captured["checkpoint_every"] == 5
    assert captured["checkpoint_path"].name == "quantum_s0.ckpt.pt"


def test_overwrite_forces_resume_false_and_deletes_checkpoint(monkeypatch, tmp_path):
    stale = tmp_path / "quantum_s0.ckpt.pt"
    stale.write_bytes(b"stale")
    captured = _run_cell(monkeypatch, tmp_path, overwrite=True)
    assert captured["resume"] is False
    assert not stale.exists()  # deleted before training started


def test_checkpoint_mismatch_propagates(monkeypatch, tmp_path):
    def boom(dataset, **kwargs):
        raise V.CheckpointMismatchError("hidden_dim differs")

    monkeypatch.setattr(V, "contrastive_pretrain_on_dataset", boom)
    dataset = make_tiny_quantum_dataset(tmp_path)
    with pytest.raises(V.CheckpointMismatchError):
        V.run_one_cell(
            "quantum", {}, 0, dataset, [], _PRETRAIN_CFG, [], {}, tmp_path,
            overwrite=False,
        )
