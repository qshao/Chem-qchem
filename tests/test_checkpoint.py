from __future__ import annotations

from pathlib import Path

import torch

from qchem_gnn.checkpoint import build_checkpoint_state, load_checkpoint, save_checkpoint
from qchem_gnn.model import MolecularQuantumGNN


def test_checkpoint_save_load_round_trip_model_state(tmp_path: Path):
    path = tmp_path / "checkpoint.pt"

    model = MolecularQuantumGNN(
        atom_vocab_size=128,
        bond_vocab_size=8,
        hidden_dim=16,
        num_message_passing_steps=1,
        graph_targets=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    state = build_checkpoint_state(
        loss_history=[1.0, 0.5],
        embeddings=torch.zeros(2, 16),
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        epoch=3,
        global_step=42,
        target_normalization={
            "node_mean": torch.tensor([1.0, 2.0]),
            "node_std": torch.tensor([3.0, 4.0]),
        },
        dataset_config={
            "csv": "subset_000.csv",
            "dataset_root": None,
            "subset_ids": (0, 1),
            "geometry": None,
            "results": None,
            "use_results": True,
            "limit": 16,
            "limit_per_shard": 16,
        },
        split_metadata={"subset_ids": [0, 1]},
        model_config={
            "atom_vocab_size": 128,
            "bond_vocab_size": 8,
            "hidden_dim": 16,
            "num_message_passing_steps": 1,
            "graph_targets": 2,
        },
        run_metadata={"num_examples": 2, "limit": 16, "limit_per_shard": 16, "epochs": 2, "run_id": "abc123", "git_commit": None},
        scheduler_state_dict=scheduler.state_dict(),
    )

    save_checkpoint(path, state)
    loaded = load_checkpoint(path)

    assert path.exists()
    assert loaded["epoch"] == 3
    assert loaded["global_step"] == 42
    assert loaded["dataset_config"] == state["dataset_config"]
    assert loaded["split_metadata"] == state["split_metadata"]
    assert torch.equal(
        loaded["target_normalization"]["node_mean"],
        state["target_normalization"]["node_mean"],
    )
    assert torch.equal(
        loaded["model_state_dict"]["atom_head.weight"],
        state["model_state_dict"]["atom_head.weight"],
    )
    assert loaded["scheduler_state_dict"] == state["scheduler_state_dict"]
    assert loaded["run_metadata"]["run_id"] == "abc123"

    round_tripped_model = MolecularQuantumGNN(**loaded["model_config"])
    round_tripped_model.load_state_dict(loaded["model_state_dict"])

    for original, restored in zip(model.parameters(), round_tripped_model.parameters(), strict=True):
        assert torch.allclose(original, restored)
