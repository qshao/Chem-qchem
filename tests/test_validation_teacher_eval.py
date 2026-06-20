import pytest
import torch

from qchem_gnn.conformer import ConformerEncoderBatch
from qchem_gnn.encoder3d import Conformer3DEncoder
from qchem_gnn.quantum_data import compute_target_normalization, normalize_targets
from qchem_gnn.teacher_heads import (
    QuantumTeacherHeads,
    assemble_conformer_targets,
    teacher_loss,
)
from qchem_gnn.validation import evaluate_teacher, split_holdout
from tests._validation_fixtures import make_tiny_quantum_dataset


def test_evaluate_teacher_returns_finite_metrics(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    _, holdout = split_holdout(dataset, fraction=0.5, seed=0)
    norm = compute_target_normalization(dataset)
    enc = Conformer3DEncoder(atom_vocab_size=128, hidden_dim=16, num_message_passing_steps=2)
    teacher = QuantumTeacherHeads(hidden_dim=16)
    metrics = evaluate_teacher(teacher, enc, holdout.examples, norm)
    for prop in ("chelpg", "energy", "iso_polarizability", "wbi"):
        assert torch.isfinite(torch.tensor(metrics[prop]["r"]))
        assert torch.isfinite(torch.tensor(metrics[prop]["mae"]))
        assert -1.0 <= metrics[prop]["r"] <= 1.0


def test_metric_reflects_learning(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    _, holdout = split_holdout(dataset, fraction=0.5, seed=0)
    examples = holdout.examples
    norm = compute_target_normalization(dataset)
    enc = Conformer3DEncoder(atom_vocab_size=128, hidden_dim=16, num_message_passing_steps=2)
    teacher = QuantumTeacherHeads(hidden_dim=16)

    random_metrics = evaluate_teacher(teacher, enc, examples, norm)

    # Fit the teacher heads on the NORMALIZED holdout targets with the encoder frozen.
    batch = ConformerEncoderBatch.from_molecule_conformers(
        [ex.graph for ex in examples],
        [ex.conformer_coords for ex in examples],
        conformer_energies=[ex.conformer_energies for ex in examples],
    )
    with torch.no_grad():
        node_states, conf_emb = enc.forward_with_nodes(
            batch.atomic_numbers, batch.edge_index, batch.positions,
            batch.node_conformer_index, batch.num_conformers,
        )
    node_t, edge_t, graph_t, _ = assemble_conformer_targets(examples)
    node_tn, edge_tn, graph_tn = normalize_targets(node_t, edge_t, graph_t, norm)
    opt = torch.optim.Adam(teacher.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        np_, ep_, gp_ = teacher(node_states, batch.edge_index, conf_emb)
        teacher_loss(np_, ep_, gp_, node_tn, edge_tn, graph_tn).backward()
        opt.step()

    fit_metrics = evaluate_teacher(teacher, enc, examples, norm)
    assert fit_metrics["chelpg"]["mae"] < random_metrics["chelpg"]["mae"]


def test_raises_when_no_usable_examples():
    # The empty-examples guard fires before normalization is used, so {} is fine here.
    with pytest.raises(ValueError):
        enc = Conformer3DEncoder(atom_vocab_size=128, hidden_dim=16, num_message_passing_steps=2)
        evaluate_teacher(QuantumTeacherHeads(hidden_dim=16), enc, [], {})
