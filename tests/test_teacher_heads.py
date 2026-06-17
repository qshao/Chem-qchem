# tests/test_teacher_heads.py
import torch

from qchem_gnn.teacher_heads import (
    QuantumTeacherHeads,
    assemble_conformer_targets,
    teacher_loss,
)


class _FakeExample:
    def __init__(self, c, n, e):
        self.conformer_node_targets = torch.randn(c, n, 1)
        self.conformer_edge_targets = torch.randn(c, e, 1)
        self.conformer_graph_targets = torch.randn(c, 2)
        self.conformer_energies = torch.randn(c)


def test_assemble_orders_molecule_then_conformer():
    ex0 = _FakeExample(c=2, n=3, e=4)
    ex1 = _FakeExample(c=1, n=3, e=4)
    node_t, edge_t, graph_t, energies = assemble_conformer_targets([ex0, ex1])
    assert node_t.shape == (2 * 3 + 1 * 3, 1)
    assert edge_t.shape == (2 * 4 + 1 * 4, 1)
    assert graph_t.shape == (3, 2)
    assert energies.shape == (3,)
    # first molecule's first conformer node block matches its source
    assert torch.allclose(node_t[:3], ex0.conformer_node_targets[0])


def test_heads_output_shapes():
    heads = QuantumTeacherHeads(hidden_dim=16)
    node_states = torch.randn(6, 16)            # 2 conformers x 3 atoms
    edge_index = torch.tensor([[0, 1, 3, 4], [1, 0, 4, 3]])
    conf_emb = torch.randn(2, 16)
    node_pred, edge_pred, graph_pred = heads(node_states, edge_index, conf_emb)
    assert node_pred.shape == (6, 1)
    assert edge_pred.shape == (4, 1)
    assert graph_pred.shape == (2, 2)


def test_teacher_loss_decreases_on_overfit():
    torch.manual_seed(0)
    heads = QuantumTeacherHeads(hidden_dim=16)
    node_states = torch.randn(6, 16)
    edge_index = torch.tensor([[0, 1, 3, 4], [1, 0, 4, 3]])
    conf_emb = torch.randn(2, 16)
    node_t = torch.randn(6, 1)
    edge_t = torch.randn(4, 1)
    graph_t = torch.randn(2, 2)
    opt = torch.optim.Adam(heads.parameters(), lr=0.05)

    def step():
        opt.zero_grad()
        np_, ep_, gp_ = heads(node_states, edge_index, conf_emb)
        loss = teacher_loss(np_, ep_, gp_, node_t, edge_t, graph_t)
        loss.backward()
        opt.step()
        return float(loss)

    first = step()
    for _ in range(200):
        last = step()
    assert last < first * 0.5
