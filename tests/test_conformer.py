import pytest
import torch

from qchem_gnn.conformer import ConformerBatch, pool_conformer_embeddings
from qchem_gnn.graph import GraphBatch, build_graph_from_smiles


def test_conformer_batch_preserves_graph_and_conformer_boundaries():
    graphs = [
        build_graph_from_smiles("C"),
        build_graph_from_smiles("CC"),
    ]
    graph_batch = GraphBatch.from_graphs(graphs)
    conformer_batch = ConformerBatch(
        graph_batch=graph_batch,
        conformer_index=torch.tensor([0, 0, 1], dtype=torch.long),
        conformer_energy=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
    )

    assert conformer_batch.graph_batch.num_graphs == 2
    assert conformer_batch.graph_batch.num_nodes == graph_batch.num_nodes
    assert conformer_batch.conformer_index.tolist() == [0, 0, 1]
    assert conformer_batch.conformer_counts_per_graph().tolist() == [2, 1]
    assert conformer_batch.conformer_energy.shape == (3,)
    assert conformer_batch.conformer_energy.dtype == torch.float32


def test_conformer_batch_rejects_invalid_graph_boundaries():
    graph_batch = GraphBatch.from_graphs([build_graph_from_smiles("C"), build_graph_from_smiles("CC")])

    with pytest.raises(ValueError, match="outside graph_batch"):
        ConformerBatch(
            graph_batch=graph_batch,
            conformer_index=torch.tensor([0, 2], dtype=torch.long),
            conformer_energy=torch.tensor([0.1, 0.2], dtype=torch.float32),
        )


def test_pool_conformer_embeddings_is_permutation_invariant():
    conformer_embeddings = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=torch.float32,
    )
    conformer_energy = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)
    permutation = torch.tensor([2, 0, 1], dtype=torch.long)

    pooled = pool_conformer_embeddings(
        conformer_embeddings,
        conformer_energy=conformer_energy,
        mode="mean",
    )
    permuted_pooled = pool_conformer_embeddings(
        conformer_embeddings[permutation],
        conformer_energy=conformer_energy[permutation],
        mode="mean",
    )

    assert pooled.shape == (3,)
    assert torch.allclose(pooled, permuted_pooled)


def test_pool_conformer_embeddings_supports_energy_weighting_and_empty_inputs():
    conformer_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=torch.float32,
    )
    conformer_energy = torch.tensor([0.0, 2.0], dtype=torch.float32)

    weighted = pool_conformer_embeddings(
        conformer_embeddings,
        conformer_energy=conformer_energy,
        mode="weighted",
    )
    assert weighted.shape == (2,)
    assert weighted[0] > weighted[1]

    with pytest.raises(ValueError, match="at least one conformer"):
        pool_conformer_embeddings(torch.empty(0, 2), mode="mean")
