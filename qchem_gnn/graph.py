from __future__ import annotations

from dataclasses import dataclass

import torch
from rdkit import Chem


_BOND_TYPE_TO_INDEX = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}


@dataclass(frozen=True)
class GraphData:
    atomic_numbers: torch.LongTensor
    edge_index: torch.LongTensor
    edge_attr: torch.LongTensor

    @property
    def num_nodes(self) -> int:
        return int(self.atomic_numbers.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


@dataclass(frozen=True)
class GraphBatch:
    atomic_numbers: torch.LongTensor
    edge_index: torch.LongTensor
    edge_attr: torch.LongTensor
    batch: torch.LongTensor
    ptr: torch.LongTensor

    @property
    def num_nodes(self) -> int:
        return int(self.atomic_numbers.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_graphs(self) -> int:
        return int(self.ptr.shape[0] - 1)

    @classmethod
    def from_graphs(cls, graphs: list[GraphData]) -> "GraphBatch":
        atomic_numbers = []
        edge_indices = []
        edge_attrs = []
        batches = []
        ptr = [0]
        node_offset = 0

        for graph_idx, graph in enumerate(graphs):
            atomic_numbers.append(graph.atomic_numbers)
            edge_indices.append(graph.edge_index + node_offset)
            edge_attrs.append(graph.edge_attr)
            batches.append(
                torch.full(
                    (graph.num_nodes,),
                    graph_idx,
                    dtype=torch.long,
                    device=graph.atomic_numbers.device,
                )
            )
            node_offset += graph.num_nodes
            ptr.append(node_offset)

        return cls(
            atomic_numbers=torch.cat(atomic_numbers, dim=0),
            edge_index=torch.cat(edge_indices, dim=1),
            edge_attr=torch.cat(edge_attrs, dim=0),
            batch=torch.cat(batches, dim=0),
            ptr=torch.tensor(ptr, dtype=torch.long, device=graphs[0].atomic_numbers.device),
        )


def build_graph_from_smiles(smiles: str, add_hs: bool = True) -> GraphData:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    if add_hs:
        mol = Chem.AddHs(mol)

    atomic_numbers = torch.tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        dtype=torch.long,
    )

    src = []
    dst = []
    bond_types = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_type = _BOND_TYPE_TO_INDEX.get(bond.GetBondType(), 4)
        src.extend([begin, end])
        dst.extend([end, begin])
        bond_types.extend([bond_type, bond_type])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(bond_types, dtype=torch.long).unsqueeze(-1)

    return GraphData(
        atomic_numbers=atomic_numbers,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
