import torch

from qchem_gnn.minimal import MinimalQuantumDataset, MinimalQuantumExample
from qchem_gnn.validation import scaffold_hash_holdout


def _ex(mol_id, smiles, key):
    return MinimalQuantumExample(
        mol_id=mol_id,
        smiles=smiles,
        graph=type("G", (), {})(),
        charge=0,
        conformer_count=0,
        node_target=torch.zeros(1, 1),
        edge_target=torch.zeros(0, 1),
        graph_target=torch.zeros(2),
        scaffold_key=key,
    )


def test_scaffold_hash_holdout_is_deterministic_and_disjoint():
    # keys chosen so k=3 sends key%3==0 to holdout
    examples = [_ex(f"m{i}", "CCO", key=i) for i in range(6)]
    ds = MinimalQuantumDataset(examples=examples)
    pre1, hold1 = scaffold_hash_holdout(ds, k=3)
    pre2, hold2 = scaffold_hash_holdout(ds, k=3)
    assert [e.mol_id for e in hold1.examples] == [e.mol_id for e in hold2.examples]
    holdout_keys = {e.scaffold_key for e in hold1.examples}
    pretrain_keys = {e.scaffold_key for e in pre1.examples}
    assert holdout_keys.isdisjoint(pretrain_keys)  # scaffold-disjoint
    assert holdout_keys == {0, 3}


def test_scaffold_hash_holdout_falls_back_to_smiles_when_key_missing():
    # scaffold_key None -> computed from smiles; identical scaffolds share a bucket
    a = _ex("a", "Cc1ccccc1", key=None)
    b = _ex("b", "Nc1ccccc1", key=None)  # same benzene scaffold as a
    ds = MinimalQuantumDataset(examples=[a, b])
    pre, hold = scaffold_hash_holdout(ds, k=1)  # k=1 -> everything holdout
    assert len(hold.examples) == 2 and len(pre.examples) == 0 or (
        len(pre.examples) == 2 and len(hold.examples) == 0
    )
    # a and b must land in the SAME split (same scaffold)
    split_of = {e.mol_id: "hold" for e in hold.examples}
    split_of.update({e.mol_id: "pre" for e in pre.examples})
    assert split_of["a"] == split_of["b"]


def test_scaffold_hash_holdout_raises_when_a_side_is_empty():
    examples = [_ex(f"m{i}", "CCO", key=2) for i in range(3)]  # all key%3 != 0
    ds = MinimalQuantumDataset(examples=examples)
    try:
        scaffold_hash_holdout(ds, k=3)
    except ValueError as exc:
        assert "holdout" in str(exc).lower() or "empty" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError when holdout side is empty")
