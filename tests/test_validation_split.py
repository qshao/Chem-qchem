import pytest

from qchem_gnn.validation import split_holdout
from tests._validation_fixtures import make_tiny_quantum_dataset


def test_split_is_deterministic_disjoint_and_sized(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)  # 4 molecules
    pre_a, hold_a = split_holdout(dataset, fraction=0.5, seed=7)
    pre_b, hold_b = split_holdout(dataset, fraction=0.5, seed=7)

    ids_pre = {ex.mol_id for ex in pre_a.examples}
    ids_hold = {ex.mol_id for ex in hold_a.examples}

    # deterministic
    assert [ex.mol_id for ex in hold_a.examples] == [ex.mol_id for ex in hold_b.examples]
    assert [ex.mol_id for ex in pre_a.examples] == [ex.mol_id for ex in pre_b.examples]
    # disjoint and complete
    assert ids_pre.isdisjoint(ids_hold)
    assert ids_pre | ids_hold == {ex.mol_id for ex in dataset.examples}
    # sized: round(4 * 0.5) == 2
    assert len(hold_a.examples) == 2
    assert len(pre_a.examples) == 2


def test_different_seed_changes_holdout(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    _, hold_0 = split_holdout(dataset, fraction=0.5, seed=0)
    _, hold_1 = split_holdout(dataset, fraction=0.5, seed=999)
    # at least the ordering/content differs for some seed pair
    assert [e.mol_id for e in hold_0.examples] != [e.mol_id for e in hold_1.examples]


def test_holdout_at_least_one(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    _, hold = split_holdout(dataset, fraction=0.01, seed=0)
    assert len(hold.examples) == 1
