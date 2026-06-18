from types import SimpleNamespace

import torch

from qchem_gnn.eval import build_scaffold_negative_mask


def _ex(smi):
    return SimpleNamespace(smiles=smi)


def test_shared_scaffold_entries_are_true():
    # Toluene and aniline both reduce to benzene under Murcko; ethanol is unique.
    examples = [_ex("Cc1ccccc1"), _ex("Nc1ccccc1"), _ex("CCO")]
    mask = build_scaffold_negative_mask(examples)
    assert mask.shape == (3, 3)
    assert mask[0, 1].item() is True
    assert mask[1, 0].item() is True


def test_unique_scaffolds_all_false():
    examples = [_ex("CO"), _ex("CCO"), _ex("CCN")]
    mask = build_scaffold_negative_mask(examples)
    assert not mask.any()


def test_diagonal_always_false():
    examples = [_ex("Cc1ccccc1"), _ex("Nc1ccccc1")]
    mask = build_scaffold_negative_mask(examples)
    assert not mask[0, 0].item()
    assert not mask[1, 1].item()


def test_cross_entries_with_unique_scaffold_are_false():
    examples = [_ex("Cc1ccccc1"), _ex("Nc1ccccc1"), _ex("CCO")]
    mask = build_scaffold_negative_mask(examples)
    assert not mask[0, 2].item()
    assert not mask[2, 0].item()
    assert not mask[1, 2].item()
    assert not mask[2, 1].item()


def test_returns_bool_cpu_tensor():
    examples = [_ex("CO"), _ex("CCO")]
    mask = build_scaffold_negative_mask(examples)
    assert mask.dtype == torch.bool
    assert not mask.is_cuda


def test_scaffold_key_same_for_shared_murcko_scaffold():
    from qchem_gnn.eval import scaffold_key_from_smiles
    # toluene and aniline both reduce to benzene under Murcko
    assert scaffold_key_from_smiles("Cc1ccccc1") == scaffold_key_from_smiles("Nc1ccccc1")


def test_scaffold_key_differs_for_distinct_scaffold():
    from qchem_gnn.eval import scaffold_key_from_smiles
    assert scaffold_key_from_smiles("Cc1ccccc1") != scaffold_key_from_smiles("CCO")


def test_scaffold_key_deterministic_across_calls():
    from qchem_gnn.eval import scaffold_key_from_smiles
    assert scaffold_key_from_smiles("Nc1ccccc1") == scaffold_key_from_smiles("Nc1ccccc1")


def test_scaffold_key_unparseable_falls_back_to_smiles():
    from qchem_gnn.eval import scaffold_key_from_smiles
    # an unparseable string still yields a stable, self-consistent key
    assert scaffold_key_from_smiles("not_a_smiles") == scaffold_key_from_smiles("not_a_smiles")


def test_scaffold_mask_from_keys_groups_equal_keys():
    import torch
    from qchem_gnn.eval import scaffold_mask_from_keys
    mask = scaffold_mask_from_keys([10, 10, 20])
    assert mask.dtype == torch.bool
    assert mask.shape == (3, 3)
    assert mask[0, 1] and mask[1, 0]
    assert not mask[0, 2] and not mask[1, 2]
    assert not mask[0, 0] and not mask[1, 1] and not mask[2, 2]
