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
