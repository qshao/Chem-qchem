from qchem_gnn.adapt.registry import METHODS, get_method


def test_registry_has_three_methods():
    assert set(METHODS) == {"mlp_head", "finetune", "engine"}
    assert get_method("engine").name == "engine"


def test_registry_unknown_raises():
    import pytest
    with pytest.raises(KeyError):
        get_method("nope")
