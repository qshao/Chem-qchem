from qchem_gnn.splits import scaffold_or_random_split


def test_scaffold_or_random_split_is_molecule_level_only():
    molecule_ids = ["mol1", "mol1", "mol2", "mol3", "mol3", "mol4"]
    scaffolds = {
        "mol1": "scaffold-a",
        "mol2": "scaffold-b",
        "mol3": "scaffold-a",
        "mol4": "scaffold-c",
    }

    split = scaffold_or_random_split(molecule_ids, scaffolds=scaffolds, seed=0)

    index_to_split = {}
    for split_name, indices in split.items():
        for index in indices:
            index_to_split[index] = split_name

    assert index_to_split[0] == index_to_split[1]
    assert index_to_split[3] == index_to_split[4]
    assert set(index_to_split.values()) <= {"train", "val", "test"}
    assert sorted(index_to_split) == list(range(len(molecule_ids)))
