from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from random import Random
from typing import Iterable


@dataclass(frozen=True)
class SplitGroups:
    train: list[int]
    val: list[int]
    test: list[int]

    def as_dict(self) -> dict[str, list[int]]:
        return {"train": self.train, "val": self.val, "test": self.test}


def _group_key(molecule_id: str, scaffold: str | None) -> str:
    return scaffold if scaffold else molecule_id


def _group_indices(
    molecule_ids: Iterable[str],
    scaffolds: dict[str, str] | None,
) -> list[tuple[str, list[int]]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, molecule_id in enumerate(molecule_ids):
        groups[_group_key(molecule_id, None if scaffolds is None else scaffolds.get(molecule_id))].append(index)
    return list(groups.items())


def scaffold_or_random_split(
    molecule_ids: Iterable[str],
    scaffolds: dict[str, str] | None = None,
    seed: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, list[int]]:
    molecule_ids = list(molecule_ids)
    if not molecule_ids:
        return {"train": [], "val": [], "test": []}

    grouped_indices = _group_indices(molecule_ids, scaffolds)
    rng = Random(seed)
    rng.shuffle(grouped_indices)

    total = len(molecule_ids)
    target_val = max(0, int(round(total * val_fraction)))
    target_test = max(0, int(round(total * test_fraction)))

    split_assignments = {"train": [], "val": [], "test": []}
    current_counts = {"train": 0, "val": 0, "test": 0}

    for _, indices in grouped_indices:
        if current_counts["test"] < target_test:
            destination = "test"
        elif current_counts["val"] < target_val:
            destination = "val"
        else:
            destination = "train"
        split_assignments[destination].extend(indices)
        current_counts[destination] += len(indices)

    for split_name in split_assignments:
        split_assignments[split_name].sort()

    assigned = sorted(index for indices in split_assignments.values() for index in indices)
    if assigned != list(range(total)):
        raise ValueError("Split did not cover every example exactly once")

    return split_assignments
