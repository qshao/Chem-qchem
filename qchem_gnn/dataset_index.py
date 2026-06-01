from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class DatasetIndex:
    subset_id: int
    csv_path: Path
    geometry_path: Path
    results_path: Path | None
    mol_ids: list[str]


def _parse_subset_id(path: Path) -> int:
    match = re.search(r"subset_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not parse subset id from {path}")
    return int(match.group(1))


def _subset_mol_ids(csv_path: Path, subset_id: int, limit: int | None = None) -> list[str]:
    df = pd.read_csv(csv_path, usecols=["smiles"])
    if limit is not None:
        df = df.head(limit)
    return [f"subset_{subset_id}_idx_{row_idx}" for row_idx in range(len(df))]


def _preferred_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _cache_path_for_root(dataset_root: Path, cache_path: Path | None = None) -> Path:
    return cache_path if cache_path is not None else dataset_root / ".qchem_gnn_index_cache.json"


def _load_dataset_index_cache(cache_path: Path):
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text())


def _write_dataset_index_cache(cache_path: Path, *, indices: Iterable[DatasetIndex], context: dict[str, object]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "context": context,
        "indices": [
            {
                "subset_id": index.subset_id,
                "csv_path": str(index.csv_path),
                "geometry_path": str(index.geometry_path),
                "results_path": str(index.results_path) if index.results_path is not None else None,
                "mol_ids": list(index.mol_ids),
            }
            for index in indices
        ],
    }
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def build_dataset_index(
    csv_path: str | Path,
    geometry_path: str | Path,
    results_path: str | Path | None = None,
    limit: int | None = None,
) -> DatasetIndex:
    csv_path = Path(csv_path)
    geometry_path = Path(geometry_path)
    results_path = Path(results_path) if results_path is not None else None
    subset_id = _parse_subset_id(csv_path)

    return DatasetIndex(
        subset_id=subset_id,
        csv_path=csv_path,
        geometry_path=geometry_path,
        results_path=results_path,
        mol_ids=_subset_mol_ids(csv_path, subset_id, limit=limit),
    )


def iter_dataset_indices(
    dataset_root: str | Path,
    subset_ids: list[int],
    limit_per_shard: int | None = None,
    use_results: bool = False,
    results_root: str | Path | None = None,
    cache_path: str | Path | None = None,
):
    dataset_root = Path(dataset_root)
    cache_path = _cache_path_for_root(dataset_root, Path(cache_path) if cache_path is not None else None)
    requested_context = {
        "subset_ids": list(subset_ids),
        "limit_per_shard": limit_per_shard,
        "use_results": use_results,
        "results_root": str(Path(results_root) if results_root is not None else dataset_root / "results"),
    }
    cached_payload = _load_dataset_index_cache(cache_path)
    if cached_payload:
        cached_context = cached_payload.get("context", {})
        if cached_context == requested_context:
            cached_indices = [
                DatasetIndex(
                    subset_id=int(entry["subset_id"]),
                    csv_path=Path(entry["csv_path"]),
                    geometry_path=Path(entry["geometry_path"]),
                    results_path=Path(entry["results_path"]) if entry.get("results_path") else None,
                    mol_ids=list(entry["mol_ids"]),
                )
                for entry in cached_payload.get("indices", [])
            ]
            for index in cached_indices:
                yield index
            return

    results_root = Path(results_root) if results_root is not None else dataset_root / "results"
    built_indices: list[DatasetIndex] = []

    for subset_id in subset_ids:
        csv_path = _preferred_path(
            dataset_root / "subsets" / f"subset_{subset_id:03d}.csv",
            dataset_root / f"subset_{subset_id:03d}.csv",
        )
        geometry_path = _preferred_path(
            dataset_root / "geometries" / f"coords_{subset_id:03d}.pkl",
            dataset_root / f"coords_{subset_id:03d}.pkl",
        )
        results_path = None
        if use_results:
            results_path = _preferred_path(
                results_root / f"results_{subset_id:03d}.h5",
                dataset_root / f"results_{subset_id:03d}.h5",
            )

        index = DatasetIndex(
            subset_id=subset_id,
            csv_path=csv_path,
            geometry_path=geometry_path,
            results_path=results_path,
            mol_ids=_subset_mol_ids(csv_path, subset_id, limit=limit_per_shard),
        )
        built_indices.append(index)
        yield index

    _write_dataset_index_cache(cache_path, indices=built_indices, context=requested_context)
