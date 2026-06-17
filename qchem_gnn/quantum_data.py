from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset_index import DatasetIndex, build_dataset_index, iter_dataset_indices
from .graph import build_graph_from_smiles


def _parse_subset_id(path: Path) -> int:
    match = re.search(r"subset_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not parse subset id from {path}")
    return int(match.group(1))


def _load_pickle_mapping(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    with path.open("rb") as handle:
        return pickle.load(handle)


def _as_tensor(value) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def _conformer_coords_from_geometry(geometry: dict, num_nodes: int) -> list[torch.Tensor] | None:
    conformers = geometry.get("conformers")
    if not conformers:
        return None
    coords = []
    for conformer in conformers:
        tensor = _as_tensor(conformer)
        if tensor.ndim != 2 or tensor.shape[0] != num_nodes or tensor.shape[1] != 3:
            continue
        coords.append(tensor)
    return coords if coords else None


def _scalarize(value) -> float:
    tensor = _as_tensor(value)
    if tensor.numel() == 1:
        return float(tensor.item())
    return float(tensor.mean().item())


def _build_proxy_targets(graph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    node_target = torch.stack(
        [graph.atomic_numbers.float() / 10.0, graph.atomic_numbers.float() / 20.0],
        dim=-1,
    )
    edge_target = graph.edge_attr.float() / 4.0
    graph_target = torch.tensor(
        [graph.num_nodes / 10.0, graph.num_edges / 10.0],
        dtype=torch.float32,
    )
    return node_target, edge_target, graph_target


def _result_group(results_handle, mol_id: str):
    if results_handle is None:
        return None
    try:
        return results_handle[mol_id]
    except Exception:
        return None


def _conformer_groups(mol_group) -> list:
    conf_names = sorted(name for name in mol_group.keys() if name.startswith("conf_"))
    return [mol_group[name] for name in conf_names]


def _dataset_to_tensor(conf_group, name: str):
    value = conf_group.get(name)
    if value is None:
        return None
    if hasattr(value, "__array__"):
        return _as_tensor(value)
    return _as_tensor(value[()])


def _collect_conf_value(conf_group, names: tuple[str, ...]):
    for name in names:
        value = _dataset_to_tensor(conf_group, name)
        if value is not None:
            return value
    return None


def _aggregate_targets(graph, mol_group) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    conf_groups = _conformer_groups(mol_group)
    if not conf_groups:
        raise ValueError("Result group contains no conformer groups")

    chelpg_values = []
    iao_values = []
    wbi_values = []
    energies = []
    polar_values = []

    for conf_group in conf_groups:
        chelpg = _collect_conf_value(conf_group, ("chelpg",))
        iao_pops = _collect_conf_value(conf_group, ("iao_pops",))
        wbi = _collect_conf_value(conf_group, ("wbi",))

        if chelpg is None or iao_pops is None or wbi is None:
            raise ValueError("Missing required quantum descriptor in conformer group")

        chelpg_values.append(chelpg)
        iao_values.append(iao_pops)
        wbi_values.append(wbi)

        attrs = getattr(conf_group, "attrs", {})
        if "energy" in attrs:
            energies.append(float(attrs["energy"]))
        if "polarizability" in attrs:
            polar_values.append(attrs["polarizability"])
        elif "alpha" in attrs:
            polar_values.append(attrs["alpha"])
        else:
            polar_dataset = _collect_conf_value(conf_group, ("polarizability", "alpha"))
            if polar_dataset is not None:
                polar_values.append(polar_dataset)

    node_target = torch.stack(
        [
            torch.stack(chelpg_values, dim=0).mean(dim=0),
            torch.stack(iao_values, dim=0).mean(dim=0),
        ],
        dim=-1,
    )

    wbi_mean = torch.stack(wbi_values, dim=0).mean(dim=0)
    edge_values = []
    for src, dst in graph.edge_index.t().tolist():
        edge_values.append(wbi_mean[src, dst])
    edge_target = torch.stack(edge_values, dim=0).unsqueeze(-1)

    if not energies:
        energies = [0.0]
    if not polar_values:
        polar_values = [0.0]

    graph_target = torch.tensor(
        [sum(energies) / len(energies), sum(_scalarize(v) for v in polar_values) / len(polar_values)],
        dtype=torch.float32,
    )

    return node_target, edge_target, graph_target


def _resolve_results_candidate(results_path: Path | None, subset_id: int) -> Path | None:
    if results_path is None:
        return None
    if results_path.is_dir() or results_path.suffix != ".h5":
        return results_path / f"results_{subset_id:03d}.h5"
    return results_path


def _coerce_dataset_index(
    source: DatasetIndex | str | Path,
    geometry_path: str | Path | None = None,
    results_path: str | Path | None = None,
    limit: int = 16,
) -> DatasetIndex:
    if isinstance(source, DatasetIndex):
        return source

    csv_path = Path(source)
    geometry_path = Path(geometry_path) if geometry_path is not None else csv_path.with_name(
        f"coords_{_parse_subset_id(csv_path):03d}.pkl"
    )
    return build_dataset_index(
        csv_path,
        geometry_path=geometry_path,
        results_path=results_path,
        limit=limit,
    )


def load_quantum_zinc_dataset(
    source: DatasetIndex | str | Path,
    geometry_path: str | Path | None = None,
    results_path: str | Path | None = None,
    limit: int = 16,
    results_handle=None,
    use_results: bool = False,
):
    from .minimal import MinimalQuantumDataset, MinimalQuantumExample

    index = _coerce_dataset_index(
        source,
        geometry_path=geometry_path,
        results_path=results_path,
        limit=limit,
    )

    geometry_lookup = _load_pickle_mapping(index.geometry_path)
    csv_df = pd.read_csv(index.csv_path, usecols=lambda col: col == "smiles" or col in {"logP", "qed", "SAS"})
    df = csv_df.head(len(index.mol_ids)).reset_index(drop=True)
    aux_cols = [col for col in ("logP", "qed", "SAS") if col in df.columns]
    aux_available = len(aux_cols) == 3
    examples = []
    skipped_mol_ids: list[str] = []

    h5_handle = results_handle if use_results else None
    if h5_handle is None and use_results:
        candidate = _resolve_results_candidate(index.results_path, index.subset_id)
        if candidate is not None and candidate.exists():
            try:
                import h5py  # type: ignore
            except ImportError:
                h5_handle = None
            else:
                h5_handle = h5py.File(candidate, "r")

    try:
        for mol_id, row in zip(index.mol_ids, df.itertuples(index=False)):
            try:
                smiles = str(row.smiles).strip()
                graph = build_graph_from_smiles(smiles)
                geometry = geometry_lookup.get(mol_id, {})
                mol_group = _result_group(h5_handle, mol_id)

                if mol_group is None:
                    node_target, edge_target, graph_target = _build_proxy_targets(graph)
                    conformer_count = len(geometry.get("conformers", []))
                else:
                    try:
                        node_target, edge_target, graph_target = _aggregate_targets(graph, mol_group)
                        conformer_count = len(_conformer_groups(mol_group))
                    except Exception:
                        node_target, edge_target, graph_target = _build_proxy_targets(graph)
                        conformer_count = len(geometry.get("conformers", []))

                aux_target = None
                if aux_available:
                    aux_target = torch.tensor([float(row.logP), float(row.qed), float(row.SAS)], dtype=torch.float32)
                conformer_coords = _conformer_coords_from_geometry(geometry, graph.num_nodes)
                conformer_energies = None  # energies require HDF5 results; mean pooling is used otherwise
                examples.append(
                    MinimalQuantumExample(
                        mol_id=mol_id,
                        smiles=smiles,
                        graph=graph,
                        charge=int(geometry.get("charge", 0)),
                        conformer_count=conformer_count,
                        node_target=node_target,
                        edge_target=edge_target,
                        graph_target=graph_target,
                        aux_target=aux_target,
                        conformer_coords=conformer_coords,
                        conformer_energies=conformer_energies,
                    )
                )
            except Exception:
                skipped_mol_ids.append(mol_id)
    finally:
        if results_handle is None and hasattr(h5_handle, "close"):
            h5_handle.close()

    return MinimalQuantumDataset(examples=examples, skipped_mol_ids=tuple(skipped_mol_ids))


def load_quantum_zinc_subset_range(
    dataset_root: str | Path,
    subset_ids: list[int],
    limit_per_shard: int = 16,
    results_path: str | Path | None = None,
    use_results: bool = False,
):
    from .minimal import MinimalQuantumDataset

    dataset_root = Path(dataset_root)
    all_examples = []

    skipped_mol_ids: list[str] = []
    for index in iter_dataset_indices(
        dataset_root,
        subset_ids=subset_ids,
        limit_per_shard=limit_per_shard,
        use_results=use_results,
        results_root=results_path,
    ):
        shard_dataset = load_quantum_zinc_dataset(
            index,
            use_results=use_results,
        )
        all_examples.extend(shard_dataset.examples)
        skipped_mol_ids.extend(shard_dataset.skipped_mol_ids)

    return MinimalQuantumDataset(examples=all_examples, skipped_mol_ids=tuple(skipped_mol_ids))


def _expand_targets(dataset):
    node_targets = []
    edge_targets = []
    graph_targets = []

    for example in dataset.examples:
        node_targets.append(example.node_target.reshape(-1, example.node_target.shape[-1]))
        edge_targets.append(example.edge_target.reshape(-1, example.edge_target.shape[-1]))
        graph_targets.append(example.graph_target.reshape(1, -1))

    return torch.cat(node_targets, dim=0), torch.cat(edge_targets, dim=0), torch.cat(graph_targets, dim=0)


def compute_target_normalization(dataset):
    node_targets, edge_targets, graph_targets = _expand_targets(dataset)

    def _stats(values):
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False).clamp_min(1e-8)
        return mean, std

    node_mean, node_std = _stats(node_targets)
    edge_mean, edge_std = _stats(edge_targets)
    graph_mean, graph_std = _stats(graph_targets)

    return {
        'node_mean': node_mean,
        'node_std': node_std,
        'edge_mean': edge_mean,
        'edge_std': edge_std,
        'graph_mean': graph_mean,
        'graph_std': graph_std,
    }


def normalize_targets(node_target, edge_target, graph_target, stats):
    node = (node_target - stats['node_mean']) / stats['node_std']
    edge = (edge_target - stats['edge_mean']) / stats['edge_std']
    graph = (graph_target - stats['graph_mean'].squeeze(0)) / stats['graph_std'].squeeze(0)
    return node, edge, graph
