from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from .eval import scaffold_key_from_smiles
from .minimal import MinimalQuantumDataset
from .quantum_data import load_quantum_zinc_subset_range

SHARD_CACHE_VERSION = 1


def _cache_path(cache_dir: Path, subset_id: int) -> Path:
    return Path(cache_dir) / f"shard_{subset_id:03d}.pt"


def _is_valid_cache(path: Path) -> bool:
    try:
        payload = torch.load(path, weights_only=False)
    except Exception:
        return False
    return (
        isinstance(payload, dict)
        and payload.get("version") == SHARD_CACHE_VERSION
        and "examples" in payload
        and "skipped_mol_ids" in payload
    )


def preprocess_shard(
    dataset_root, subset_id: int, cache_dir, *, overwrite: bool = False
) -> Path:
    """Extract one shard into a compact, scaffold-keyed cache file.

    Loads the shard through the existing extractor (which already ignores the
    density matrix), attaches a globally stable ``scaffold_key`` to each example,
    and serializes a versioned payload. Skips work if a valid cache exists.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = _cache_path(cache_dir, subset_id)
    if out.exists() and not overwrite and _is_valid_cache(out):
        return out

    dataset = load_quantum_zinc_subset_range(
        dataset_root,
        subset_ids=[subset_id],
        limit_per_shard=None,
        use_results=True,
    )
    examples = [
        dataclasses.replace(ex, scaffold_key=scaffold_key_from_smiles(ex.smiles))
        for ex in dataset.examples
    ]
    torch.save(
        {
            "version": SHARD_CACHE_VERSION,
            "examples": examples,
            "skipped_mol_ids": tuple(dataset.skipped_mol_ids),
        },
        out,
    )
    return out


def load_compact_shard(path):
    """Load one compact shard cache, validating its format version."""
    payload = torch.load(Path(path), weights_only=False)
    if not isinstance(payload, dict) or payload.get("version") != SHARD_CACHE_VERSION:
        got = payload.get("version") if isinstance(payload, dict) else None
        raise ValueError(
            f"shard cache version mismatch at {path}: expected "
            f"{SHARD_CACHE_VERSION}, got {got}"
        )
    return payload["examples"], tuple(payload["skipped_mol_ids"])


def load_compact_shards(cache_dir, shard_ids) -> MinimalQuantumDataset:
    """Preload a set of compact shard caches into one in-memory dataset."""
    examples = []
    skipped: list[str] = []
    for subset_id in shard_ids:
        exs, sk = load_compact_shard(_cache_path(Path(cache_dir), subset_id))
        examples.extend(exs)
        skipped.extend(sk)
    return MinimalQuantumDataset(examples=examples, skipped_mol_ids=tuple(skipped))
