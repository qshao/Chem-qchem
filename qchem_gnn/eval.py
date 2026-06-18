from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch import nn

from .checkpoint import load_checkpoint
from .graph import GraphBatch
from .model import MolecularQuantumGNN
from .splits import scaffold_or_random_split


def _as_2d_array(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array[:, None]
    return array


def _split_indices(split: dict[str, Iterable[int]], key: str) -> list[int]:
    return [int(index) for index in split.get(key, [])]


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0.0}

    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    residual_sum = float(np.sum(diff**2))
    centered = y_true - y_true.mean(axis=0, keepdims=True)
    total_sum = float(np.sum(centered**2))
    r2 = 0.0 if total_sum == 0.0 else float(1.0 - residual_sum / total_sum)

    return {"mae": mae, "rmse": rmse, "r2": r2, "n": float(y_true.shape[0])}


def _fit_ridge_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    ridge: float = 1e-4,
) -> np.ndarray:
    x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=x_train.dtype)], axis=1)
    identity = np.eye(x_aug.shape[1], dtype=x_aug.dtype)
    identity[-1, -1] = 0.0
    system = x_aug.T @ x_aug + ridge * identity
    target = x_aug.T @ y_train
    try:
        weights = np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(system) @ target
    return weights


def _predict_ridge(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ weights


def _score_regression(
    features: np.ndarray,
    labels: np.ndarray,
    split: dict[str, Iterable[int]],
    ridge: float = 1e-4,
) -> dict[str, object]:
    features = np.asarray(features, dtype=np.float32)
    labels = _as_2d_array(labels)

    train_idx = _split_indices(split, "train")
    if not train_idx:
        raise ValueError("Training split is empty")

    weights = _fit_ridge_regression(features[train_idx], labels[train_idx], ridge=ridge)
    predictions = _predict_ridge(features, weights)

    scored = {
        "train": _regression_metrics(labels[train_idx], predictions[train_idx]),
        "val": _regression_metrics(labels[_split_indices(split, "val")], predictions[_split_indices(split, "val")]),
        "test": _regression_metrics(labels[_split_indices(split, "test")], predictions[_split_indices(split, "test")]),
        "all": _regression_metrics(labels, predictions),
    }
    return {
        "mae": scored["all"]["mae"],
        "rmse": scored["all"]["rmse"],
        "r2": scored["all"]["r2"],
        "train": scored["train"],
        "val": scored["val"],
        "test": scored["test"],
        "all": scored["all"],
    }


def run_linear_probe(
    embeddings,
    labels,
    split: dict[str, Iterable[int]],
    ridge: float = 1e-4,
) -> dict[str, object]:
    return _score_regression(np.asarray(embeddings, dtype=np.float32), np.asarray(labels, dtype=np.float32), split, ridge=ridge)


def run_sample_efficiency(
    embeddings,
    labels,
    split: dict[str, Iterable[int]],
    fractions: Iterable[float] = (0.01, 0.05, 0.1, 1.0),
    seed: int = 0,
    ridge: float = 1e-4,
) -> dict[float, dict[str, object]]:
    rng = np.random.default_rng(seed)
    train_idx = list(_split_indices(split, "train"))
    if not train_idx:
        raise ValueError("Training split is empty")

    shuffled = np.array(train_idx, dtype=int)
    rng.shuffle(shuffled)

    results: dict[float, dict[str, object]] = {}
    for fraction in fractions:
        fraction = float(fraction)
        k = max(1, int(round(len(shuffled) * fraction)))
        k = min(k, len(shuffled))
        subset_split = {name: list(indices) for name, indices in split.items()}
        subset_split["train"] = shuffled[:k].tolist()
        results[fraction] = run_linear_probe(embeddings, labels, subset_split, ridge=ridge)
    return results


def _morgan_features(smiles: Iterable[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    smiles = list(smiles)
    features = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for index, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smile}")
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fingerprint, features[index])
    return features


def run_morgan_baseline(
    smiles: Iterable[str],
    labels,
    split: dict[str, Iterable[int]],
    radius: int = 2,
    n_bits: int = 2048,
    ridge: float = 1e-4,
) -> dict[str, object]:
    smiles = list(smiles)
    features = _morgan_features(smiles, radius=radius, n_bits=n_bits)
    return _score_regression(features, np.asarray(labels, dtype=np.float32), split, ridge=ridge)


def _infer_scaffolds(molecule_ids: Iterable[str], smiles: Iterable[str]) -> dict[str, str]:
    scaffolds: dict[str, str] = {}
    for molecule_id, smile in zip(molecule_ids, smiles):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            scaffolds[molecule_id] = smile
            continue
        scaffolds[molecule_id] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or smile
    return scaffolds


def scaffold_key_from_smiles(smiles: str) -> int:
    """Globally stable integer key for a molecule's Murcko scaffold.

    Two molecules sharing a Murcko scaffold get the same key, deterministically
    across processes and shards (unlike Python's per-process-salted ``hash()``).
    Unparseable SMILES fall back to the raw SMILES, so they only collide with an
    identical SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        scaffold = smiles
    else:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or smiles
    digest = hashlib.blake2b(scaffold.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def scaffold_mask_from_keys(keys: Sequence[int]) -> torch.Tensor:
    """Return [B, B] CPU bool mask; True at [i,j] iff i!=j and keys[i]==keys[j]."""
    t = torch.tensor(list(keys), dtype=torch.int64)
    eq = t[:, None] == t[None, :]
    eq.fill_diagonal_(False)
    return eq


def _dataset_targets(dataset) -> np.ndarray:
    targets = [example.graph_target.detach().cpu().numpy() for example in dataset.examples]
    return np.stack(targets, axis=0)


def _dataset_smiles(dataset) -> list[str]:
    return [example.smiles for example in dataset.examples]


def _dataset_mol_ids(dataset) -> list[str]:
    return [example.mol_id for example in dataset.examples]


def _split_for_dataset(dataset) -> dict[str, list[int]]:
    smiles = _dataset_smiles(dataset)
    mol_ids = _dataset_mol_ids(dataset)
    scaffolds = _infer_scaffolds(mol_ids, smiles)
    return scaffold_or_random_split(mol_ids, scaffolds=scaffolds, seed=0)


def run_fine_tuning(
    dataset,
    checkpoint_path: str | Path,
    epochs: int = 50,
    learning_rate: float = 1e-2,
    seed: int = 0,
) -> dict[str, object]:
    torch.manual_seed(seed)
    checkpoint = load_checkpoint(Path(checkpoint_path))
    model = MolecularQuantumGNN(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])

    batch = GraphBatch.from_graphs([example.graph for example in dataset.examples])
    labels = _dataset_targets(dataset)
    split = _split_for_dataset(dataset)

    model.train()
    with torch.no_grad():
        initial_embeddings = model.encode_graph_embeddings(batch).detach()
    target_dim = labels.shape[1]
    head = nn.Linear(initial_embeddings.shape[-1], target_dim)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=learning_rate)
    criterion = nn.MSELoss()

    train_idx = torch.tensor(split["train"], dtype=torch.long)
    if train_idx.numel() == 0:
        train_idx = torch.arange(labels.shape[0], dtype=torch.long)

    label_tensor = torch.as_tensor(labels, dtype=torch.float32)
    train_mean = label_tensor[train_idx].mean(dim=0, keepdim=True)
    train_std = label_tensor[train_idx].std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    normalized_labels = (label_tensor - train_mean) / train_std

    for _ in range(epochs):
        optimizer.zero_grad()
        embeddings = model.encode_graph_embeddings(batch)
        predictions = head(embeddings)
        loss = criterion(predictions[train_idx], normalized_labels[train_idx])
        loss.backward()
        optimizer.step()

    model.eval()
    head.eval()
    with torch.no_grad():
        embeddings = model.encode_graph_embeddings(batch)
        predictions = head(embeddings) * train_std + train_mean

    return {
        **_score_predictions(predictions.detach().cpu().numpy(), labels, split),
        "split": split,
    }


def _score_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    split: dict[str, Iterable[int]],
) -> dict[str, object]:
    labels = _as_2d_array(labels)
    predictions = _as_2d_array(predictions)
    scored = {
        "train": _regression_metrics(labels[_split_indices(split, "train")], predictions[_split_indices(split, "train")]),
        "val": _regression_metrics(labels[_split_indices(split, "val")], predictions[_split_indices(split, "val")]),
        "test": _regression_metrics(labels[_split_indices(split, "test")], predictions[_split_indices(split, "test")]),
        "all": _regression_metrics(labels, predictions),
    }
    return {
        "mae": scored["all"]["mae"],
        "rmse": scored["all"]["rmse"],
        "r2": scored["all"]["r2"],
        "train": scored["train"],
        "val": scored["val"],
        "test": scored["test"],
        "all": scored["all"],
    }
