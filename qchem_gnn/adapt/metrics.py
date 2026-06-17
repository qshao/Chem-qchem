from __future__ import annotations

import numpy as np


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt, yp = _as_2d(y_true), _as_2d(y_pred)
    per_target = []
    for t in range(yt.shape[1]):
        diff = yp[:, t] - yt[:, t]
        ss_res = float(np.sum(diff ** 2))
        ss_tot = float(np.sum((yt[:, t] - yt[:, t].mean()) ** 2))
        per_target.append({
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        })
    return {
        "per_target": per_target,
        "mae": float(np.mean([m["mae"] for m in per_target])),
        "rmse": float(np.mean([m["rmse"] for m in per_target])),
        "r2": float(np.mean([m["r2"] for m in per_target])),
    }


def _auc_binary(y: np.ndarray, p: np.ndarray) -> float:
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Mann-Whitney U statistic / (n_pos * n_neg)
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    auc = (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    yt, yp = _as_2d(y_true), _as_2d(y_prob)
    per_target = []
    for t in range(yt.shape[1]):
        y = yt[:, t].astype(int)
        p = yp[:, t]
        pred = (p >= 0.5).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_target.append({
            "auc": _auc_binary(y, p),
            "accuracy": float(np.mean(pred == y)),
            "f1": float(f1),
        })
    return {
        "per_target": per_target,
        "auc": float(np.mean([m["auc"] for m in per_target])),
        "accuracy": float(np.mean([m["accuracy"] for m in per_target])),
        "f1": float(np.mean([m["f1"] for m in per_target])),
    }
