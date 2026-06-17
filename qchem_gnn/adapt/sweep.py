# qchem_gnn/adapt/sweep.py
from __future__ import annotations

import copy
import csv
import itertools
from pathlib import Path

from .config import AdaptConfig, _deep_merge, resolve_adapt_config, to_raw_dict


def _nest(dotted: str, value):
    parts = dotted.split(".")
    out = cur = {}
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
    return out


def expand_grid(grid: dict[str, list]) -> list[dict]:
    """Return the Cartesian product of a sweep grid as a list of nested override dicts."""
    keys = list(grid)
    cells = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        override: dict = {}
        for k, v in zip(keys, combo):
            override = _deep_merge(override, _nest(k, v))
        cells.append(override)
    return cells


def _flatten_metrics(metrics: dict) -> dict:
    flat = {}
    for k in ("mae", "rmse", "r2", "auc", "accuracy", "f1"):
        if k in metrics:
            flat[f"test_{k}"] = round(metrics[k], 4)
    return flat


def _flat_items(d: dict, prefix: str = ""):
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from _flat_items(v, key)
        else:
            yield key, v


def _print_table(cols, rows):
    widths = {c: max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(str(c).rjust(widths[c]) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).rjust(widths[c]) for c in cols))


def run_sweep(cfg: AdaptConfig) -> list[dict]:
    """Run each grid cell and return a list of result rows; optionally write a CSV report."""
    from .runner import run  # local import avoids circular dependency

    grid = cfg.sweep["grid"]
    report_path = cfg.sweep.get("report")
    base_raw = to_raw_dict(cfg)
    cells = expand_grid(grid)

    rows: list[dict] = []
    for i, override in enumerate(cells):
        raw = _deep_merge(copy.deepcopy(base_raw), override)
        # Give each cell a unique adapter output path to avoid collisions
        if base_raw["outputs"].get("adapter"):
            stem = Path(base_raw["outputs"]["adapter"])
            raw = _deep_merge(raw, {"outputs": {"adapter": str(stem.with_name(f"{stem.stem}_cell{i}{stem.suffix}"))}})
        cell_cfg = resolve_adapt_config(raw)
        summary = run(cell_cfg)
        flat_over = dict(_flat_items(override))
        rows.append({**flat_over, **_flatten_metrics(summary["test_metrics"])})

    if rows:
        cols = list(rows[0].keys())
        if report_path:
            rp = Path(report_path)
            rp.parent.mkdir(parents=True, exist_ok=True)
            with rp.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
        _print_table(cols, rows)
    return rows
