"""Tabulate downstream MAE vs. pretraining data scale from validation reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def aggregate_scaling(report_paths: dict[int, str]) -> list[dict]:
    """Read per-scale validation reports into MAE-vs-scale rows.

    ``report_paths`` maps a scale (number of shards) to that run's report.json.
    ``extrinsic`` is nested as ``extrinsic[method][arm]["mae_mean"]``.
    """
    rows: list[dict] = []
    for scale in sorted(report_paths):
        payload = json.loads(Path(report_paths[scale]).read_text())
        extrinsic = payload["aggregate"]["extrinsic"]
        for method, arms in extrinsic.items():
            for arm, stats in arms.items():
                rows.append(
                    {
                        "scale": scale,
                        "method": method,
                        "arm": arm,
                        "mae_mean": float(stats["mae_mean"]),
                    }
                )
    return rows


def _print_table(rows: list[dict]) -> None:
    print(f"{'scale':>6} {'method':>10} {'arm':>18} {'mae_mean':>10}")
    for r in sorted(rows, key=lambda x: (x["method"], x["scale"])):
        print(f"{r['scale']:>6} {r['method']:>10} {r['arm']:>18} {r['mae_mean']:>10.4f}")


if __name__ == "__main__":
    # Usage: aggregate_scaling.py 10=runs/s10/report.json 50=runs/s50/report.json
    paths = {}
    for arg in sys.argv[1:]:
        scale_str, _, path = arg.partition("=")
        paths[int(scale_str)] = path
    _print_table(aggregate_scaling(paths))
