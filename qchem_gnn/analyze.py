from __future__ import annotations

import json
import statistics
from pathlib import Path


def parse_metrics_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return records


def compute_summary(records: list[dict]) -> dict:
    if not records:
        return {}

    def _stat(key: str) -> dict | None:
        vals = [(r[key], r["step"]) for r in records if key in r]
        if not vals:
            return None
        min_val = min(v for v, _ in vals)
        min_step = next(s for v, s in vals if v == min_val)
        return {
            "initial": vals[0][0],
            "final": vals[-1][0],
            "min": min_val,
            "min_step": min_step,
        }

    terms = {
        "train_total":      _stat("train_loss"),
        "train_supervised": _stat("train_supervised"),
        "train_contrastive": _stat("train_contrastive"),
        "train_teacher":    _stat("train_teacher"),
        "val_total":        _stat("val_loss"),
    }

    # Convergence: last step where train_loss improved by >1% over the prior record.
    convergence_step: int | None = None
    train_pairs = [(r["step"], r["train_loss"]) for r in records if "train_loss" in r]
    for i in range(1, len(train_pairs)):
        prev_loss = train_pairs[i - 1][1]
        curr_loss = train_pairs[i][1]
        if prev_loss > 0 and (prev_loss - curr_loss) / prev_loss > 0.01:
            convergence_step = train_pairs[i][0]

    rates = [r["steps_per_sec"] for r in records if "steps_per_sec" in r]
    avg_rate = statistics.mean(rates) if rates else 0.0

    first, last = records[0], records[-1]
    return {
        "first_step": first.get("step"),
        "last_step": last.get("step"),
        "total_steps": last.get("total_steps"),
        "wall_seconds": last.get("wall_seconds", 0.0),
        "avg_steps_per_sec": avg_rate,
        "terms": terms,
        "convergence_step": convergence_step,
    }


def format_summary(summary: dict) -> str:
    if not summary:
        return "(no records)"

    step = summary["last_step"]
    total = summary["total_steps"]
    wall = int(summary["wall_seconds"])
    h, rem = divmod(wall, 3600)
    m, s = divmod(rem, 60)
    wall_str = f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"
    rate = summary["avg_steps_per_sec"]

    lines = [
        f"Steps: {step}/{total}  |  Wall: {wall_str}  |  Throughput: {rate:.1f} step/s",
        f"{'Loss':<22}{'initial':>10}{'final':>10}{'min (step)':>20}",
        "─" * 62,
    ]

    _ROWS = [
        ("train total",     "train_total"),
        ("  supervised",    "train_supervised"),
        ("  contrastive",   "train_contrastive"),
        ("  teacher",       "train_teacher"),
        ("val total",       "val_total"),
    ]
    for label, key in _ROWS:
        stat = summary["terms"].get(key)
        if stat is None:
            continue
        lines.append(
            f"{label:<22}{stat['initial']:>10.4f}{stat['final']:>10.4f}"
            f"  {stat['min']:.4f} (step {stat['min_step']})"
        )

    conv = summary["convergence_step"]
    if conv is not None:
        lines.append(f"Convergence: step {conv} (last >1% improvement in train loss)")
    else:
        lines.append("Convergence: not detected (loss may still be improving or plateaued early)")

    return "\n".join(lines)


def format_record(record: dict) -> str:
    step = record.get("step", "?")
    total = record.get("total_steps", "?")
    loss = record.get("train_loss", float("nan"))
    sup = record.get("train_supervised", float("nan"))
    con = record.get("train_contrastive", float("nan"))
    tea = record.get("train_teacher", float("nan"))
    rate = record.get("steps_per_sec", float("nan"))

    parts = [
        f"step {step:>6}/{total}",
        f"loss {loss:.2f} (sup {sup:.2f} / con {con:.2f} / tea {tea:.2f})",
        f"{rate:.1f} step/s",
    ]
    if "val_loss" in record:
        parts.insert(2, f"val {record['val_loss']:.2f}")

    return " | ".join(parts)
