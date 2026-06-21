# Training Analytics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `analyze` CLI subcommand that reads a `metrics.jsonl` produced by contrastive pretraining and prints either a post-run summary or a live-tail formatted log (one line per new record, append mode).

**Architecture:** A new pure-function module `qchem_gnn/analyze.py` (parse → compute → format) is wired into `cli.py` as an `analyze` subcommand with an optional `--follow` flag. The module has no side effects; the CLI layer owns printing and polling. The follow mode polls the file every 5 seconds and prints one formatted line per new record; it exits on `KeyboardInterrupt`.

**Tech Stack:** Python 3.10+, stdlib only (`json`, `time`, `pathlib`, `statistics`). No new dependencies.

## Global Constraints

- Python 3.10+ type hints only: `list[dict]`, `dict[str, object]`, `T | None` — no `Optional[T]`
- No new third-party dependencies
- `parse_metrics_jsonl` must silently skip malformed lines — the file may be partially written by a concurrent training run
- Follow mode exits cleanly on `KeyboardInterrupt` (return 0)
- All new public functions in `analyze.py` take `Path` objects, not strings

## Metrics record format (from `contrastive_pretrain.py`)

Every record written by `_append_metrics_line` contains exactly these keys:

```python
{
    "step": int,               # current training step
    "total_steps": int,        # target total steps
    "train_loss": float,       # total training loss (average over log interval)
    "train_supervised": float,
    "train_contrastive": float,
    "train_teacher": float,
    "steps_per_sec": float,    # steps/sec over completed portion of run
    "wall_seconds": float,     # elapsed wall time since training started
    # Optional — present only when val_dataset was provided:
    "val_loss": float,
    "val_supervised": float,
    "val_contrastive": float,
    "val_teacher": float,
}
```

---

### Task 1: `qchem_gnn/analyze.py` — parse, compute, format

**Files:**
- Create: `qchem_gnn/analyze.py`
- Test: `tests/test_analyze.py`

**Interfaces:**
- Produces:
  - `parse_metrics_jsonl(path: Path) -> list[dict]`
  - `compute_summary(records: list[dict]) -> dict`
  - `format_summary(summary: dict) -> str`
  - `format_record(record: dict) -> str`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_analyze.py
from __future__ import annotations
import json
from pathlib import Path
import pytest
from qchem_gnn.analyze import (
    parse_metrics_jsonl,
    compute_summary,
    format_summary,
    format_record,
)

def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

_BASE_RECORD = {
    "step": 10, "total_steps": 100,
    "train_loss": 2.0, "train_supervised": 0.8,
    "train_contrastive": 0.7, "train_teacher": 0.5,
    "steps_per_sec": 80.0, "wall_seconds": 0.125,
}

def test_parse_returns_all_valid_records(tmp_path):
    path = tmp_path / "m.jsonl"
    r1 = {**_BASE_RECORD, "step": 10}
    r2 = {**_BASE_RECORD, "step": 20, "train_loss": 1.8}
    _write_jsonl(path, [r1, r2])
    records = parse_metrics_jsonl(path)
    assert len(records) == 2
    assert records[0]["step"] == 10
    assert records[1]["step"] == 20

def test_parse_skips_malformed_lines(tmp_path):
    path = tmp_path / "m.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps(_BASE_RECORD) + "\n")
        f.write("not-json\n")
        f.write(json.dumps({**_BASE_RECORD, "step": 20}) + "\n")
    records = parse_metrics_jsonl(path)
    assert len(records) == 2

def test_parse_empty_file_returns_empty_list(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text("")
    assert parse_metrics_jsonl(path) == []

def test_compute_summary_finds_min_step(tmp_path):
    records = [
        {**_BASE_RECORD, "step": 10, "train_loss": 2.0, "wall_seconds": 0.1},
        {**_BASE_RECORD, "step": 20, "train_loss": 1.5, "wall_seconds": 0.2},
        {**_BASE_RECORD, "step": 30, "train_loss": 1.6, "wall_seconds": 0.3},
    ]
    summary = compute_summary(records)
    assert summary["terms"]["train_total"]["min"] == pytest.approx(1.5)
    assert summary["terms"]["train_total"]["min_step"] == 20
    assert summary["terms"]["train_total"]["initial"] == pytest.approx(2.0)
    assert summary["terms"]["train_total"]["final"] == pytest.approx(1.6)

def test_compute_summary_convergence_detection(tmp_path):
    # 10→20: 25% drop (>1%); 20→30: 6.7% drop (>1%); 30→40: 0.1% drop (<1%)
    records = [
        {**_BASE_RECORD, "step": 10, "train_loss": 2.00, "wall_seconds": 0.1},
        {**_BASE_RECORD, "step": 20, "train_loss": 1.50, "wall_seconds": 0.2},
        {**_BASE_RECORD, "step": 30, "train_loss": 1.40, "wall_seconds": 0.3},
        {**_BASE_RECORD, "step": 40, "train_loss": 1.399, "wall_seconds": 0.4},
    ]
    summary = compute_summary(records)
    assert summary["convergence_step"] == 30

def test_compute_summary_val_loss_tracked(tmp_path):
    records = [
        {**_BASE_RECORD, "step": 10, "val_loss": 2.5, "val_supervised": 1.0,
         "val_contrastive": 0.9, "val_teacher": 0.6, "wall_seconds": 0.1},
        {**_BASE_RECORD, "step": 20, "val_loss": 2.1, "val_supervised": 0.8,
         "val_contrastive": 0.8, "val_teacher": 0.5, "wall_seconds": 0.2},
    ]
    summary = compute_summary(records)
    assert summary["terms"]["val_total"]["min"] == pytest.approx(2.1)
    assert summary["terms"]["val_total"]["min_step"] == 20

def test_compute_summary_no_val_loss_gives_none(tmp_path):
    records = [{**_BASE_RECORD, "step": 10, "wall_seconds": 0.1}]
    summary = compute_summary(records)
    assert summary["terms"]["val_total"] is None

def test_format_summary_contains_key_fields():
    records = [
        {**_BASE_RECORD, "step": 10, "train_loss": 2.0, "wall_seconds": 60.0},
        {**_BASE_RECORD, "step": 20, "train_loss": 1.5, "wall_seconds": 120.0},
    ]
    output = format_summary(compute_summary(records))
    assert "Steps:" in output
    assert "Throughput:" in output
    assert "train total" in output
    assert "supervised" in output
    assert "contrastive" in output
    assert "teacher" in output

def test_format_summary_empty_returns_message():
    assert "(no records)" in format_summary({})

def test_format_record_without_val():
    record = {**_BASE_RECORD, "step": 50}
    line = format_record(record)
    assert "50/100" in line
    assert "step/s" in line
    assert "val" not in line

def test_format_record_with_val():
    record = {**_BASE_RECORD, "step": 50, "val_loss": 1.9,
              "val_supervised": 0.7, "val_contrastive": 0.6, "val_teacher": 0.4}
    line = format_record(record)
    assert "50/100" in line
    assert "val" in line
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_analyze.py -v
```
Expected: ImportError — `qchem_gnn.analyze` does not exist yet.

- [ ] **Step 3: Write `qchem_gnn/analyze.py`**

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_analyze.py -v
```
Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add qchem_gnn/analyze.py tests/test_analyze.py
git commit -m "feat(analyze): parse/compute/format module for metrics.jsonl"
```

---

### Task 2: CLI integration — `analyze` subcommand

**Files:**
- Modify: `qchem_gnn/cli.py` (add subparser at line ~148, add `run_analyze`, dispatch in `main`)
- Test: `tests/test_analyze.py` (append CLI tests)

**Interfaces:**
- Consumes: `parse_metrics_jsonl`, `compute_summary`, `format_summary`, `format_record` from `qchem_gnn.analyze`
- Produces: `python -m qchem_gnn analyze <path>` (summary) and `python -m qchem_gnn analyze --follow <path>` (live tail)

- [ ] **Step 1: Write failing CLI tests**

Append to `tests/test_analyze.py`:

```python
import subprocess
import sys

def test_cli_analyze_summary(tmp_path):
    path = tmp_path / "metrics.jsonl"
    records = [
        {**_BASE_RECORD, "step": 10, "train_loss": 2.0, "wall_seconds": 60.0},
        {**_BASE_RECORD, "step": 20, "train_loss": 1.5, "wall_seconds": 120.0},
    ]
    _write_jsonl(path, records)
    result = subprocess.run(
        [sys.executable, "-m", "qchem_gnn", "analyze", str(path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Steps:" in result.stdout
    assert "train total" in result.stdout

def test_cli_analyze_missing_file_returns_nonzero(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "qchem_gnn", "analyze", str(tmp_path / "missing.jsonl")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
python -m pytest tests/test_analyze.py::test_cli_analyze_summary tests/test_analyze.py::test_cli_analyze_missing_file_returns_nonzero -v
```
Expected: FAIL — `analyze` is not a recognized command yet.

- [ ] **Step 3: Add `analyze` subparser to `build_parser()` in `cli.py`**

In `qchem_gnn/cli.py`, find the line `return parser` at the end of `build_parser()` (around line 150). Insert the `analyze` subparser immediately before it:

```python
    analyze_cmd = subparsers.add_parser(
        "analyze", help="Summarize or tail a metrics.jsonl from contrastive pretraining"
    )
    analyze_cmd.add_argument("metrics_path", help="Path to metrics.jsonl file")
    analyze_cmd.add_argument(
        "--follow", "-f",
        action="store_true",
        default=False,
        help="Poll the file and print one line per new record (Ctrl-C to stop)",
    )

    return parser
```

- [ ] **Step 4: Add `run_analyze()` function to `cli.py`**

Add this function anywhere before `main()` (e.g., after `run_preprocess`):

```python
def run_analyze(args) -> int:
    import sys
    import time
    from .analyze import parse_metrics_jsonl, compute_summary, format_summary, format_record

    path = Path(args.metrics_path)

    if not args.follow:
        records = parse_metrics_jsonl(path)
        if not records:
            print(f"No records found in {path}", file=sys.stderr)
            return 1
        print(format_summary(compute_summary(records)))
        return 0

    # Follow mode: poll for new lines, print one formatted line per new record.
    offset = 0
    try:
        while True:
            if path.exists():
                with open(path) as f:
                    f.seek(offset)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            import json as _json
                            record = _json.loads(line)
                            print(format_record(record), flush=True)
                        except Exception:  # noqa: BLE001
                            pass
                    offset = f.tell()
            time.sleep(5)
    except KeyboardInterrupt:
        return 0
```

- [ ] **Step 5: Dispatch `analyze` in `main()`**

In `main()`, add the `analyze` dispatch immediately after the `preprocess` check (around line 841), before `_resolved_namespace_from_args`:

```python
    if raw_args.command == "analyze":
        return run_analyze(raw_args)
```

The full updated block in `main()` should read:

```python
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = parser.parse_args(argv)
    if raw_args.command == "adapt":
        return run_adapt(raw_args)
    if raw_args.command == "preprocess":
        return run_preprocess(raw_args)
    if raw_args.command == "analyze":
        return run_analyze(raw_args)
    args = _resolved_namespace_from_args(raw_args)
    ...
```

- [ ] **Step 6: Run all tests**

```bash
python -m pytest tests/test_analyze.py -v
```
Expected: all 13 tests PASS.

Then run the full suite to confirm no regressions:

```bash
python -m pytest tests/ -q
```
Expected: 234 passed (221 existing + 13 new), 1 warning.

- [ ] **Step 7: Smoke-test the CLI manually**

Generate a small metrics file and verify both modes work:

```bash
# Generate a sample metrics.jsonl via a quick training run
# (log_every defaults to 100; use --total-steps 100 to get at least one log line)
python -m qchem_gnn contrastive-pretrain \
  --dataset-root zinc-250k/ --subset-ids 0 --limit 50 \
  --total-steps 100 \
  --output /tmp/test_ckpt.pt

# Find the generated metrics file (same path as output with .metrics.jsonl suffix)
# Then test summary mode:
python -m qchem_gnn analyze /tmp/test_ckpt.metrics.jsonl

# Expected output (example):
# Steps: 20/20  |  Wall: 0m 01s  |  Throughput: 83.7 step/s
# Loss                    initial     final       min (step)
# ──────────────────────────────────────────────────────────────
# train total              2.9100    1.2400   1.2100 (step 20)
#   supervised             1.1200    0.4400   0.4100 (step 20)
#   contrastive            1.0300    0.5100   0.4900 (step 15)
#   teacher                0.7600    0.2900   0.2700 (step 20)
# Convergence: step 15 (last >1% improvement in train loss)
```

- [ ] **Step 8: Commit**

```bash
git add qchem_gnn/cli.py tests/test_analyze.py
git commit -m "feat(cli): analyze subcommand — summary and follow modes for metrics.jsonl"
```
