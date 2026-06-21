from __future__ import annotations
import json
import subprocess
import sys
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

def test_parse_file_not_found(tmp_path):
    result = parse_metrics_jsonl(tmp_path / "does_not_exist.jsonl")
    assert result == []

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
