import json

from qchem_gnn.validation import run_validation
from tests._validation_fixtures import make_tiny_quantum_dataset


def _tiny_esol_csv(tmp_path):
    csv = tmp_path / "esol_tiny.csv"
    rows = ["smiles,y"]
    for i, smi in enumerate(["CO", "CCO", "CCN", "CCC", "CCCC", "CCCO",
                             "CCCN", "CCCCO", "c1ccccc1", "CC", "CCCl", "CCBr"]):
        rows.append(f"{smi},{0.1 * i:.2f}")
    csv.write_text("\n".join(rows) + "\n")
    return csv


def test_run_validation_writes_report(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    out_dir = tmp_path / "validate"
    cfg = {
        "pretrain": {"hidden_dim": 16, "message_passing_steps": 2, "epochs": 2,
                     "learning_rate": 0.01, "batch_size": 4, "hidden_dim_3d": 16,
                     "message_passing_steps_3d": 2},
        "arms": {"baseline": {"teacher_weight": 0.0, "conformer_pool_mode": "mean"},
                 "quantum": {"teacher_weight": 1.0, "conformer_pool_mode": "energy"},
                 "quantum_vicreg": {"teacher_weight": 1.0, "conformer_pool_mode": "energy",
                                    "contrastive_loss": "vicreg"}},
        "comparisons": [
            {"name": "teacher_vs_baseline", "reference": "baseline", "treatment": "quantum"},
            {"name": "vicreg_vs_infonce", "reference": "quantum", "treatment": "quantum_vicreg"},
        ],
        "seeds": [0],
        "holdout": {"fraction": 0.25, "seed": 1234},
        "probes": [{"method": "mlp_head"}],
        "adapt": {"dataset": {"csv": str(_tiny_esol_csv(tmp_path)),
                              "smiles_col": "smiles", "targets": ["y"]},
                  "task": "regression",
                  "adapter": {"hidden_dims": [8], "dropout": 0.0},
                  "training": {"epochs": 2, "lr": 1.0e-3, "batch_size": 4,
                               "patience": 5, "seed": 42},
                  "split": {"test_frac": 0.25, "val_frac": 0.25, "seed": 42,
                            "stratify": False}},
        "outputs": {"dir": str(out_dir), "report": str(out_dir / "report")},
    }
    aggregate = run_validation(cfg, dataset=dataset, overwrite=True)

    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.md").exists()
    assert "verdicts" in aggregate
    saved = json.loads((out_dir / "report.json").read_text())
    assert "rows" in saved and "aggregate" in saved
    # one cell per arm at seed 0 -> three backbones written
    assert (out_dir / "baseline_s0.pt").exists()
    assert (out_dir / "quantum_s0.pt").exists()
    assert (out_dir / "quantum_vicreg_s0.pt").exists()
    assert len(aggregate["verdicts"]) == 2
    assert {v["name"] for v in aggregate["verdicts"]} == {"teacher_vs_baseline", "vicreg_vs_infonce"}
    report_text = (out_dir / "report.md").read_text()
    assert "vicreg_vs_infonce" in report_text
