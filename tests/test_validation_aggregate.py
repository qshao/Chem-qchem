import pytest

from qchem_gnn.validation import aggregate_results, render_report


def _ex(arm, seed, method, mae, r2):
    return {"arm": arm, "seed": seed, "method": method, "status": "ok", "mae": mae, "r2": r2}


def test_mean_std_and_helps_verdict():
    # baseline mlp_head MAE ~1.20, quantum ~1.00, tight spread -> delta beats combined std
    rows = [
        _ex("baseline", 0, "mlp_head", 1.20, 0.40),
        _ex("baseline", 1, "mlp_head", 1.22, 0.39),
        _ex("baseline", 2, "mlp_head", 1.18, 0.41),
        _ex("quantum", 0, "mlp_head", 1.00, 0.52),
        _ex("quantum", 1, "mlp_head", 1.01, 0.51),
        _ex("quantum", 2, "mlp_head", 0.99, 0.53),
    ]
    agg = aggregate_results(rows, [])
    base = agg["extrinsic"]["mlp_head"]["baseline"]
    assert abs(base["mae_mean"] - 1.20) < 1e-6
    assert base["n"] == 3
    assert agg["verdicts"][0]["result"] == "helps"
    assert agg["verdicts"][0]["delta"] > 0


def test_within_noise_verdict():
    rows = [
        _ex("baseline", 0, "mlp_head", 1.20, 0.40),
        _ex("baseline", 1, "mlp_head", 0.90, 0.55),  # wide spread
        _ex("baseline", 2, "mlp_head", 1.50, 0.20),
        _ex("quantum", 0, "mlp_head", 1.15, 0.42),
        _ex("quantum", 1, "mlp_head", 0.95, 0.50),
        _ex("quantum", 2, "mlp_head", 1.45, 0.25),
    ]
    agg = aggregate_results(rows, [])
    assert agg["verdicts"][0]["result"] == "within noise"


def test_insufficient_seeds_verdict():
    rows = [
        _ex("baseline", 0, "mlp_head", 1.2, 0.4),
        _ex("quantum", 0, "mlp_head", 1.0, 0.5),
    ]
    agg = aggregate_results(rows, [])
    assert agg["verdicts"][0]["result"] == "insufficient seeds"


def test_failed_rows_excluded_and_na_when_empty():
    rows = [
        {"arm": "baseline", "seed": 0, "method": "mlp_head",
         "status": "failed", "mae": None, "r2": None},
        _ex("quantum", 0, "mlp_head", 1.0, 0.5),
        _ex("quantum", 1, "mlp_head", 1.0, 0.5),
    ]
    agg = aggregate_results(rows, [])
    assert agg["extrinsic"]["mlp_head"]["baseline"]["n"] == 0
    assert agg["verdicts"][0]["result"] == "n/a"


def test_render_report_contains_sections():
    rows = [_ex("baseline", 0, "mlp_head", 1.2, 0.4), _ex("quantum", 0, "mlp_head", 1.0, 0.5)]
    intr = [{"arm": "quantum", "seed": 0, "status": "ok",
             "properties": {"chelpg": {"r": 0.7, "mae": 0.1},
                            "energy": {"r": 0.8, "mae": 0.2},
                            "iso_polarizability": {"r": 0.6, "mae": 0.3},
                            "wbi": {"r": 0.5, "mae": 0.4}}}]
    text = render_report(aggregate_results(rows, intr))
    assert "Extrinsic" in text
    assert "Intrinsic" in text
    assert "Verdict" in text


def test_multiple_comparisons_produce_multiple_verdicts():
    rows = [
        _ex("baseline", 0, "mlp_head", 1.30, 0.30),
        _ex("baseline", 1, "mlp_head", 1.32, 0.29),
        _ex("baseline", 2, "mlp_head", 1.28, 0.31),
        _ex("quantum", 0, "mlp_head", 1.00, 0.52),
        _ex("quantum", 1, "mlp_head", 1.01, 0.51),
        _ex("quantum", 2, "mlp_head", 0.99, 0.53),
        _ex("quantum_vicreg", 0, "mlp_head", 0.80, 0.62),
        _ex("quantum_vicreg", 1, "mlp_head", 0.81, 0.61),
        _ex("quantum_vicreg", 2, "mlp_head", 0.79, 0.63),
    ]
    comparisons = [
        {"name": "teacher_vs_baseline", "reference": "baseline", "treatment": "quantum"},
        {"name": "vicreg_vs_infonce", "reference": "quantum", "treatment": "quantum_vicreg"},
    ]
    agg = aggregate_results(rows, [], arms=["baseline", "quantum", "quantum_vicreg"], comparisons=comparisons)
    names = [v["name"] for v in agg["verdicts"]]
    assert names == ["teacher_vs_baseline", "vicreg_vs_infonce"]
    assert agg["verdicts"][0]["result"] == "helps"
    assert agg["verdicts"][1]["result"] == "helps"
    assert agg["verdicts"][1]["delta"] == pytest.approx(0.20, abs=1e-6)


def test_comparison_with_missing_arm_is_na():
    rows = [_ex("baseline", 0, "mlp_head", 1.2, 0.4), _ex("baseline", 1, "mlp_head", 1.2, 0.4)]
    comparisons = [{"name": "vicreg_vs_infonce", "reference": "quantum", "treatment": "quantum_vicreg"}]
    agg = aggregate_results(rows, [], arms=["baseline"], comparisons=comparisons)
    assert agg["verdicts"][0]["result"] == "n/a"


def test_default_comparison_is_single_teacher_vs_baseline():
    rows = [
        _ex("baseline", 0, "mlp_head", 1.20, 0.40),
        _ex("baseline", 1, "mlp_head", 1.22, 0.39),
        _ex("quantum", 0, "mlp_head", 1.00, 0.52),
        _ex("quantum", 1, "mlp_head", 1.01, 0.51),
    ]
    agg = aggregate_results(rows, [])
    assert len(agg["verdicts"]) == 1
    assert agg["verdicts"][0]["name"] == "teacher_vs_baseline"
