import json

from scripts.aggregate_scaling import aggregate_scaling


def _write_report(path, extrinsic):
    path.write_text(json.dumps({"aggregate": {"extrinsic": extrinsic}}))


def test_aggregate_scaling_collects_mae_by_scale(tmp_path):
    r10 = tmp_path / "s10.json"
    r50 = tmp_path / "s50.json"
    # extrinsic is nested: method -> arm -> stats
    _write_report(r10, {"mlp_head": {"quantum_scaffold": {"mae_mean": 0.90}}})
    _write_report(r50, {"mlp_head": {"quantum_scaffold": {"mae_mean": 0.80}}})

    rows = aggregate_scaling({10: str(r10), 50: str(r50)})
    by_scale = {r["scale"]: r["mae_mean"] for r in rows if r["method"] == "mlp_head"}
    assert by_scale == {10: 0.90, 50: 0.80}
    assert all({"scale", "method", "arm", "mae_mean"} <= set(r) for r in rows)
    assert rows[0]["arm"] == "quantum_scaffold"
