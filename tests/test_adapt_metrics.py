from __future__ import annotations

import numpy as np

from qchem_gnn.adapt.metrics import classification_metrics, regression_metrics


def test_regression_metrics_perfect():
    y = np.array([[1.0], [2.0], [3.0]])
    m = regression_metrics(y, y.copy())
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0
    assert len(m["per_target"]) == 1


def test_regression_metrics_multitarget():
    y = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    pred = y + 1.0
    m = regression_metrics(y, pred)
    assert len(m["per_target"]) == 2
    assert abs(m["mae"] - 1.0) < 1e-6


def test_classification_metrics_perfect():
    y = np.array([[0], [1], [1], [0]])
    prob = np.array([[0.1], [0.9], [0.8], [0.2]])
    m = classification_metrics(y, prob)
    assert abs(m["accuracy"] - 1.0) < 1e-6
    assert abs(m["auc"] - 1.0) < 1e-6
