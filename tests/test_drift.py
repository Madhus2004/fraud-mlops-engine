import numpy as np
import pandas as pd

from core.features import DRIFT_COLUMNS
from training.drift import compute_drift, psi


def _frame(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(shift, 1, size=(n, len(DRIFT_COLUMNS))), columns=DRIFT_COLUMNS)


def test_no_drift_on_same_distribution():
    res = compute_drift(_frame(5000, 1), _frame(300, 2))
    assert not res["drift_detected"]


def test_detects_shifted_distribution():
    res = compute_drift(_frame(5000, 1), _frame(300, 2, shift=2.0))
    assert res["drift_detected"] and res["drift_share"] > 0.9


def test_insufficient_samples_is_safe():
    assert not compute_drift(_frame(1000, 1), _frame(5, 2))["drift_detected"]


def test_psi_zero_for_identical():
    x = np.random.default_rng(0).normal(size=2000)
    assert psi(x, x) < 1e-6
