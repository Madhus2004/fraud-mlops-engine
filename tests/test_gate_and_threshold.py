import numpy as np

from core.features import FEATURE_COLUMNS, apply_scaler, fit_scaler
from tests.conftest import make_frame
from training.evaluate import find_optimal_threshold, should_promote


def test_gate_promotes_only_when_better():
    assert should_promote(0.90, 0.85)
    assert not should_promote(0.85, 0.85)
    assert not should_promote(0.80, 0.85)
    assert not should_promote(0.851, 0.85, min_gain=0.01)


def test_threshold_respects_recall_floor():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    probs = np.clip(y * 0.5 + rng.normal(0.25, 0.2, 5000), 0, 1)
    _, precision, recall = find_optimal_threshold(y, probs, min_recall=0.8)
    assert recall >= 0.8 and 0 < precision <= 1


def test_scaler_roundtrip_and_column_order():
    df = make_frame(200)
    scaler = fit_scaler(df)
    out = apply_scaler(df.sample(frac=1, axis=1, random_state=1), scaler)  # shuffled columns in
    assert list(out.columns) == FEATURE_COLUMNS
    assert abs(out["Amount"].mean()) < 1e-3
