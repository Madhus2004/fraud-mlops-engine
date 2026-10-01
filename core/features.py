"""Single source of truth for feature order and scaling.

Rule: the DB, reference data and drift checks always hold RAW features.
Scaling is applied only right before the model sees the data, using the
scaler parameters stored inside the model artifact's manifest.
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
SCALE_COLUMNS = ["Time", "Amount"]
DRIFT_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]  # Time is non-stationary by nature


def fit_scaler(df: pd.DataFrame) -> dict:
    """Fit a standard scaler on raw data; returned as plain JSON-serialisable params."""
    return {
        "columns": SCALE_COLUMNS,
        "mean": [float(df[c].mean()) for c in SCALE_COLUMNS],
        "scale": [float(df[c].std(ddof=0)) or 1.0 for c in SCALE_COLUMNS],
    }


def apply_scaler(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    """Return model-ready features: correct column order, scaled Time/Amount, float32."""
    out = df[FEATURE_COLUMNS].astype("float64").copy()
    for col, mean, scale in zip(scaler["columns"], scaler["mean"], scaler["scale"]):
        out[col] = (out[col] - mean) / (scale or 1.0)
    return out.astype("float32")
