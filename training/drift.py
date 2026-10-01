"""Drift detection on RAW features (live logs vs reference): KS test + PSI."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from core.db import expand_features, fetch_logs
from core.features import DRIFT_COLUMNS
from training.config import REFERENCE_PATH


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(reference, edges)[0] / len(reference), 1e-4, None)
    c = np.clip(np.histogram(current, edges)[0] / len(current), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame, alpha: float = 0.05,
                  min_ks: float = 0.10, psi_threshold: float = 0.25,
                  share_threshold: float = 0.30, min_samples: int = 30) -> dict:
    """A feature drifts if (KS significant AND effect size >= min_ks) OR PSI > psi_threshold.
    The effect-size guard stops huge samples from flagging trivial differences."""
    if len(current) < min_samples:
        return {"drift_detected": False, "drift_share": 0.0, "sample_size": len(current),
                "features": [], "message": f"Need >= {min_samples} live samples."}
    rows = []
    for col in DRIFT_COLUMNS:
        if col not in reference or col not in current:
            continue
        ref, cur = reference[col].dropna().values, current[col].dropna().values
        stat, p = ks_2samp(ref, cur)
        score = psi(ref, cur)
        rows.append({"feature": col, "ks_stat": round(float(stat), 4), "p_value": float(p),
                     "psi": round(score, 4), "drifted": bool((p < alpha and stat >= min_ks) or score > psi_threshold)})
    share = sum(r["drifted"] for r in rows) / max(len(rows), 1)
    return {"drift_detected": share >= share_threshold, "drift_share": round(share, 4),
            "drifted_features": [r["feature"] for r in rows if r["drifted"]],
            "sample_size": len(current), "features": rows}


def run_drift_check(window_hours: float = 24, **kwargs) -> dict:
    reference = pd.read_parquet(REFERENCE_PATH)
    current = expand_features(fetch_logs(since_hours=window_hours))
    return compute_drift(reference, current, **kwargs)


if __name__ == "__main__":
    res = run_drift_check()
    print({k: v for k, v in res.items() if k != "features"})
