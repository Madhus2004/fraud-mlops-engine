import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score


def find_optimal_threshold(y_true, y_probs, min_recall: float = 0.80):
    """Highest-precision threshold that keeps recall >= min_recall (F1-best fallback)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    p, r = precisions[:-1], recalls[:-1]
    ok = r >= min_recall
    if ok.any():
        idx = int(np.argmax(np.where(ok, p, -1.0)))
    else:
        f1 = 2 * p * r / (p + r + 1e-12)
        idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(p[idx]), float(r[idx])


def metrics_at_threshold(y_true, y_probs, threshold: float) -> dict:
    pred = (np.asarray(y_probs) >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_probs)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def should_promote(candidate_pr_auc: float, champion_pr_auc: float, min_gain: float = 0.0) -> bool:
    """Circuit breaker: the challenger must beat the champion on the untouched holdout."""
    return candidate_pr_auc > champion_pr_auc + min_gain
