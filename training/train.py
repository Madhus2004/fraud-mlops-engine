"""Train the initial champion (v1). Threshold tuned on validation, honest metrics on holdout."""
import pandas as pd

from core.artifacts import artifact_root, promote_to_active, save_artifact
from core.features import FEATURE_COLUMNS, apply_scaler, fit_scaler
from training.config import HOLDOUT_PATH, REFERENCE_PATH, VAL_PATH, build_model
from training.evaluate import find_optimal_threshold, metrics_at_threshold
from training.tracking import log_run


def train_baseline(version: str = "v1"):
    ref, val, hold = (pd.read_parquet(p) for p in (REFERENCE_PATH, VAL_PATH, HOLDOUT_PATH))
    scaler = fit_scaler(ref)  # fitted on train only, stored in the manifest

    y = ref["Class"]
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    model = build_model(spw)
    model.fit(apply_scaler(ref, scaler), y)

    threshold, _, _ = find_optimal_threshold(val["Class"], model.predict_proba(apply_scaler(val, scaler))[:, 1])
    hold_m = metrics_at_threshold(hold["Class"], model.predict_proba(apply_scaler(hold, scaler))[:, 1], threshold)
    metrics = {"holdout_pr_auc": round(hold_m["pr_auc"], 4), "holdout_precision": round(hold_m["precision"], 4),
               "holdout_recall": round(hold_m["recall"], 4), "train_samples": int(len(ref))}

    out = artifact_root() / "versions" / version
    save_artifact(out, model, scaler, version, threshold, metrics, FEATURE_COLUMNS)
    promote_to_active(out)
    log_run(f"train-{version}", {"scale_pos_weight": spw, "threshold": threshold}, metrics, [out / "manifest.json"])
    print(f"Saved {version} -> {out}\nThreshold {threshold:.4f}\nHoldout metrics: {metrics}")


if __name__ == "__main__":
    train_baseline()
