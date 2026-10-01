"""Challenger training + champion/challenger gate. Heavy work: runs locally, never on Render."""
import pandas as pd

from core.artifacts import (artifact_root, load_artifact, next_version, predict_proba,
                            promote_to_active, save_artifact)
from core.db import expand_features, fetch_logs
from core.features import FEATURE_COLUMNS, apply_scaler
from training.config import HOLDOUT_PATH, REFERENCE_PATH, VAL_PATH, build_model
from training.evaluate import find_optimal_threshold, metrics_at_threshold, should_promote
from training.tracking import log_run


def build_training_frame() -> tuple[pd.DataFrame, int]:
    """Reference data + labeled live logs, all RAW so they are consistent by construction."""
    ref = pd.read_parquet(REFERENCE_PATH)[FEATURE_COLUMNS + ["Class"]]
    logs = fetch_logs(labeled_only=True)
    if logs.empty:
        return ref, 0
    live = expand_features(logs)[FEATURE_COLUMNS].copy()
    live["Class"] = logs["actual_label"].astype(int).values
    return pd.concat([ref, live], ignore_index=True), len(live)


def run_retraining(min_gain: float = 0.0) -> dict:
    champion, champ_manifest = load_artifact(artifact_root() / "active")
    scaler = champ_manifest["scaler"]  # keep the champion's scaler so features stay consistent

    train_df, n_live = build_training_frame()
    y = train_df["Class"]
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    candidate = build_model(spw)
    candidate.fit(apply_scaler(train_df, scaler), y)

    val, hold = pd.read_parquet(VAL_PATH), pd.read_parquet(HOLDOUT_PATH)
    Xv, Xh = apply_scaler(val, scaler), apply_scaler(hold, scaler)

    threshold, _, _ = find_optimal_threshold(val["Class"], candidate.predict_proba(Xv)[:, 1])  # tuned on validation
    cand_m = metrics_at_threshold(hold["Class"], candidate.predict_proba(Xh)[:, 1], threshold)  # gate on holdout
    champ_pr_auc = metrics_at_threshold(hold["Class"], predict_proba(champion, Xh), champ_manifest["threshold"])["pr_auc"]

    promoted = should_promote(cand_m["pr_auc"], champ_pr_auc, min_gain)
    result = {"promoted": promoted, "champion_version": champ_manifest["version"],
              "champion_pr_auc": round(champ_pr_auc, 4), "candidate_pr_auc": round(cand_m["pr_auc"], 4),
              "labeled_live_samples": n_live, "threshold": round(threshold, 4)}

    version = next_version()
    result["candidate_version"] = version
    metrics = {"holdout_pr_auc": round(cand_m["pr_auc"], 4), "holdout_precision": round(cand_m["precision"], 4),
               "holdout_recall": round(cand_m["recall"], 4), "train_samples": int(len(train_df)),
               "labeled_live_samples": n_live, "champion_pr_auc": round(champ_pr_auc, 4)}
    out = artifact_root() / "versions" / version
    save_artifact(out, candidate, scaler, version, threshold, metrics, FEATURE_COLUMNS)  # every candidate is archived
    if promoted:
        promote_to_active(out)
        result["artifact_dir"] = str(out)
    log_run(f"retrain-{version}", {"scale_pos_weight": spw, "promoted": promoted, "min_gain": min_gain},
            {**metrics, "gate_passed": float(promoted)}, [out / "manifest.json"])
    print("GATE:", "PROMOTED ✅" if promoted else "REJECTED ❌ (champion kept)", result)
    return result


if __name__ == "__main__":
    run_retraining()
