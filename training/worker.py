"""Local MLOps loop: drift check -> trigger decision -> retrain -> gate -> publish -> hot reload.

  python -m training.worker --once            # evaluate triggers, retrain only if needed
  python -m training.worker --once --force    # retrain now
  python -m training.worker --loop --interval 3600
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from core.artifacts import artifact_root
from core.db import count_labeled, init_db
from training.drift import run_drift_check
from training.publish import publish
from training.retrain import run_retraining

STATE_FILE = Path("reports/worker_state.json")


def _load_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {"labels_at_last_retrain": 0}


def run_once(force: bool = False, min_new_labels: int = 50, drift_share: float = 0.30,
             window_hours: float = 24) -> dict:
    init_db()
    Path("reports").mkdir(exist_ok=True)
    state = _load_state()

    drift = run_drift_check(window_hours, share_threshold=drift_share)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    (Path("reports") / f"drift_{stamp}.json").write_text(json.dumps(drift, indent=2))
    print(f"Drift: share={drift['drift_share']:.0%} detected={drift['drift_detected']} samples={drift['sample_size']}")

    new_labels = count_labeled() - state["labels_at_last_retrain"]
    triggers = []
    if force:
        triggers.append("manual")
    if drift["drift_detected"]:
        triggers.append("drift")
    if new_labels >= min_new_labels:
        triggers.append(f"{new_labels} new labels")
    if not triggers:
        print(f"No retrain trigger (new labels: {new_labels}/{min_new_labels}).")
        return {"retrained": False, "drift": drift}

    print("Retrain triggers:", ", ".join(triggers))
    result = run_retraining()
    state["labels_at_last_retrain"] = count_labeled()
    STATE_FILE.write_text(json.dumps(state))
    if result["promoted"]:
        publish(artifact_root() / "versions" / result["candidate_version"], result["candidate_version"])
    return {"retrained": True, "triggers": triggers, "result": result, "drift": drift}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=3600, help="seconds between loop runs")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-new-labels", type=int, default=50)
    ap.add_argument("--window-hours", type=float, default=24)
    a = ap.parse_args()
    while True:
        run_once(a.force, a.min_new_labels, window_hours=a.window_hours)
        if not a.loop:
            break
        time.sleep(a.interval)
