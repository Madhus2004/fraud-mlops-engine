"""Model artifacts: local versioned store + Hugging Face Hub as the shared registry.

Layout:
  artifacts/versions/vN/{model.json, manifest.json}   every trained/promoted version
  artifacts/active/{model.json, manifest.json}        current champion (local copy)
  artifacts/cache/                                    what the API pulled from the Hub
"""
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb

MODEL_FILE = "model.json"
MANIFEST_FILE = "manifest.json"


def artifact_root() -> Path:
    return Path(os.getenv("ARTIFACT_ROOT", "artifacts"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_artifact(out_dir, model, scaler: dict, version: str, threshold: float,
                  metrics: dict, feature_columns: list) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(out / MODEL_FILE))
    manifest = {
        "version": version,
        "threshold": float(threshold),
        "feature_columns": list(feature_columns),
        "scaler": scaler,
        "metrics": metrics,
        "model_sha256": _sha256(out / MODEL_FILE),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return manifest


def load_artifact(directory):
    """Returns (Booster, manifest). Booster needs no scikit-learn, keeping serving light."""
    d = Path(directory)
    manifest = json.loads((d / MANIFEST_FILE).read_text())
    booster = xgb.Booster()
    booster.load_model(str(d / MODEL_FILE))
    return booster, manifest


def predict_proba(booster, X) -> np.ndarray:
    return booster.predict(xgb.DMatrix(X))


def next_version() -> str:
    vdir = artifact_root() / "versions"
    nums = [int(m.group(1)) for p in vdir.glob("v*") if (m := re.fullmatch(r"v(\d+)", p.name))] if vdir.exists() else []
    return f"v{max(nums, default=0) + 1}"


def promote_to_active(version_dir) -> Path:
    active = artifact_root() / "active"
    if active.exists():
        shutil.rmtree(active)
    shutil.copytree(version_dir, active)
    return active


# ---- Hugging Face Hub registry -------------------------------------------------
def hub_configured() -> bool:
    return bool(os.getenv("HF_REPO_ID"))


def pull_from_hub(dest) -> Path:
    from huggingface_hub import hf_hub_download
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    repo, token = os.environ["HF_REPO_ID"], os.getenv("HF_TOKEN")
    for name in (MANIFEST_FILE, MODEL_FILE):
        cached = hf_hub_download(repo_id=repo, filename=f"active/{name}", token=token, force_download=True)
        shutil.copyfile(cached, dest / name)
    return dest


def push_to_hub(version_dir, version: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    repo = os.environ["HF_REPO_ID"]
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(version_dir), repo_id=repo, repo_type="model",
                      path_in_repo=f"versions/{version}", commit_message=f"Add {version}")
    api.upload_folder(folder_path=str(version_dir), repo_id=repo, repo_type="model",
                      path_in_repo="active", commit_message=f"Promote {version} to active")
