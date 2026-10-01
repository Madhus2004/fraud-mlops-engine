import threading

from core.artifacts import (artifact_root, hub_configured, load_artifact,
                            predict_proba, pull_from_hub)
from core.features import apply_scaler

import pandas as pd


class ModelStore:
    """Loads the champion once and hot-swaps it atomically on reload."""

    def __init__(self):
        self._lock = threading.Lock()
        self.booster = None
        self.manifest = None

    def load(self) -> dict:
        directory = artifact_root() / "active"
        if hub_configured():
            try:
                directory = pull_from_hub(artifact_root() / "cache")
            except Exception as exc:  # fall back to a bundled/local copy
                print(f"Hub pull failed ({exc}); trying local artifacts.")
        booster, manifest = load_artifact(directory)
        with self._lock:
            self.booster, self.manifest = booster, manifest
        return manifest

    @property
    def ready(self) -> bool:
        return self.booster is not None

    def score(self, features: dict) -> float:
        with self._lock:
            booster, manifest = self.booster, self.manifest
        X = apply_scaler(pd.DataFrame([features]), manifest["scaler"])
        return float(predict_proba(booster, X)[0])


store = ModelStore()
