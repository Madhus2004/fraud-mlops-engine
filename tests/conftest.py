import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from core.artifacts import save_artifact
from core.features import FEATURE_COLUMNS, apply_scaler, fit_scaler


def make_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df["Amount"] = np.abs(df["Amount"]) * 100
    df["Time"] = np.abs(df["Time"]) * 1000
    df["Class"] = (df["V1"] + df["V2"] > 1.5).astype(int)
    return df


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    monkeypatch.delenv("HF_REPO_ID", raising=False)

    df = make_frame()
    scaler = fit_scaler(df)
    model = XGBClassifier(n_estimators=10, max_depth=3, random_state=0)
    model.fit(apply_scaler(df, scaler), df["Class"])
    save_artifact(tmp_path / "artifacts" / "active", model, scaler, "v1", 0.5, {"holdout_pr_auc": 0.9}, FEATURE_COLUMNS)

    from fastapi.testclient import TestClient
    from core import db
    db.reset_engine()
    from serving.main import app
    with TestClient(app) as client:
        yield client
    db.reset_engine()
