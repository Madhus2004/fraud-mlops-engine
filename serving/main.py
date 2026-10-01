import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from core.db import init_db, log_feedback, log_prediction
from core.features import FEATURE_COLUMNS
from serving.model_loader import store


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        store.load()
    except Exception as exc:
        print(f"WARNING: no model loaded at startup: {exc}")
    yield


app = FastAPI(title="Real-Time Fraud Scoring API", version="2.0.0", lifespan=lifespan)


class TransactionInput(BaseModel):
    Time: float
    Amount: float
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float


class FeedbackInput(BaseModel):
    txn_id: str
    actual_label: int = Field(..., ge=0, le=1)


@app.get("/health")
def health():
    if not store.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    m = store.manifest
    return {"status": "healthy", "active_model_version": m["version"],
            "decision_threshold": m["threshold"], "metrics": m.get("metrics", {}),
            "created_at": m.get("created_at")}


@app.post("/predict")
def predict(payload: TransactionInput):
    if not store.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    features = payload.model_dump()
    assert set(features) == set(FEATURE_COLUMNS)
    prob = store.score(features)
    threshold, version = store.manifest["threshold"], store.manifest["version"]
    flagged = prob >= threshold
    txn_id = str(uuid.uuid4())
    log_prediction(txn_id, features, prob, flagged, version)  # RAW features are logged
    return {"txn_id": txn_id, "risk_score": round(prob, 4), "is_flagged": flagged,
            "decision_threshold": threshold, "model_version": version}


@app.post("/feedback")
def feedback(payload: FeedbackInput):
    if not log_feedback(payload.txn_id, payload.actual_label):
        raise HTTPException(status_code=404, detail="Transaction ID not found.")
    return {"status": "success", "txn_id": payload.txn_id, "actual_label": payload.actual_label}


@app.post("/admin/reload")
def reload_model(x_admin_token: str | None = Header(default=None)):
    expected = os.getenv("ADMIN_TOKEN")
    if not expected or x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Invalid admin token.")
    m = store.load()
    return {"status": "reloaded", "active_model_version": m["version"]}
