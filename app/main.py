import os
import json
import uuid
import sqlite3
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.db import init_db, log_predict, log_feedback
from src.retrain import execute_retraining_pipeline

import gc
from fastapi import FastAPI, HTTPException, status


# Paths
V1_MODEL_PATH = "app/models/xgboost_v1.pkl"
ACTIVE_MODEL_PATH = "app/models/xgboost_active.pkl"
SCALER_PATH = "app/models/scaler.pkl"
ACTIVE_METRICS_PATH = "app/models/metrics_active.json"
V1_METRICS_PATH = "app/models/metrics_v1.json"
DB_PATH = os.getenv("DB_PATH", "fraud_logs.db")

app = FastAPI(
    title="Real-Time Credit Card Fraud Engine",
    version="1.0.0",
    description="Microservice with real-time risk scoring, SQLite logging, delayed label feedback, and model retraining."
)

# Initialize database table on startup
@app.on_event("startup")
def startup_event():
    init_db()

# Pydantic Schemas
class TransactionInput(BaseModel):
    Amount: float
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float

class FeedbackInput(BaseModel):
    txn_id: str
    actual_label: int = Field(..., ge=0, le=1)

def load_active_artifacts():
    """Loads active model, scaler, and decision threshold."""
    model_path = ACTIVE_MODEL_PATH if os.path.exists(ACTIVE_MODEL_PATH) else V1_MODEL_PATH
    metrics_path = ACTIVE_METRICS_PATH if os.path.exists(ACTIVE_METRICS_PATH) else V1_METRICS_PATH
    
    if not os.path.exists(model_path) or not os.path.exists(SCALER_PATH):
        raise RuntimeError("Model or scaler artifacts not found. Run training script first.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(SCALER_PATH)
    
    threshold = 0.5
    model_ver = "v1"
    
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            threshold = metrics.get("optimal_threshold", 0.5)
            model_ver = metrics.get("model_version", "v1")
            
    return model, scaler, threshold, model_ver


# ==========================================
# 1. HEALTH CHECK ENDPOINT
# ==========================================
@app.get("/health")
def health_check():
    _, _, threshold, model_ver = load_active_artifacts()
    return {
        "status": "healthy",
        "active_model_version": model_ver,
        "decision_threshold": threshold
    }


# ==========================================
# 2. REAL-TIME PREDICTION ENDPOINT
# ==========================================
@app.post("/predict")
def predict_fraud(payload: TransactionInput):
    txn_id = str(uuid.uuid4())

    # Pydantic v2 compatible dict conversion
    data_dict = (
        payload.model_dump()
        if hasattr(payload, "model_dump")
        else payload.dict()
    )

    model, scaler, threshold, model_ver = load_active_artifacts()

    # Convert input dictionary to DataFrame
    df_raw = pd.DataFrame([data_dict])

    # Scale Amount and Time features
    df_scaled = df_raw.copy()
    df_scaled[["Amount", "Time"]] = scaler.transform(df_raw[["Amount", "Time"]])

    # Reorder DataFrame columns to EXACTLY match XGBoost training feature order
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    else:
        expected_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    df_features = df_scaled[expected_cols]

    # Predict probability
    prob = float(model.predict_proba(df_features)[0, 1])
    is_flagged = prob >= threshold

    # Log prediction to SQLite database
    log_predict(txn_id, data_dict, prob, is_flagged)

    return {
        "txn_id": txn_id,
        "risk_score": round(prob, 4),
        "is_flagged": is_flagged,
        "decision_threshold": threshold,
        "model_version": model_ver,
    }


# ==========================================
# 3. GROUND-TRUTH FEEDBACK ENDPOINT
# ==========================================
@app.post("/feedback")
def receive_feedback(payload: FeedbackInput):
    success = log_feedback(payload.txn_id, payload.actual_label)
    if not success:
        raise HTTPException(status_code=404, detail="Transaction ID not found in logs.")
        
    return {
        "status": "success",
        "message": f"Ground-truth label {payload.actual_label} attached to txn_id {payload.txn_id}"
    }


# ==========================================
# EXPORT LOGS ENDPOINT (For Local Worker)
# ==========================================
# Add this endpoint directly in app/main.py
@app.get("/export-logs")
def export_inference_logs():
    """Exports raw SQLite logs as JSON for the local retraining worker."""
    if not os.path.exists(DB_PATH):
        return []
    
    try:
        conn = sqlite3.connect(DB_PATH)
        df_logs = pd.read_sql_query("SELECT * FROM inference_logs", conn)
        conn.close()
        
        # Return pure JSON list of dictionaries
        return df_logs.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export logs: {str(e)}")