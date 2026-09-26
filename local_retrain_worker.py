import os
import json
import time
import requests
import sqlite3
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

# Configuration
CLOUD_APP_URL = os.getenv("CLOUD_APP_URL", "https://fraud-mlops-engine.onrender.com")
LOCAL_DB_PATH = "local_fraud_logs.db"
BASELINE_PATH = "data/processed/reference_baseline.parquet"
HOLDOUT_PATH = "data/processed/holdout_test.parquet"
ACTIVE_MODEL_PATH = "app/models/xgboost_active.pkl"
V1_MODEL_PATH = "app/models/xgboost_v1.pkl"
ACTIVE_METRICS_PATH = "app/models/metrics_active.json"

LOG_THRESHOLD_FOR_RETRAIN = 5  # Trigger retraining every 100 new records


import re
import json
import requests
import sqlite3
import pandas as pd

def sync_cloud_logs():
    base_url = CLOUD_APP_URL.rstrip('/')
    endpoint = f"{base_url}/?export=true"
    
    print(f"\n[1/5] Syncing logs from live app ({endpoint})...")
    try:
        res = requests.get(endpoint, timeout=30)
        
        if res.status_code != 200:
            print(f"⚠️ Render HTTP {res.status_code}. Retry in next cycle...")
            return 0

        # Parse output
        raw_text = res.text.strip()
        
        # Handle cases where response text contains JSON
        if raw_text.startswith("[") or raw_text.startswith("{"):
            logs = json.loads(raw_text)
        else:
            # Fallback regex extraction if wrapped in Streamlit text containers
            match = re.search(r'(\[\s*\{.*\}\s*\])', raw_text, re.DOTALL)
            if match:
                logs = json.loads(match.group(1))
            else:
                print("⚠️ Render is warming up or empty array returned.")
                return 0

        if not logs:
            print("ℹ️ No logs found on cloud app.")
            return 0

        # Save synced logs to local SQLite database
        conn = sqlite3.connect(LOCAL_DB_PATH)
        df_new = pd.DataFrame(logs)
        df_new.to_sql("inference_logs", conn, if_exists="replace", index=False)
        conn.close()
        
        print(f"✅ Successfully synced {len(df_new)} total logs into `{LOCAL_DB_PATH}`.")
        return len(df_new)
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return 0


def check_ks_drift(df_live):
    """Executes Kolmogorov-Smirnov distribution drift test against baseline dataset."""
    print("\n[2/5] Running KS Feature Distribution Drift Check...")
    if not os.path.exists(BASELINE_PATH):
        print(f"⚠️ Baseline file missing at {BASELINE_PATH}. Skipping drift check.")
        return False, 0

    df_baseline = pd.read_parquet(BASELINE_PATH)
    
    # Extract features from JSON or columns
    if "features_json" in df_live.columns:
        feature_rows = [json.loads(row) for row in df_live["features_json"]]
        df_live_features = pd.DataFrame(feature_rows)
    else:
        drop_cols = [c for c in ["txn_id", "timestamp", "predicted_score", "risk_score", "is_flagged", "actual_label", "feedback_timestamp"] if c in df_live.columns]
        df_live_features = df_live.drop(columns=drop_cols)

    drift_count = 0
    features_to_check = [c for c in df_baseline.columns if c in df_live_features.columns and c not in ["Time", "Class"]]

    for col in features_to_check:
        stat, p_val = ks_2samp(df_baseline[col].dropna(), df_live_features[col].dropna())
        if p_val < 0.05:
            drift_count += 1

    drift_ratio = drift_count / len(features_to_check) if features_to_check else 0
    print(f"📊 Drift Check Complete: {drift_count}/{len(features_to_check)} features shifted (p < 0.05).")
    
    # Significant drift if >20% of features shifted
    has_drift = drift_ratio >= 0.20
    return has_drift, drift_count


def find_optimal_threshold(y_true, y_probs, min_recall=0.80):
    """Finds optimal decision threshold balancing precision and recall floor."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    optimal_thresh = 0.5
    best_prec = 0.0

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if r >= min_recall and p > best_prec:
            best_prec = float(p)
            optimal_thresh = float(t)

    return optimal_thresh


def evaluate_circuit_breaker():
    """Trains candidate v2 model and compares holdout PR-AUC against active v1 model."""
    print("\n[3/5] Training Candidate Model (v2) on Reference + Local Feedback Logs...")
    
    # 1. Load Baseline Data
    ref_df = pd.read_parquet(BASELINE_PATH)
    
    # Load feedback logs from local SQLite
    conn = sqlite3.connect(LOCAL_DB_PATH)
    df_logs = pd.read_sql_query("SELECT * FROM inference_logs WHERE actual_label IS NOT NULL", conn)
    conn.close()

    if not df_logs.empty:
        if "features_json" in df_logs.columns:
            feat_list = [json.loads(r) for r in df_logs["features_json"]]
            df_feat = pd.DataFrame(feat_list)
            df_feat["Class"] = df_logs["actual_label"].values
        else:
            drop_cols = [c for c in ["txn_id", "timestamp", "predicted_score", "risk_score", "is_flagged", "feedback_timestamp"] if c in df_logs.columns]
            df_feat = df_logs.drop(columns=drop_cols).rename(columns={"actual_label": "Class"})
            
        combined_df = pd.concat([ref_df, df_feat], ignore_index=True)
    else:
        combined_df = ref_df.copy()

    # Prep training matrices
    X_train = combined_df.drop(columns=["Class", "index", "Unnamed: 0"], errors="ignore").astype("float32")
    y_train = combined_df["Class"].astype("int8")

    # Fit XGBoost v2
    num_neg = int((y_train == 0).sum())
    num_pos = int((y_train == 1).sum())
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

    model_v2 = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        tree_method="hist"
    )
    model_v2.fit(X_train, y_train)

    # 2. Evaluate on Holdout Test Set
    print("\n[4/5] Evaluating Candidate v2 vs Active v1 on Holdout Test Set...")
    holdout_df = pd.read_parquet(HOLDOUT_PATH)
    X_holdout = holdout_df.drop(columns=["Class", "index", "Unnamed: 0"], errors="ignore").astype("float32")
    y_holdout = holdout_df["Class"].astype("int8")

    # Evaluate Candidate v2
    v2_probs = model_v2.predict_proba(X_holdout)[:, 1]
    v2_pr_auc = float(average_precision_score(y_holdout, v2_probs))
    v2_thresh = find_optimal_threshold(y_holdout, v2_probs)

    # Evaluate Active v1
    active_path = ACTIVE_MODEL_PATH if os.path.exists(ACTIVE_MODEL_PATH) else V1_MODEL_PATH
    model_v1 = joblib.load(active_path)
    v1_probs = model_v1.predict_proba(X_holdout)[:, 1]
    v1_pr_auc = float(average_precision_score(y_holdout, v1_probs))

    print(f"   📈 Active Model (v1) Holdout PR-AUC:   {v1_pr_auc:.4f}")
    print(f"   🚀 Candidate Model (v2) Holdout PR-AUC: {v2_pr_auc:.4f}")

    # 3. Pre-Deployment Promotion Gate
    print("\n[5/5] Circuit Breaker Promotion Decision:")
    if v2_pr_auc > v1_pr_auc:
        print("🎉 SUCCESS: Candidate v2 OUTPERFORMED active v1!")
        print("Saving candidate model locally...")

        os.makedirs("app/models", exist_ok=True)
        joblib.dump(model_v2, ACTIVE_MODEL_PATH)

        metrics = {
            "model_version": "v2_active",
            "pr_auc": round(v2_pr_auc, 4),
            "v1_pr_auc_baseline": round(v1_pr_auc, 4),
            "optimal_threshold": round(v2_thresh, 4),
            "total_trained_samples": len(combined_df),
            "status": "DEPLOYED"
        }

        with open(ACTIVE_METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"✅ Local model artifact updated. Commit & push `{ACTIVE_MODEL_PATH}` to deploy live!")
        return True
    else:
        print("⚠️ CIRCUIT BREAKER REJECTED: Candidate v2 did not beat active v1. Retaining active model.")
        return False


def run_worker_loop():
    """Main worker loop that monitors log growth and triggers retraining."""
    print("==================================================")
    print("🛡️ LOCAL MLOPS RETRAINING WORKER INITIALIZED")
    print("==================================================")
    
    last_processed_count = 0

    while True:
        total_logs = sync_cloud_logs()
        new_records = total_logs - last_processed_count

        if new_records >= LOG_THRESHOLD_FOR_RETRAIN:
            print(f"\n⚡ Threshold Reached: {new_records} new records accumulated!")
            
            conn = sqlite3.connect(LOCAL_DB_PATH)
            df_live = pd.read_sql_query("SELECT * FROM inference_logs", conn)
            conn.close()

            has_drift, drift_count = check_ks_drift(df_live)

            if has_drift:
                print(f"🚨 Statistical drift detected in {drift_count} features. Executing Retraining Circuit Breaker...")
                evaluate_circuit_breaker()
            else:
                print("✅ Population feature distributions match baseline. Skipping retraining.")

            last_processed_count = total_logs
        else:
            print(f"⏳ Monitoring... Current log growth: {new_records}/{LOG_THRESHOLD_FOR_RETRAIN} new records.")

        # Check cloud app every 30 seconds
        time.sleep(30)


if __name__ == "__main__":
    run_worker_loop()