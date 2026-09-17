import os
import sqlite3
import pandas as pd
import streamlit as st
import json
import sys

# Fix Python path so Streamlit can locate the 'src' package from root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drift import calculate_drift
from src.retrain import execute_retraining_pipeline

st.set_page_config(page_title="Fraud Engine MLOps Console", layout="wide")

st.title("🛡️ Real-Time Fraud Engine: MLOps Control Console")
st.markdown("Automated Unsupervised Drift Detection & Pre-Deployment Evaluation Gate")

DB_PATH = "fraud_logs.db"
ACTIVE_METRICS_PATH = "app/models/metrics_active.json"
V1_METRICS_PATH = "app/models/metrics_v1.json"

# Sidebar System Health
st.sidebar.header("System Status")
metrics_path = ACTIVE_METRICS_PATH if os.path.exists(ACTIVE_METRICS_PATH) else V1_METRICS_PATH

if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    st.sidebar.success(f"Active Model: {metrics.get('model_version', 'v1')}")
    st.sidebar.metric("Baseline PR-AUC", metrics.get("pr_auc", 0.0))
    st.sidebar.metric("Decision Cutoff", metrics.get("optimal_threshold", 0.5))

# Tab Layout
tab1, tab2, tab3 = st.tabs(["📊 Live Database Logs", "🔍 Evidently Drift Analysis", "🚀 Retraining & Evaluation Gate"])

# TAB 1: DB Logs
with tab1:
    st.subheader("Recent Inferences & Delayed Feedback")
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        df_logs = pd.read_sql_query("SELECT txn_id, timestamp, Amount, predicted_score, is_flagged, actual_label, feedback_timestamp FROM transaction_logs ORDER BY timestamp DESC LIMIT 50", conn)
        conn.close()
        
        st.metric("Total Inferences Logged", len(df_logs))
        st.dataframe(df_logs, use_container_width=True)
    else:
        st.info("No transaction logs found in database yet. Run simulator.py to generate traffic.")

# TAB 2: Drift Analysis
with tab2:
    st.subheader("Unsupervised Distribution Drift Check (PSI & KS Test)")
    if st.button("Run Drift Check Now"):
        with st.spinner("Analyzing feature distributions against reference baseline..."):
            drift_results = calculate_drift()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Drift Status", "ALERT: DRIFT DETECTED" if drift_results["drift_detected"] else "NORMAL")
            col2.metric("Drifted Columns Share", f"{drift_results.get('drift_share', 0):.2%}")
            col3.metric("Sample Size", drift_results.get("sample_size", 0))
            
            if drift_results["drift_detected"]:
                st.warning("Significant feature distribution shift detected! Retraining recommended.")
            else:
                st.success("Incoming data matches baseline distribution.")

# TAB 3: Manual Retrain Trigger
with tab3:
    st.subheader("Pre-Deployment Evaluation Gate (Circuit Breaker)")
    st.write("Retrains XGBoost candidate model on fresh DB logs and validates against isolated holdout set before promotion.")
    
    if st.button("Execute Retraining Workflow"):
        with st.spinner("Retraining candidate model v2 and running holdout evaluation gate..."):
            success, results = execute_retraining_pipeline()
            
            if success:
                st.success(f"Model v2 Deployed! PR-AUC Improved to {results['pr_auc']}")
                st.json(results)
            else:
                st.error("Circuit Breaker Activated: Candidate model failed to outperform baseline.")
                st.json(results)