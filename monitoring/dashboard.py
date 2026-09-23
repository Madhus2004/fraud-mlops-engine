import os
import sqlite3
import json
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from scipy.stats import ks_2samp


# ==========================================
# PAGE CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(
    page_title="MLOps Fraud Engine & Governance Dashboard",
    page_icon="🛡️",
    layout="wide"
)

DB_PATH = os.getenv("DB_PATH", "fraud_logs.db")
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
BASELINE_PATH = os.getenv("BASELINE_PATH", "data/processed/reference_baseline.parquet")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_db_logs():
    """Fetch all logged inference requests from SQLite database."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM inference_logs ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error reading database: {e}")
        return pd.DataFrame()

# ==========================================
# DASHBOARD UI HEADER
# ==========================================
st.title("🛡️ Enterprise Fraud Detection & Governance Control Panel")
st.markdown("""
This dashboard monitors real-time FastAPI inference streams, computes **Kolmogorov-Smirnov (KS)** feature distribution drift against baseline sets, and executes candidate model evaluation gates ($v2$ vs $v1$).
""")

# Define 4 Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Database Logs", 
    "🔍 Feature Drift Engine", 
    "⚙️ Retrain Circuit Breaker", 
    "🧪 Test Single Prediction"
])

# ==========================================
# TAB 1: LIVE DATABASE LOGS
# ==========================================
with tab1:
    st.header("📊 Real-Time Ingestion Logs")
    df_logs = load_db_logs()

    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    
    if not df_logs.empty:
        total_txns = len(df_logs)
        flagged_txns = df_logs["is_flagged"].sum() if "is_flagged" in df_logs.columns else 0
        labeled_txns = df_logs["actual_label"].notnull().sum() if "actual_label" in df_logs.columns else 0

        col_metrics1.metric("Total Ingested Logs", f"{total_txns:,}")
        col_metrics2.metric("Flagged High-Risk", f"{flagged_txns:,}", f"{(flagged_txns/total_txns)*100:.1f}%")
        col_metrics3.metric("Labeled Chargebacks", f"{labeled_txns:,}")

        st.subheader("Raw SQLite Ingestion Table")
        st.dataframe(df_logs, width="stretch")
    else:
        st.warning("No records found in `fraud_logs.db`. Run `simulator.py` or use Tab 4 to generate test traffic.")

# ==========================================
# TAB 2: STATISTICAL DRIFT ENGINE (KS TEST)
# ==========================================
with tab2:
    st.header("🔍 Feature Population Drift Detection (KS-Test)")
    st.write("Compares distributions of live SQL traffic features against reference baseline distributions.")

    if st.button("Run Population KS-Drift Check"):
        df_logs = load_db_logs()
        
        if df_logs.empty:
            st.warning("Cannot calculate drift: Live database is empty.")
        elif not os.path.exists(BASELINE_PATH):
            st.error(f"Reference baseline file not found at `{BASELINE_PATH}`.")
        else:
            try:
                # Unpack JSON features from SQLite logs
                feature_list = [json.loads(row) for row in df_logs["features_json"]]
                df_live_features = pd.DataFrame(feature_list)
                df_baseline = pd.read_parquet(BASELINE_PATH)

                drift_results = []
                features_to_check = [c for c in df_baseline.columns if c in df_live_features.columns and c not in ["Time", "Class"]]

                for col in features_to_check:
                    stat, p_val = ks_2samp(df_baseline[col].dropna(), df_live_features[col].dropna())
                    drift_detected = p_val < 0.05
                    drift_results.append({
                        "Feature": col,
                        "KS Statistic": round(stat, 4),
                        "p-value": round(p_val, 5),
                        "Drift Alert": "🚨 DETECTED" if drift_detected else "✅ Normal"
                    })

                df_drift = pd.DataFrame(drift_results)
                st.dataframe(df_drift, width="stretch")

                drift_count = sum(1 for r in drift_results if "🚨" in r["Drift Alert"])
                if drift_count > 0:
                    st.error(f"🚨 Population Drift Alert: {drift_count} features show significant statistical distribution shift!")
                else:
                    st.success("✅ All feature distributions match baseline reference data.")

            except Exception as e:
                st.error(f"Failed to execute drift test: {e}")

# ==========================================
# TAB 3: RETRAIN CIRCUIT BREAKER GATE
# ==========================================
with tab3:
    st.header("⚙️ Automated Model Retraining & Circuit Breaker")
    st.write(
        "Triggers candidate XGBoost (v2) model training on accumulated feedback logs "
        "and enforces pre-deployment performance promotion criteria."
    )

    if st.button("🚀 Execute Retraining Workflow"):
        with st.spinner("Training candidate v2 model and evaluating holdout PR-AUC..."):
            try:
                # Send HTTP POST request to FastAPI endpoint
                response = requests.post(f"{FASTAPI_URL}/retrain", timeout=120)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("promoted"):
                        st.success("🎉 Candidate Model (v2) OUTPERFORMED Active Model (v1) & Was Promoted to Production!")
                    else:
                        st.warning("⚠️ Circuit Breaker Triggered: Candidate v2 did not beat active v1. Active model retained.")

                    # Metric summary cards
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Promotion Status", data.get("promotion_gate", "N/A"))
                    col2.metric("Candidate PR-AUC", data.get("v2_pr_auc_score", 0.0))
                    col3.metric("Optimal Threshold", data.get("optimal_threshold", 0.5))

                    # Expandable JSON payload details
                    with st.expander("📄 View Full Pipeline Execution Metrics"):
                        st.json(data)
                else:
                    st.error(f"Retrain API failed with status code {response.status_code}: {response.text}")
                    
            except Exception as e:
                st.error(f"Failed to connect to FastAPI endpoint at {FASTAPI_URL}: {e}")

# ==========================================
# TAB 4: TEST SINGLE PREDICTION
# ==========================================
with tab4:
    st.header("🧪 Single Transaction Risk Scorer")
    st.write("Send a real-time transaction payload to the internal FastAPI microservice (`POST /predict`).")

    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amount = st.number_input("Transaction Amount ($)", value=149.62, min_value=0.0)
            v1 = st.number_input("V1 Feature", value=-1.3598)
            v2 = st.number_input("V2 Feature", value=-0.0727)
            v3 = st.number_input("V3 Feature", value=2.5363)
            
        with col2:
            v4 = st.number_input("V4 Feature", value=1.3781)
            v5 = st.number_input("V5 Feature", value=-0.3383)
            v6 = st.number_input("V6 Feature", value=0.4623)
            v7 = st.number_input("V7 Feature", value=0.2395)

        with col3:
            v8 = st.number_input("V8 Feature", value=0.0986)
            v9 = st.number_input("V9 Feature", value=0.3637)
            v10 = st.number_input("V10 Feature", value=0.0907)
            v11 = st.number_input("V11 Feature", value=-0.5516)

        submit_button = st.form_submit_button("Evaluate Risk Score")

    if submit_button:
        # Construct exact feature JSON required by FastAPI schema
        payload = {
            "Time": 100,
            "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5,
            "V6": v6, "V7": v7, "V8": v8, "V9": v9, "V10": v10,
            "V11": v11, "V12": -0.6178, "V13": -0.9913, "V14": -0.3111,
            "V15": 1.4681, "V16": -0.4704, "V17": 0.2079, "V18": 0.0257,
            "V19": 0.4039, "V20": 0.2514, "V21": -0.0183, "V22": 0.2778,
            "V23": -0.1104, "V24": 0.0669, "V25": 0.1285, "V26": -0.1891,
            "V27": 0.1335, "V28": -0.0210,
            "Amount": amount
        }

        try:
            # Internal call to FastAPI on port 8000 within the container
            response = requests.post(f"{FASTAPI_URL}/predict", json=payload, timeout=5)
            
            if response.status_code == 200:
                # 1. Store response in session_state to survive the rerun
                st.session_state["last_prediction"] = response.json()
                
                # 2. Trigger instant rerun so Tab 1 loads the updated SQL table immediately
                st.rerun()
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")

        except Exception as e:
            st.error(f"Failed to connect to internal FastAPI server at `{FASTAPI_URL}`: {e}")

    # Render saved prediction output if available in session_state
    if "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]
        
        st.success("✅ Real-Time Inference Complete!")
        
        # Display output cards
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Transaction ID", f"{result['txn_id'][:8]}...")
        res_col2.metric("Calculated Risk Score", f"{result['risk_score']:.4f}")
        
        flag_status = "🚨 FLAGGED FOR FRAUD" if result['is_flagged'] else "✅ APPROVED"
        res_col3.metric("Decision Output", flag_status)

        st.subheader("FastAPI Response JSON Payload")
        st.json(result)
        st.info("💡 Transaction logged to `fraud_logs.db`. Check Tab 1 to view the updated table!")