import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import ks_2samp

DB_PATH = "fraud_logs.db"
REFERENCE_DATA_PATH = "data/processed/reference_baseline.parquet"

def fetch_recent_logs(hours=24):
    """Fetches inference logs from SQLite from the last N hours."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
        
    conn = sqlite3.connect(DB_PATH)
    time_limit = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    
    query = f"SELECT * FROM transaction_logs WHERE timestamp >= '{time_limit}'"
    df_logs = pd.read_sql_query(query, conn)
    conn.close()
    
    if df_logs.empty:
        return pd.DataFrame()
        
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    available_cols = [col for col in feature_cols if col in df_logs.columns]
    return df_logs[available_cols]

def calculate_drift(alpha=0.05, threshold_share=0.3):
    """
    Computes distribution drift between baseline reference data and recent live logs 
    using the Kolmogorov-Smirnov (KS) statistical test.
    """
    print("--- Running Statistical Data Drift Analysis (KS Test) ---")
    
    if not os.path.exists(REFERENCE_DATA_PATH):
        raise FileNotFoundError(f"Reference baseline file not found at: {REFERENCE_DATA_PATH}")
        
    reference_df = pd.read_parquet(REFERENCE_DATA_PATH)
    current_df = fetch_recent_logs(hours=24)
    
    if current_df.empty or len(current_df) < 5:
        return {
            "drift_detected": False,
            "drift_share": 0.0,
            "drifted_columns_count": 0,
            "sample_size": len(current_df),
            "message": "Insufficient live log samples for drift calculation."
        }
        
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    
    drifted_features = []
    
    for col in feature_cols:
        if col in reference_df.columns and col in current_df.columns:
            ref_data = reference_df[col].dropna()
            cur_data = current_df[col].dropna()
            
            # Perform Kolmogorov-Smirnov 2-sample test
            stat, p_value = ks_2samp(ref_data, cur_data)
            
            # If p-value < alpha (0.05), reject H0 -> distributions are significantly different
            if p_value < alpha:
                drifted_features.append(col)
                
    drifted_count = len(drifted_features)
    drift_share = drifted_count / len(feature_cols)
    dataset_drift_detected = drift_share >= threshold_share
    
    return {
        "drift_detected": dataset_drift_detected,
        "drift_share": round(float(drift_share), 4),
        "drifted_columns_count": drifted_count,
        "drifted_features": drifted_features,
        "sample_size": len(current_df)
    }

if __name__ == "__main__":
    results = calculate_drift()
    print(results)