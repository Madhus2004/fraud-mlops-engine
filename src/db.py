import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "fraud_logs.db"

def init_db():
    """Creates the SQLite database and table if they do not exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table stores raw features, predictions, and delayed ground truth
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transaction_logs (
        txn_id TEXT PRIMARY KEY,
        timestamp TEXT,
        Amount REAL,
        Time REAL,
        V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL,
        V6 REAL, V7 REAL, V8 REAL, V9 REAL, V10 REAL,
        V11 REAL, V12 REAL, V13 REAL, V14 REAL, V15 REAL,
        V16 REAL, V17 REAL, V18 REAL, V19 REAL, V20 REAL,
        V21 REAL, V22 REAL, V23 REAL, V24 REAL, V25 REAL,
        V26 REAL, V27 REAL, V28 REAL,
        predicted_score REAL,
        is_flagged INTEGER,
        actual_label INTEGER,
        feedback_timestamp TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_prediction(txn_id: str, features: dict, score: float, is_flagged: bool):
    """Logs incoming transaction features and initial inference predictions."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    now = datetime.utcnow().isoformat()
    
    # Extract feature keys dynamically
    feature_keys = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    feature_vals = [features.get(k, 0.0) for k in feature_keys]
    
    query = f"""
    INSERT INTO transaction_logs (
        txn_id, timestamp, {', '.join(feature_keys)}, predicted_score, is_flagged
    ) VALUES (?, ?, {', '.join(['?']*len(feature_keys))}, ?, ?)
    """
    
    params = [txn_id, now] + feature_vals + [float(score), int(is_flagged)]
    cursor.execute(query, params)
    conn.commit()
    conn.close()

def log_feedback(txn_id: str, actual_label: int):
    """Appends ground-truth feedback label to an existing transaction row."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    now = datetime.utcnow().isoformat()
    cursor.execute("""
    UPDATE transaction_logs
    SET actual_label = ?, feedback_timestamp = ?
    WHERE txn_id = ?
    """, (actual_label, now, txn_id))
    
    updated_rows = cursor.rowcount
    conn.commit()
    conn.close()
    return updated_rows > 0

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully!")