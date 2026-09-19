import sqlite3
import os
import json

DB_PATH = os.getenv("DB_PATH", "fraud_logs.db")

def init_db():
    """Ensure the SQLite table schema exists before services read or write."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inference_logs (
        txn_id TEXT PRIMARY KEY,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        features_json TEXT NOT NULL,
        risk_score REAL NOT NULL,
        is_flagged INTEGER NOT NULL,
        actual_label INTEGER DEFAULT NULL
    )
    """)
    conn.commit()
    conn.close()

def log_predict(txn_id: str, features: dict, risk_score: float, is_flagged: bool):
    """Log real-time inference payload and model output."""
    init_db()  # Ensures table exists before inserting
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO inference_logs (txn_id, features_json, risk_score, is_flagged)
    VALUES (?, ?, ?, ?)
    """, (txn_id, json.dumps(features), risk_score, int(is_flagged)))
    conn.commit()
    conn.close()

def log_feedback(txn_id: str, actual_label: int):
    """Update existing transaction log with ground-truth feedback."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE inference_logs
    SET actual_label = ?
    WHERE txn_id = ?
    """, (actual_label, txn_id))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print(f"Database initialized successfully at {DB_PATH}")