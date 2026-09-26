import os

# Limit thread allocations to prevent RAM duplication across CPU cores
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import sqlite3
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from xgboost import XGBClassifier
import gc

# File Paths
HOLDOUT_DATA_PATH = os.getenv("HOLDOUT_DATA_PATH", "data/processed/holdout_test.parquet")
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "data/processed/reference_baseline.parquet")
DB_PATH = os.getenv("DB_PATH", "fraud_logs.db")

V1_MODEL_PATH = os.getenv("V1_MODEL_PATH", "app/models/xgboost_v1.pkl")
V1_METRICS_PATH = os.getenv("V1_METRICS_PATH", "app/models/metrics_v1.json")

ACTIVE_MODEL_PATH = os.getenv("ACTIVE_MODEL_PATH", "app/models/xgboost_active.pkl")
ACTIVE_METRICS_PATH = os.getenv("ACTIVE_METRICS_PATH", "app/models/metrics_active.json")


def find_optimal_threshold(y_true, y_probs, min_recall=0.80):
    """Finds the decision threshold maximizing Precision while preserving Recall floor."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    optimal_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if r >= min_recall and p > best_precision:
            best_precision = float(p)
            best_recall = float(r)
            optimal_threshold = float(t)

    if best_precision == 0.0:
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = float(thresholds[best_idx])
        best_precision = float(precisions[best_idx])
        best_recall = float(recalls[best_idx])

    return optimal_threshold, best_precision, best_recall


def fetch_feedback_logs_from_db():
    """Extracts inference logs from SQLite that have ground-truth feedback attached."""
    if not os.path.exists(DB_PATH):
        print("ℹ️ No database found yet. Returning empty DataFrame.")
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM inference_logs WHERE actual_label IS NOT NULL"
        df_logs = pd.read_sql_query(query, conn)
        conn.close()
    except Exception:
        try:
            conn = sqlite3.connect(DB_PATH)
            query = "SELECT * FROM transaction_logs WHERE actual_label IS NOT NULL"
            df_logs = pd.read_sql_query(query, conn)
            conn.close()
        except Exception as e:
            print(f"⚠️ Could not read logs from SQLite: {e}")
            return pd.DataFrame()

    if df_logs.empty:
        print("ℹ️ No ground-truth feedback logs available yet.")
        return pd.DataFrame()

    if "features_json" in df_logs.columns:
        features_list = [json.loads(row) for row in df_logs["features_json"]]
        df_features = pd.DataFrame(features_list)
        df_features["Class"] = df_logs["actual_label"].values
    else:
        drop_cols = [c for c in ["txn_id", "timestamp", "predicted_score", "risk_score", "is_flagged", "feedback_timestamp"] if c in df_logs.columns]
        df_features = df_logs.drop(columns=drop_cols)
        df_features.rename(columns={"actual_label": "Class"}, inplace=True)

    return df_features


def execute_retraining_pipeline():
    print("\n--- Phase 2: Starting Automated Retraining & Pre-Deployment Gate ---")

    # 1. Load Baseline Training Data & Ground-Truth Logs
    ref_df = pd.read_parquet(REFERENCE_DATA_PATH)
    new_logs_df = fetch_feedback_logs_from_db()

    if not new_logs_df.empty:
        print(f"Combining {len(ref_df)} baseline samples with {len(new_logs_df)} new labeled logs.")
        combined_df = pd.concat([ref_df, new_logs_df], ignore_index=True)
    else:
        print("No feedback logs found. Retraining solely on baseline dataset.")
        combined_df = ref_df.copy()

    del ref_df, new_logs_df
    gc.collect()

    MAX_SAMPLES = 20000

    if len(combined_df) > MAX_SAMPLES:
        print(f"Subsampling dataset from {len(combined_df)} to {MAX_SAMPLES} to enforce RAM budget...")
        combined_df = combined_df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

    # Store total_samples before dropping combined_df
    total_samples = len(combined_df)

    # Downcast types immediately
    X_retrain = combined_df.drop(columns=["Class", "index", "Unnamed: 0"], errors="ignore").astype("float32")
    y_retrain = combined_df["Class"].astype("int8")
    
    # Free raw temporary dataframe from memory
    del combined_df
    gc.collect()

    # 2. Fit Candidate Model (v2) with Memory-Efficient Hyperparameters
    num_neg = int((y_retrain == 0).sum())
    num_pos = int((y_retrain == 1).sum())
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

    hyperparams = {
        "n_estimators": 60,            # Conservative tree count for 512MB RAM
        "max_depth": 4,                # Bounded tree depth prevents large matrix allocations
        "learning_rate": 0.05,
        "scale_pos_weight": float(scale_pos_weight),
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": 1,                   # Single thread prevents memory duplication across workers
        "tree_method": "hist"          # Histogram method drastically cuts training RAM
    }

    print("Training candidate model (v2)...")
    model_v2 = XGBClassifier(**hyperparams)
    model_v2.fit(X_retrain, y_retrain)

    # Free training matrices
    del X_retrain, y_retrain
    gc.collect()

    # 3. Load Holdout Test Set for Objective Gate Evaluation
    holdout_df = pd.read_parquet(HOLDOUT_DATA_PATH)
    X_holdout = holdout_df.drop(columns=["Class", "index", "Unnamed: 0"], errors="ignore").astype("float32")
    y_holdout = holdout_df["Class"].astype("int8")

    del holdout_df
    gc.collect()

    # Evaluate Candidate Model (v2) on Holdout
    v2_probs = model_v2.predict_proba(X_holdout)[:, 1]
    v2_pr_auc = float(average_precision_score(y_holdout, v2_probs))
    v2_thresh, v2_prec, v2_rec = find_optimal_threshold(y_holdout, v2_probs)

    print(f"\n[Candidate Model (v2) Holdout PR-AUC]: {v2_pr_auc:.4f}")

    # 4. Evaluate Active Baseline Model (v1) on Holdout
    active_model_path = ACTIVE_MODEL_PATH if os.path.exists(ACTIVE_MODEL_PATH) else V1_MODEL_PATH
    model_v1 = joblib.load(active_model_path)
    
    if hasattr(model_v1, "feature_names_in_"):
        X_holdout_v1 = X_holdout[list(model_v1.feature_names_in_)]
    else:
        X_holdout_v1 = X_holdout

    v1_probs = model_v1.predict_proba(X_holdout_v1)[:, 1]
    v1_pr_auc = float(average_precision_score(y_holdout, v1_probs))

    print(f"[Active Model (v1) Holdout PR-AUC]: {v1_pr_auc:.4f}")

    # Clean up holdout evaluation objects
    del X_holdout, X_holdout_v1, y_holdout, v1_probs, v2_probs, model_v1
    gc.collect()

    # 5. Pre-Deployment Gate Logic (Circuit Breaker)
    if v2_pr_auc > v1_pr_auc:
        print("\n✅ SUCCESS: Candidate model (v2) OUTPERFORMS active model (v1)!")
        print("Promoting v2 to Production...")

        os.makedirs("app/models", exist_ok=True)

        joblib.dump(model_v2, ACTIVE_MODEL_PATH)

        v2_metrics = {
            "model_version": "v2_active",
            "pr_auc": round(v2_pr_auc, 4),
            "v1_pr_auc_baseline": round(v1_pr_auc, 4),
            "optimal_threshold": round(v2_thresh, 4),
            "precision": round(v2_prec, 4),
            "recall": round(v2_rec, 4),
            "total_trained_samples": int(total_samples),
            "hyperparameters": hyperparams,
            "status": "DEPLOYED",
        }

        with open(ACTIVE_METRICS_PATH, "w") as f:
            json.dump(v2_metrics, f, indent=4)

        print(f"Active production model updated at {ACTIVE_MODEL_PATH}")
        
        del model_v2
        gc.collect()
        return True, v2_metrics
    else:
        print("\n❌ REJECTED: Candidate model (v2) DID NOT surpass active model (v1).")
        print("Circuit Breaker Triggered: Keeping active model live in production.")
        
        reject_summary = {
            "status": "REJECTED",
            "v1_pr_auc": round(v1_pr_auc, 4),
            "v2_pr_auc": round(v2_pr_auc, 4),
            "action": "Active model retained"
        }
        
        del model_v2
        gc.collect()
        return False, reject_summary


if __name__ == "__main__":
    execute_retraining_pipeline()