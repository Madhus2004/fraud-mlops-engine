import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from xgboost import XGBClassifier

# File Paths
REFERENCE_DATA_PATH = "data/processed/reference_baseline.parquet"
MODEL_SAVE_PATH = "app/models/xgboost_v1.pkl"
METRICS_SAVE_PATH = "app/models/metrics_v1.json"


def find_optimal_threshold(y_true, y_probs, min_recall=0.80):
    """Finds the decision threshold that maximizes Precision while maintaining a minimum Recall floor."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    optimal_threshold = 0.5  # Default fallback
    best_precision = 0.0
    best_recall = 0.0

    # Iterate over precision/recall/threshold pairs
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if r >= min_recall and p > best_precision:
            best_precision = float(p)
            best_recall = float(r)
            optimal_threshold = float(t)

    # Fallback if no threshold meets the minimum recall criteria
    if best_precision == 0.0:
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = float(thresholds[best_idx])
        best_precision = float(precisions[best_idx])
        best_recall = float(recalls[best_idx])

    return optimal_threshold, best_precision, best_recall


def train_baseline_model():
    print("--- Phase 1: Training Initial XGBoost Model (v1) ---")

    if not os.path.exists(REFERENCE_DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {REFERENCE_DATA_PATH}. Run 'python src/make_dataset.py' first."
        )

    # 1. Load Preprocessed Training Baseline
    df = pd.read_parquet(REFERENCE_DATA_PATH)
    X_train = df.drop(columns=["Class"])
    y_train = df["Class"]

    # 2. Calculate Class Imbalance Weight (Checklist Rule #3)
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos
    print(
        f"Dataset loaded. Total rows: {len(df)} | Fraud cases: {num_pos} | Non-Fraud cases: {num_neg}"
    )
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # 3. Instantiate XGBoost optimized for PR-AUC (Checklist Rule #1)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    # 4. Train Model
    print("Fitting XGBoost Classifier...")
    model.fit(X_train, y_train)

    # 5. Predict Probabilities & Calculate PR-AUC Score
    y_probs = model.predict_proba(X_train)[:, 1]
    pr_auc = average_precision_score(y_train, y_probs)
    print(f"\n[Training PR-AUC Score]: {pr_auc:.4f}")

    # 6. Tune Decision Threshold (Checklist Rule #4 & #5)
    optimal_thresh, precision, recall = find_optimal_threshold(
        y_train, y_probs, min_recall=0.80
    )
    print(f"[Optimal Decision Threshold]: {optimal_thresh:.4f}")
    print(
        f"[Metrics at Threshold]: Precision = {precision:.4f} | Recall = {recall:.4f}"
    )

    # 7. Save Model Artifact & Metadata Metrics
    os.makedirs("app/models", exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\nModel artifact saved successfully to {MODEL_SAVE_PATH}")

    metrics_payload = {
        "model_version": "v1",
        "pr_auc": round(float(pr_auc), 4),
        "optimal_threshold": round(float(optimal_thresh), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "train_samples": int(len(X_train)),
        "fraud_samples": int(num_pos),
    }

    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=4)

    print(f"Metrics saved successfully to {METRICS_SAVE_PATH}")


if __name__ == "__main__":
    train_baseline_model()