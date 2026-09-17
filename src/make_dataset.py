import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define file paths
RAW_DATA_PATH = "data/raw/creditcard.csv"
PROCESSED_DIR = "data/processed"
REFERENCE_PATH = os.path.join(PROCESSED_DIR, "reference_baseline.parquet")
HOLDOUT_PATH = os.path.join(PROCESSED_DIR, "holdout_test.parquet")
SCALER_PATH = "app/models/scaler.pkl"

def prepare_data():
    print("Loading raw Kaggle credit card dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 1. Split BEFORE any scaling (Checklist Rule #2)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    
    # 70% Train/Validation (used for initial model + reference baseline)
    # 30% Streaming Pool + Holdout
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    # Split the 30% temp data into Streaming Pool (20%) and Holdout Test (10%)
    X_stream, X_holdout, y_stream, y_holdout = train_test_split(
        X_temp, y_temp, test_size=0.3333, random_state=42, stratify=y_temp
    )
    
    print(f"Train Set Shape: {X_train.shape} (Fraud ratio: {y_train.mean():.4%})")
    print(f"Holdout Set Shape: {X_holdout.shape} (Fraud ratio: {y_holdout.mean():.4%})")
    
    # 2. Fit scaler ONLY on the training set (No Data Leakage)
    scaler = StandardScaler()
    
    # Scale Amount and Time features
    scale_cols = ["Amount", "Time"]
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_holdout[scale_cols] = scaler.transform(X_holdout[scale_cols])
    X_stream[scale_cols] = scaler.transform(X_stream[scale_cols])
    
    # Ensure directories exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs("app/models", exist_ok=True)
    
    # Save the fitted scaler artifact
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved fitted scaler to {SCALER_PATH}")
    
    # Save reference baseline for Evidently AI (Train set with labels)
    train_df = X_train.copy()
    train_df["Class"] = y_train
    train_df.to_parquet(REFERENCE_PATH, index=False)
    print(f"Saved reference baseline to {REFERENCE_PATH}")
    
    # Save holdout test set for model evaluation gate
    holdout_df = X_holdout.copy()
    holdout_df["Class"] = y_holdout
    holdout_df.to_parquet(HOLDOUT_PATH, index=False)
    print(f"Saved holdout test set to {HOLDOUT_PATH}")
    
    # Save stream pool (for simulator)
    stream_df = X_stream.copy()
    stream_df["Class"] = y_stream
    stream_df.to_parquet(os.path.join(PROCESSED_DIR, "stream_pool.parquet"), index=False)
    print("Saved stream pool data successfully.")

if __name__ == "__main__":
    prepare_data()