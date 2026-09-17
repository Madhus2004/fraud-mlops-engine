import time
import requests
import pandas as pd
import numpy as np

API_URL = "http://127.0.0.1:8000"
STREAM_DATA_PATH = "data/processed/stream_pool.parquet"

def run_simulation(num_batches=5, batch_size=20, inject_drift=False):
    print("--- Starting Transaction Stream Simulator ---")
    df_stream = pd.read_parquet(STREAM_DATA_PATH)
    
    total_processed = 0
    
    for batch_idx in range(num_batches):
        print(f"\n[Batch {batch_idx + 1}/{num_batches}] Streaming {batch_size} transactions...")
        
        # Sample batch from stream pool
        sample_df = df_stream.sample(n=batch_size).copy()
        
        # Inject Synthetic Feature Drift if requested (Simulates adversarial shift)
        if inject_drift:
            print("ALERT: Injecting synthetic fraud drift into amount & transaction frequency!")
            sample_df["Amount"] = sample_df["Amount"] * np.random.uniform(5.0, 15.0, size=len(sample_df))
            sample_df["V1"] = sample_df["V1"] - 4.0
            
        pending_feedback = []
        
        # 1. Send Real-Time Inferences (/predict)
        for _, row in sample_df.iterrows():
            payload = row.drop("Class").to_dict()
            actual_label = int(row["Class"])
            
            try:
                res = requests.post(f"{API_URL}/predict", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    txn_id = data["txn_id"]
                    pending_feedback.append((txn_id, actual_label))
                    total_processed += 1
            except Exception as e:
                print(f"Error connecting to API: {e}")
                
        print(f"Logged {len(pending_feedback)} inferences to database.")
        
        # 2. Simulate 30-Day Ground-Truth Delay (10 seconds)
        print("Waiting 10 seconds (Simulating 30-day chargeback label delay)...")
        time.sleep(10)
        
        # 3. Post Ground-Truth Labels (/feedback)
        print("Sending ground-truth feedback labels to database...")
        for txn_id, actual_label in pending_feedback:
            fb_payload = {"txn_id": txn_id, "actual_label": actual_label}
            requests.post(f"{API_URL}/feedback", json=fb_payload)
            
        print("Feedback batch update complete.")

if __name__ == "__main__":
    # Test run: 2 normal batches, then 1 drifted batch
    run_simulation(num_batches=2, batch_size=20, inject_drift=False)
    run_simulation(num_batches=2, batch_size=20, inject_drift=True)