import pandas as pd
import requests
import time
import os
API_URL = "http://127.0.0.1:8000/predict"

STREAM_DATA_PATH = "data/processed/stream_pool.parquet"

def run_simulation(num_batches=2, batch_size=20, inject_drift=False):
    print("--- Starting Transaction Stream Simulator ---")
    
    if not os.path.exists(STREAM_DATA_PATH):
        print(f"❌ Error: Cannot find data file at {STREAM_DATA_PATH}")
        return

    df_stream = pd.read_parquet(STREAM_DATA_PATH)
    
    for i in range(num_batches):
        sample_df = df_stream.sample(n=batch_size).copy()
        
        if inject_drift:
            print("ALERT: Injecting synthetic fraud drift into amount & features!")
            sample_df["V1"] = sample_df["V1"] * 5.0
            sample_df["Amount"] = sample_df["Amount"] * 10.0

        records = sample_df.to_dict(orient="records")
        print(f"\n[Batch {i+1}/{num_batches}] Streaming {len(records)} transactions to {API_URL}...")

        logged_count = 0
        for record in records:
            # 1. Strip target labels and auto-generated indexes
            payload = {k: v for k, v in record.items() if k not in ["Class", "index", "Unnamed: 0"]}
            
            try:
                response = requests.post(API_URL, json=payload, timeout=5)
                if response.status_code == 200:
                    logged_count += 1
                else:
                    print(f"❌ API Error ({response.status_code}): {response.text[:100]}")
            except Exception as e:
                print(f"❌ Connection Error: {e}")
                break  # Stop batch if server is unreachable
                
        print(f"Logged {logged_count}/{batch_size} inferences to database.")
        time.sleep(2)

if __name__ == "__main__":
    run_simulation(num_batches=2, batch_size=20, inject_drift=False)