"""Streams stream-pool transactions to the API and sends delayed ground-truth feedback.

  python -m ops.simulator --batches 5 --batch-size 40
  python -m ops.simulator --drift             # inject covariate drift
"""
import core
import argparse
import os
import random
import time

import pandas as pd
import requests

from training.config import STREAM_PATH


def run(api_url: str, batches: int, batch_size: int, drift: bool, feedback_rate: float, pause: float):
    df = pd.read_parquet(STREAM_PATH)
    for b in range(batches):
        sample = df.sample(n=batch_size).copy()
        if drift:
            for i in range(1, 11):        # broad covariate shift (~38% of monitored features)
                sample[f"V{i}"] += 2.0
            sample["V1"] *= 3.0
            sample["Amount"] *= 10.0
        ok = fb = 0
        for rec in sample.to_dict(orient="records"):
            label = int(rec.pop("Class"))
            try:
                r = requests.post(f"{api_url}/predict", json=rec, timeout=60)
                if not r.ok:
                    print("API error", r.status_code, r.text[:120])
                    continue
                ok += 1
                if random.random() < feedback_rate:  # chargebacks arrive later; simulated immediately
                    fb += requests.post(f"{api_url}/feedback",
                                        json={"txn_id": r.json()["txn_id"], "actual_label": label},
                                        timeout=30).ok
            except requests.RequestException as exc:
                print("Connection error:", exc)
                return
        print(f"[batch {b + 1}/{batches}] scored={ok} labeled={fb} drift={drift}")
        time.sleep(pause)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default=os.getenv("API_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=30)
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--feedback-rate", type=float, default=0.8)
    ap.add_argument("--pause", type=float, default=1.0)
    a = ap.parse_args()
    run(a.api_url.rstrip("/"), a.batches, a.batch_size, a.drift, a.feedback_rate, a.pause)
