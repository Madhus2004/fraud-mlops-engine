#!/bin/bash

# 1. Initialize DB schema
python src/db.py

# 2. Start FastAPI in the background
echo "Starting FastAPI server on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# 3. Wait for FastAPI to become responsive (up to 15 seconds)
echo "Waiting for FastAPI to be ready..."
for i in {1..15}; do
  if curl -s http://127.0.0.1:8000/docs > /dev/null; then
    echo "FastAPI is up and running!"
    break
  fi
  echo "Retrying FastAPI connection ($i/15)..."
  sleep 1
done

# 4. Start Streamlit in the foreground
echo "Starting Streamlit dashboard on port 8501..."
streamlit run monitoring/dashboard.py --server.port=8501 --server.address=0.0.0.0