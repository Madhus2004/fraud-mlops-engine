#!/bin/bash

# 1. Initialize DB schema before booting services
python src/db.py

# 2. Start FastAPI in background on port 8000
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# 3. Start Streamlit as main process on port 8501
streamlit run monitoring/dashboard.py --server.port=8501 --server.address=0.0.0.0