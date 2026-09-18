#!/bin/bash

# Initialize SQLite database schema
python src/db.py

# Start FastAPI microservice in the background on port 8000
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit dashboard on port 8501 (exposed publicly)
streamlit run monitoring/dashboard.py --server.port=8501 --server.address=0.0.0.0