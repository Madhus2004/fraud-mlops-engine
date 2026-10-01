#!/bin/bash
set -e
python -m core.db

echo "Starting FastAPI on :8000 ..."
uvicorn serving.main:app --host 0.0.0.0 --port 8000 &

echo "Waiting for API ..."
for i in $(seq 1 60); do
  python - <<'PY' && break || sleep 1
import sys, urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
except urllib.error.HTTPError:
    pass  # 503 = up but no model yet; dashboard still works
except Exception:
    sys.exit(1)
PY
done

PORT="${PORT:-8501}"
echo "Starting dashboard on :${PORT} ..."
exec streamlit run serving/dashboard.py --server.port="${PORT}" --server.address=0.0.0.0 --server.headless=true
