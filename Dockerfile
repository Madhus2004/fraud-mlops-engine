FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app OMP_NUM_THREADS=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt
COPY core ./core
COPY serving ./serving
# Optional: bundle a fallback model. Preferred: set HF_REPO_ID so the API pulls the champion at startup.
COPY artifacts ./artifacts
EXPOSE 8501
CMD ["bash", "serving/entrypoint.sh"]