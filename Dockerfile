FROM python:3.10-slim

WORKDIR /app

# Install essential system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt-get/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code, model artifacts, reference baseline, and entrypoint
COPY app/ app/
COPY fraud_logs.db .
COPY src/ src/
COPY monitoring/ monitoring/
COPY data/processed/reference_baseline.parquet data/processed/reference_baseline.parquet
COPY entrypoint.sh .

# Ensure entrypoint script has Linux execution permissions
RUN chmod +x entrypoint.sh

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Execute entrypoint script
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]