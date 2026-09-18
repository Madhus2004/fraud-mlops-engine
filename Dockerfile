FROM python:3.10-slim

WORKDIR /app

# Install essential system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt-get/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and entrypoint script
COPY app/ app/
COPY src/ src/
COPY monitoring/ monitoring/
COPY entrypoint.sh .

# Create runtime directory shells
RUN mkdir -p data/processed app/models

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Execute entrypoint script to launch both services
CMD ["/bin/bash", "entrypoint.sh"]