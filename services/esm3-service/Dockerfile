FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install ESM3 and related packages
RUN pip install fair-esm torch torchvision biotite biopython

# Copy application code
COPY . .
COPY ../../shared ./shared

# Create directories for models and data
RUN mkdir -p /app/models/esm3 /app/data /app/logs

# Create non-root user
RUN useradd --create-home --shell /bin/bash evohuman
RUN chown -R evohuman:evohuman /app
USER evohuman

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "main.py"]
