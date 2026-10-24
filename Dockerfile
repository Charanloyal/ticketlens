# Multi-stage/lean production Dockerfile for TicketLens IT Ticket Classifier
FROM python:3.11-slim as base

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source, configs, and fixtures
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY data/fixtures/ /app/data/fixtures/
COPY pyproject.toml .

# Create non-privileged user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/artifacts/models /app/reports /app/data && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "ticketlens.api:app", "--host", "0.0.0.0", "--port", "8000"]
