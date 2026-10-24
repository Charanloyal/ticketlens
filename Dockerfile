# Multi-stage production Dockerfile for TicketLens IT Ticket Classifier
# Compatible with Docker Desktop, AWS ECS, GCP Cloud Run, and Render Cloud
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY data/fixtures/ /app/data/fixtures/
COPY pyproject.toml .

RUN pip install --no-cache-dir -e .

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/artifacts/models /app/reports /app/data && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn ticketlens.api:app --host 0.0.0.0 --port ${PORT}"]
