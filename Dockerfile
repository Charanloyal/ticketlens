# Production Dockerfile for TicketLens IT Ticket Classifier
# Mirrors the Render setup: lightweight serving stack (no torch) and the TF-IDF model is
# trained at image build time, so the container starts in seconds and runs in ~220 MB.
# Compatible with Docker Desktop, AWS ECS, GCP Cloud Run, and Render (Docker runtime).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app \
    TICKETLENS_ENV=production \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-render.txt .
RUN pip install --upgrade pip && pip install -r requirements-render.txt

COPY configs/ /app/configs/
COPY data/fixtures/ /app/data/fixtures/
COPY reports/ /app/reports/
COPY src/ /app/src/
COPY main.py ./

# Download the dataset and train the serving model during the build (fails the build if the
# dataset can't be downloaded), then drop the raw data to keep the image small.
RUN python -m ticketlens.build \
    && rm -rf /app/data/raw /app/data/processed /root/.cache/huggingface

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
