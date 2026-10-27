.PHONY: help install train-quick train-full benchmark evaluate explain test api docker-build docker-run docker-build-transformer docker-run-transformer clean

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
UVICORN := .venv/bin/uvicorn
PYTEST := .venv/bin/pytest

# Windows fallback detection
ifeq ($(OS),Windows_NT)
    PYTHON := .venv/Scripts/python.exe
    PIP := .venv/Scripts/pip.exe
    UVICORN := .venv/Scripts/uvicorn.exe
    PYTEST := .venv/Scripts/pytest.exe
endif

help:
	@echo "TicketLens Automation Commands:"
	@echo "  make install        Install project dependencies into .venv"
	@echo "  make train-quick    Run functional verification training (quick mode)"
	@echo "  make train-full     Train TF-IDF, Keras CNN and DistilBERT LoRA on the benchmark subset"
	@echo "  make benchmark      train-full + evaluate + explain (same test set for every model)"
	@echo "  make evaluate       Benchmark all trained models on the same test set + error analysis"
	@echo "  make explain        Generate SHAP / feature importance & bias report"
	@echo "  make test           Run unit and API test suite"
	@echo "  make api            Start the FastAPI inference service"
	@echo "  make docker-build   Build the Docker container image"
	@echo "  make docker-run     Run the Docker container image on port 8000"
	@echo "  make docker-build-transformer  Build the DistilBERT + LoRA serving image"
	@echo "  make clean          Remove cache files and temporary artifacts"

install:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

train-quick:
	$(PYTHON) -m ticketlens.train_baseline --mode quick --model-name tfidf_benchmark --no-augment
	$(PYTHON) -m ticketlens.train_keras --mode quick
	$(PYTHON) -m ticketlens.train_transformer --mode quick

train-full:
	$(PYTHON) -m ticketlens.build
	$(PYTHON) -m ticketlens.train_baseline --mode full --model-name tfidf_benchmark --no-augment
	$(PYTHON) -m ticketlens.train_keras --mode full
	$(PYTHON) -m ticketlens.train_transformer --mode full

benchmark: train-full evaluate explain

evaluate:
	$(PYTHON) -m ticketlens.evaluate

explain:
	$(PYTHON) -m ticketlens.explain

test:
	$(PYTEST) tests/

api:
	$(UVICORN) ticketlens.api:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t ticketlens:latest .

docker-run:
	docker run -p 8000:8000 --rm ticketlens:latest

docker-build-transformer:
	docker build -f Dockerfile.transformer -t ticketlens:transformer .

docker-run-transformer:
	docker run -p 8000:8000 --rm ticketlens:transformer

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
