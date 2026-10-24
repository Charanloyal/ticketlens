.PHONY: help install train-quick train-full evaluate explain test api docker-build docker-run clean

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
	@echo "  make train-full     Run full training run (baseline + DistilBERT LoRA)"
	@echo "  make evaluate       Evaluate baseline and transformer on test set"
	@echo "  make explain        Generate SHAP / feature importance & bias report"
	@echo "  make test           Run unit and API test suite"
	@echo "  make api            Start the FastAPI inference service"
	@echo "  make docker-build   Build the Docker container image"
	@echo "  make docker-run     Run the Docker container image on port 8000"
	@echo "  make clean          Remove cache files and temporary artifacts"

install:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

train-quick:
	$(PYTHON) -m ticketlens.train_baseline --mode quick
	$(PYTHON) -m ticketlens.train_transformer --mode quick

train-full:
	$(PYTHON) -m ticketlens.train_baseline --mode full
	$(PYTHON) -m ticketlens.train_transformer --mode full

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

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
